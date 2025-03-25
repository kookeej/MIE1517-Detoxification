import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from nltk.translate.bleu_score import corpus_bleu

import wandb

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model, PeftModel

from dataset import ParadetoxDatasetForTrain, ParadetoxDatasetForEval
from evaluate import evaluate
from utils import similarity_search
from utils import set_randomness

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
from sentence_transformers import SentenceTransformer


def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, criterion, device, logger):
    # train
    model.train()
    loss_list = []
    scaler = GradScaler()

    for idx, batch in enumerate(tqdm(dataloader, desc='Training...')):

        optimizer.zero_grad()

        if isinstance(model, T5ForConditionalGeneration):
            outputs = model(**batch)

            logits = outputs.logits
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
            raise NotImplementedError
        else:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_list.append(loss.item())
        if logger is not None:
            logger.log({'train/loss': loss.item()})
    if logger is not None:
        logger.log({'train/epoch_loss': sum(loss_list) / len(loss_list)})


def generate(dataloader, model, tokenizer, prompt_type):
    model.eval()
    total_preds = []
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for idx, batch in enumerate(tqdm(dataloader, desc='Generating...')):
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                outputs = model.generate(
                    **batch,
                    max_new_tokens=64,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                if prompt_type == 'prev':
                    splitter = 'Neutral comment: '
                elif prompt_type == 'inst':
                    splitter = 'assistant'
                elif prompt_type == 'simple':
                    splitter = 'Neutral comment: '
                else:
                    raise ValueError(f"Invalid prompt type: {prompt_type}")

                generated_text = [x.split(splitter)[-1].strip().split("\n")[0].strip() for x in generated_text]
                total_preds.extend(generated_text)

    return total_preds


def valid_one_epoch(model, dataloader, tokenizer, raw_data, prompt_type):
    total_preds = generate(dataloader, model, tokenizer, prompt_type)

    ref_corpus = [x['references'] for x in raw_data]
    ref_corpus = [[tokenizer.tokenize(sent) for sent in ref] for ref in ref_corpus]
    candidates = [tokenizer.tokenize(sent) for sent in total_preds]
    # ref_corpus = [[sent.split(' ') for sent in ref] for ref in ref_corpus]
    bleu_score = corpus_bleu(ref_corpus, candidates)

    return bleu_score, total_preds


def inference(model, dataloader, output_file_name, tokenizer, raw_data, prompt_type):
    total_preds = generate(dataloader, model, tokenizer, prompt_type)
    total_output = []

    for i in range(len(total_preds)):
        x = raw_data[i].copy()
        x['generation'] = total_preds[i]

        with open(f'outputs/results_{output_file_name}.jsonl', 'a', encoding='utf-8') as f:
            json.dump(x, f, ensure_ascii=False)
            f.write("\n")
        total_output.append(x)

    print(f"Saving the output to outputs/results_{output_file_name}.jsonl")
    return total_output


def main(args):
    print("\n\n\nTrain\n\n\n")

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    with open('data/paradetox/train.json', 'r') as f:
        train = json.load(f)
    with open('data/paradetox/valid.json', 'r') as f:
        valid = json.load(f)
    with open('data/paradetox/test.json', 'r') as f:
        test = json.load(f)

    if args.debug:
        train = train[:200]
        valid = train[:10]
        test = test[:10]

    # flatten train
    train = [
        {'toxic': item['toxic'], 'neutral': ref}
        for item in train
        for ref in item['references']
    ]

    train_examples, val_examples, test_examples = None, None, None
    if args.use_demo_selection:
        print("Building DS example list for training...")
        retrieval = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.PersistentClient(path='./chroma_db')
        chroma_client.delete_collection(name='train_ds_collection')
        collection = chroma_client.get_or_create_collection(name='train_ds_collection')
        existing_ids = set(collection.get()['ids'])

        for idx, entry in enumerate(tqdm(train)):
            embedding = retrieval.encode(entry['toxic']).tolist()
            if str(idx) in existing_ids:
                continue
            collection.add(
                ids=[str(idx)],
                embeddings=[embedding],
                metadatas=[{'toxic': entry['toxic'], 'neutral': entry['neutral']}]
            )
        train_examples = [similarity_search(retrieval, collection, ex['toxic'], k=3, query_id=str(idx)) for idx, ex in
                          tqdm(enumerate(train), desc="Building DS example list for training...")]
        print(train_examples[0])
        val_examples = [similarity_search(retrieval, collection, ex['toxic'], k=3) for ex in
                        tqdm(valid, desc="Building DS example list for validation...")]
        test_examples = [similarity_search(retrieval, collection, ex['toxic'], k=3) for ex in
                         tqdm(test, desc="Building DS example list for testing...")]

    if 't5' in args.base_model_name:
        tokenizer = T5Tokenizer.from_pretrained(args.base_model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
        model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16)
        model.to(device)

        # lora tuning configuration
        for name, param in model.named_parameters():
            param.requires_grad = False
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            target_modules=target_modules,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    train_dataset = ParadetoxDatasetForTrain(train, tokenizer, args.prompt_type, examples=train_examples)
    valid_dataset = ParadetoxDatasetForEval(valid, tokenizer, args.prompt_type, examples=val_examples)
    test_dataset = ParadetoxDatasetForEval(test, tokenizer, args.prompt_type, examples=test_examples)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                  collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=test_dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=len(train_dataloader) * args.epochs,
                                                num_warmup_steps=len(train_dataloader) * 0.1)
    criterion = torch.nn.CrossEntropyLoss()

    patience = 0
    best_valid_score = -1

    if args.wandb:
        logger = wandb.init(project="1517", name=args.output_file_name, config=args)
    else:
        logger = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        patience += 1
        train_one_epoch(model, tokenizer, train_dataloader, optimizer, scheduler, criterion, device, logger)
        valid_score, valid_pred = valid_one_epoch(model, valid_dataloader, tokenizer=tokenizer, raw_data=valid,
                                                  prompt_type=args.prompt_type)
        print(f"Validation BLEU: {valid_score:.4f}")
        if logger is not None:
            logger.log({'valid/bleu': valid_score})
        json.dump(valid_pred, open(f'outputs/valid_pred_{args.output_file_name}_epoch{epoch}.json', 'w'), indent=2,
                  ensure_ascii=False)

        if best_valid_score < valid_score:
            best_valid_score = valid_score

            if isinstance(model, T5ForConditionalGeneration):
                torch.save(model.state_dict(), f'./checkpoints/best_{args.output_file_name}.pth')
            else:
                model.save_pretrained(f'./checkpoints/best_{args.output_file_name}')

            print(f'Best model saved (BLEU: {valid_score:.4f})')

        if patience > 1:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    if 't5' in args.base_model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
        model.load_state_dict(torch.load(f'./checkpoints/best_{args.output_file_name}.pth'))
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16)
        model.load_adapter(Path(f'./checkpoints/best_{args.output_file_name}/'), adapter_name='lora')
        model.to(device)

    outputs = inference(model, test_dataloader, args.output_file_name, tokenizer, test, args.prompt_type)
    performance = evaluate(outputs)

    if logger is not None:
        for k, v in performance:
            logger.log({f'test/{k}': v})


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=426)
    parser.add_argument('--base_model_name', type=str, default='t5-base')
    parser.add_argument('--prompt_type', type=str, required=True)
    parser.add_argument('--output_file_name', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_demo_selection', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)

    main(args)
