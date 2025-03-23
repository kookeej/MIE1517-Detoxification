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
from utils import set_randomness
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.log({'train/loss': loss.item()})
        loss_list.append(loss.item())

    logger.log({'train/epoch_loss': sum(loss_list) / len(loss_list)})


def generate(dataloader, model, tokenizer):
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
                generated_text = [x.split('Neutral comment: ')[-1].strip().split("\n")[0].strip() for x in generated_text]

                total_preds.extend(generated_text)

    return total_preds

def valid_one_epoch(model, dataloader, tokenizer, raw_data, device):
    total_preds = generate(dataloader, model, tokenizer)

    ref_corpus = [x['references'] for x in raw_data]
    ref_corpus = [[tokenizer.tokenize(sent) for sent in ref] for ref in ref_corpus]
    candidates = [tokenizer.tokenize(sent) for sent in total_preds]
    # ref_corpus = [[sent.split(' ') for sent in ref] for ref in ref_corpus]
    bleu_score = corpus_bleu(ref_corpus, candidates)

    return bleu_score, total_preds


def inference(model, dataloader, output_file_name, tokenizer, raw_data, device):
    total_preds = generate(dataloader, model, tokenizer)

    for i in range(len(total_preds)):
        x = raw_data[i].copy()
        x['generation'] = total_preds[i]

        with open(f'outputs/results_{output_file_name}.jsonl', 'a', encoding='utf-8') as f:
            json.dump(x, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saving the output to outputs/results_{output_file_name}.jsonl")

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

    train_dataset = ParadetoxDatasetForTrain(train, tokenizer)
    valid_dataset = ParadetoxDatasetForEval(valid, tokenizer)
    test_dataset = ParadetoxDatasetForEval(test, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=len(train_dataloader) * args.epochs,
                                                num_warmup_steps=len(train_dataloader) * 0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_valid_score = -1
    patience = 0

    logger = wandb.init(project="1517", name=args.output_file_name, config=args)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        patience += 1
        train_one_epoch(model, tokenizer, train_dataloader, optimizer, scheduler, criterion, device, logger)
        valid_score, valid_pred = valid_one_epoch(model, valid_dataloader, tokenizer=tokenizer, device=device, raw_data=valid)
        print(f"Validation BLEU: {valid_score:.4f}")
        logger.log({'valid/bleu': valid_score})
        json.dump(valid_pred, open(f'outputs/valid_pred_{args.output_file_name}_epoch{epoch}.json', 'w'), indent=2,
                  ensure_ascii=False)

        if best_valid_score < valid_score:
            patience = 0
            best_valid_score = valid_score

            if isinstance(model, T5ForConditionalGeneration):
                torch.save(model.state_dict(), f'./checkpoints/best_{args.output_file_name}.pth')
            else:
                model.save_pretrained(f'./checkpoints/best_{args.output_file_name}')

            print(f'Best model saved (BLEU: {valid_score:.4f})')

        if patience > 3:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if 't5' in args.base_model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
        model.load_state_dict(torch.load(f'./checkpoints/best_{args.output_file_name}.pth'))
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True,
                                                     torch_dtype=torch.float16)
        model.load_adapter(Path(f'./checkpoints/best_{args.output_file_name}/'), adapter_name='lora')
        model.to(device)

    inference(model, test_dataloader, args.output_file_name, tokenizer, test, device)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=426)
    parser.add_argument('--base_model_name', type=str, default='t5-base')
    parser.add_argument('--output_file_name', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)

    main(args)
