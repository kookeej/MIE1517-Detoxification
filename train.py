import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from nltk.translate.bleu_score import corpus_bleu

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel

from utils import set_randomness


def train_one_epoch(model, tokenizer, dataloader, optimizer, scheduler, criterion, device):
    # train
    model.train()
    for idx, batch in enumerate(tqdm(dataloader)):
        
        optimizer.zero_grad()
        
        if isinstance(model, T5ForConditionalGeneration):
            outputs = model(**batch)
            
            logits = outputs.logits
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
        else:
            labels = batch['labels']
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
            loss = outputs.loss


        loss.backward()
        optimizer.step()
        scheduler.step()
        
        
def valid_one_epoch(model, dataloader, tokenizer, raw_data, device):
    
    model.eval()
    total_preds = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            outputs = model.generate(
                **batch,
                max_new_tokens=64,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(generated_text)
            total_preds.extend(generated_text)
    
    ref_corpus = [x['references'] for x in raw_data]
    ref_corpus = [[tokenizer.tokenize(sent) for sent in ref] for ref in ref_corpus]
    candidates = [tokenizer.tokenize(sent) for sent in total_preds]
    # ref_corpus = [[sent.split(' ') for sent in ref] for ref in ref_corpus]
    bleu_score = corpus_bleu(ref_corpus, candidates) 
    
    return bleu_score
        
def inference(model, dataloader, output_file_name, tokenizer, raw_data, device):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            outputs = model.generate(
                **batch,
                max_new_tokens=64,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
        
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            total_preds.extend(generated_text)
        
    for i in range(len(total_preds)):
        x = raw_data[i].copy()
        x['generation'] = total_preds[i]
        
        with open(f'outputs/results_{output_file_name}.jsonl', 'a', encoding='utf-8') as f:
            json.dump(x, f, ensure_ascii=False)
            f.write("\n")

    
class ParadetoxDatasetForTrain(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        device,
        max_length: Optional[int] = 128,
        ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if isinstance(self.tokenizer, T5Tokenizer):
            inputs = self.tokenizer.encode_plus(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " + self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer.encode_plus(
                    self.data[idx]['neutral'],
                    max_length=64,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
            )
        else:
            inputs = self.tokenizer(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " + self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            targets = self.tokenizer(
                self.data[idx]['neutral'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        
        for k, v in targets.items():
            if k == 'input_ids':
                inputs['labels'] = v
            
        tokenized = {k: v.squeeze(0).to(self.device) for k, v in inputs.items()}

        return tokenized

class ParadetoxDatasetForEval(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        device,
        max_length: Optional[int] = 128,
        ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if isinstance(self.tokenizer, T5Tokenizer):
            inputs = self.tokenizer.encode_plus(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " + self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
        else:
            inputs = self.tokenizer(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " + self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        
        tokenized = {k: v.squeeze(0).to(self.device) for k, v in inputs.items()}

        return tokenized

def main(args):
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
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side='left', truncation_size='left')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True, torch_dtype=torch.float16)
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

    train_dataset = ParadetoxDatasetForTrain(train, tokenizer, device)
    valid_dataset = ParadetoxDatasetForEval(valid, tokenizer, device)
    test_dataset = ParadetoxDatasetForEval(test, tokenizer, device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_training_steps=len(train_dataloader) * args.epochs,
                                                num_warmup_steps=len(train_dataloader) * 0.1)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_valid_score = -1
    patience = 0
    
    for epoch in range(args.epochs):
        patience += 1
        train_one_epoch(model, tokenizer, train_dataloader, optimizer, scheduler, criterion, device)
        valid_score = valid_one_epoch(model, valid_dataloader, tokenizer=tokenizer, device=device, raw_data=valid)
    
        if best_valid_score < valid_score:
            patience = 0
            best_valid_score = valid_score
            
            if isinstance(model, T5ForConditionalGeneration):
                torch.save(model.state_dict(), f'./checkpoints/best_{args.output_file_name}.pth')
            else:
                model.save_pretrained(f'./checkpoints/best_{args.output_file_name}')
                
            print(f'Best model saved (BLEU: {valid_score:.4f})')
            
        if patience > 10:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    if 't5' in args.base_model_name:
        model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
        model.load_state_dict(torch.load(f'./checkpoints/best_{args.output_file_name}.pth'))
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True, torch_dtype=torch.float16)
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
    
    args = parser.parse_args()
    
    return args
    
      

if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)
    
    main(args)