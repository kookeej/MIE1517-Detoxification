import json
import argparse
from tqdm import tqdm
from typing import Optional

import chromadb

import torch
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)

from utils import set_randomness

def similarity_search(retrieval, collection, input_sentence, k):
    input_embedding = retrieval.encode(input_sentence).tolist()
    
    results = collection.query(
        query_embeddings=[input_embedding],
        n_results=k
    )
    documents = results['metadatas'][0]
    
    return documents

class ParadetoxDataset(Dataset):
    def __init__(
        self,
        data,
        template,
        tokenizer,
        device,
        examples=None,
        max_length: Optional[int] = 256,
        ):
        
        self.template = template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.examples = examples

        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.examples is not None:
            selected_examples = [f"Toxic comment: {x['toxic']}\### Response: {x['reference']}[END]" for x in self.examples[idx]]
            input_prompt = self.template.format(toxic_comment=self.data[idx]['toxic'], examples='\n'.join(selected_examples))
        else:
            input_prompt = self.template.format(toxic_comment=self.data[idx]['toxic'])
            
        tokenized = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
            
        tokenized = {k: v.squeeze(0).to(self.device) for k, v in tokenized.items()}

        return tokenized
        
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id
        
    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[:, -1] == self.stop_token_id).any()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('data/paradetox/test.json', 'r') as f:
        test = json.load(f)
        
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side='left')
    tokenizer.add_tokens(['[END]'])
    tokenizer.pad_token = tokenizer.eos_token

    stop_token_id = tokenizer.convert_tokens_to_ids('[END]')
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_token_id)])

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.to(device)
    
    if args.prompt_type == 'zs':
        template = """You are skilled at refining toxic comments into neutral, respectful language. Your task is to review the given toxic comment and convert it into a polite, neutral sentence without changing the original text too much. Do not provide any other explanation. Once you finish, generate [END].

        Toxic comment: {toxic_comment}
        ### Response:"""
        
        examples = None

                
    elif args.prompt_type == 'fs':
        template = """You are skilled at refining toxic comments into neutral, respectful language. Your task is to review the given toxic comment and convert it into a polite, neutral sentence without changing the original text too much. Do not provide any other explanation. Once you finish, generate [END].

        Toxic comment: 'im not gonna sell the fucking picture , i just want to contribute to the fucking article .'
        ### Response: "I'm not going to sell the picture, I just want to contribute to the article.[END]"
        
        Toxic comment: 'i don t know bout you but that shit never work for me ! lol'
        ### Response: "i dont know about you,but that thing never work for me!lol.[END]"
        
        Toxic comment: 'dude shut the fuck up talking , and get a life .'
        ### Response: "Dude keep quiet and live[END]"

        Toxic comment: {toxic_comment}
        ### Response:"""
        
        examples = None
        
    elif args.prompt_type == 'ds':
        template = """You are skilled at refining toxic comments into neutral, respectful language. Your task is to review the given toxic comment and convert it into a polite, neutral sentence without changing the original text too much. Do not provide any other explanation. Once you finish, generate [END].

        {examples}

        Toxic comment: {toxic_comment}
        ### Response:"""
        
        with open('data/paradetox/train.json', 'r') as f:
            document = json.load(f)
        
        retrieval = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.PersistentClient(path='./chroma_db')
        collection = chroma_client.get_or_create_collection(name='toxic_sentences')
        
        for idx, entry in enumerate(tqdm(document)):
            toxic_sentence = entry['toxic']
            embedding = retrieval.encode(toxic_sentence).tolist()
            
            collection.add(
                ids=[str(idx)],
                embeddings=[embedding],
                metadatas=[{'toxic': toxic_sentence, 'reference': entry['neutral1']}]
            )
    
        examples = [similarity_search(retrieval, collection, x['toxic'], 3) for x in test]
        
    dataset = ParadetoxDataset(test, template=template, tokenizer=tokenizer, device=device, examples=examples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
                    
    model.eval()
    
    total_preds = []
    total_gts = []
    
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc='Testing...')):
            tokenized = batch
            
            outputs = model.generate(
                **tokenized,
                max_new_tokens=64,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
                )
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_text = [x.split('### Response:')[-1].strip().split('Toxic comment:')[0].strip().replace('\n', '').replace('#', '').strip() for x in generated_text]
            print(generated_text)
            total_preds.extend(generated_text)

    result_list = []
    for i in range(len(total_preds)):
        x = test[i].copy()
        x['generation'] = total_preds[i]
        
        with open(f'outputs/results_{args.output_file_name}.jsonl', 'a', encoding='utf-8') as f:
            json.dump(x, f, ensure_ascii=False)
            f.write("\n")
    
        

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=426)
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--prompt_type', type=str, default='zs')
    parser.add_argument('--output_file_name', type=str, required=True)
    
    args = parser.parse_args()
    
    return args
    
      

if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)
    
    main(args)