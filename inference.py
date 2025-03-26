import json

import json
import argparse

from peft import PeftConfig, PeftModel
from tqdm import tqdm

import chromadb

import torch
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from dataset import ParadetoxDatasetForEval
from evaluate import evaluate
from train import inference
from utils import set_randomness, similarity_search


def main(args):
    print("\n\n\nInference\n\n\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('data/paradetox/test.json', 'r') as f:
        test = json.load(f)

    # load tokenizer and model
    if 't5' in args.base_model_name:
        raise NotImplementedError
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        if args.model_name is not None:
            print(f"Loading the model from {f'checkpoints/best_{args.model_name}'}...")
            adapter_path = f'checkpoints/best_{args.model_name}/'
            model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
            model = PeftModel.from_pretrained(model, adapter_path)

        else:
            model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True,
                                                         torch_dtype=torch.bfloat16)
    model.to(device)

    examples = generate_few_shots(args.shot_type, test)

    dataset = ParadetoxDatasetForEval(test, prompt_type=args.prompt_type, tokenizer=tokenizer, examples=examples)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    model.eval()

    preds = inference(model, dataloader, args.output_file_name, tokenizer, test, args.prompt_type)
    performance = evaluate(preds, args.base_model_name, args.output_file_name)

    print(performance)
    return performance


def generate_few_shots(shot_type, test):
    if shot_type == 'zs':
        examples = [None for _ in range(len(test))]

    elif shot_type == 'fs':
        examples = [
           [{
                "toxic": "im not gonna sell the fucking picture , i just want to contribute to the fucking article .",
                "neutral": "I'm not going to sell the picture, I just want to contribute to the article.[END]"
            },
            {
                "toxic": "i don t know bout you but that shit never work for me ! lol",
                "neutral": "i dont know about you,but that thing never work for me!lol.[END]"
            },
            {
                "toxic": "dude shut the fuck up talking , and get a life .",
                "neutral": "Dude keep quiet and live[END]"
            }] for _ in range(len(test))
        ]

    elif shot_type == 'ds':

        with open('data/paradetox/train.json', 'r') as f:
            document = json.load(f)

        retrieval = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.PersistentClient(path='./chroma_db')
        chroma_client.delete_collection(name='toxic_sentences')
        collection = chroma_client.get_or_create_collection(name='toxic_sentences')

        existing_ids = set(collection.get()['ids'])
        for idx, entry in enumerate(tqdm(document, desc='Demonstration set generation:...')):
            toxic_sentence = entry['toxic']
            embedding = retrieval.encode(toxic_sentence).tolist()

            if str(idx) in existing_ids:
                continue

            collection.add(
                ids=[str(idx)],
                embeddings=[embedding],
                metadatas=[{'toxic': toxic_sentence, 'neutral': entry['neutral1']}]
            )

        examples = [similarity_search(retrieval, collection, x['toxic'], 3) for x in test]
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")
    return examples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=426)
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--shot_type", type=str, choices=['zs', 'fs', 'ds'], default='zs')
    parser.add_argument('--prompt_type', type=str, default='inst')
    parser.add_argument('--output_file_name', type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args.seed)

    main(args)
