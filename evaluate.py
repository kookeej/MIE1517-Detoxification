import json
import argparse
import numpy as np
from datetime import datetime

from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# TODO: metric implementation
def calc_sta():
    raise NotImplementedError

def calc_sim():
    raise NotImplementedError

def calc_fl():
    raise NotImplementedError

def main(args):
    outputs = []
    with open(f'outputs/results_{args.output_file_name}.jsonl', 'r') as f:
        for line in f:
            outputs.append(json.loads(line))
            
    ref_corpus = [x['references'] for x in outputs]
    ref_sentence = [x['neutral1'] for x in outputs]
    candidates = [x['generation'] for x in outputs]
    
    if args.split_type == 'model':

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

        ref_corpus = [[tokenizer.tokenize(sent) for sent in ref] for ref in ref_corpus]
        ref_sentence = [tokenizer.tokenize(sent) for sent in ref_sentence]
        candidates = [tokenizer.tokenize(sent) for sent in candidates]
        
    elif args.split_type == 'split':
        
        ref_corpus = [[sent.split(' ') for sent in ref] for ref in ref_corpus]
        ref_sentence = [sent.split(' ') for sent in ref_sentence]
        candidates = [sent.split(' ') for sent in candidates]

    # Calculate BLEU score
    bleu_corpus_score = corpus_bleu(ref_corpus, candidates)
    print(bleu_corpus_score)
    
    return

    # TODO: metric calculation
    # Calculate STA (style accuracy)
    sta_corpus_score = calc_sta(ref_corpus, candidates)
    
    # Calculate SIM (content preservation)
    sim_corpus_score = calc_sim(ref_corpus, candidates)
    
    # Calculate FL (fluency)
    fl_corpus_score = calc_fl(ref_corpus, candidates)
    
    # Aggregate all metrics (J)
    j_corpus_score = None
    
    print("corpus:", bleu_corpus_score)
    
    performance = {
        'base_model_name': args.base_model_name,
        'output_file_name': args.output_file_name,
        'split_type': args.split_type,
        'bleu_corpus': bleu_corpus_score,
        # 'bleu_sentence': bleu_sentence_score,
        'sta_corpus': sta_corpus_score,
        # 'sta_sentence': sta_sentence_score,
        'sim_corpus': sim_corpus_score,
        # 'sim_sentence': sim_sentence_score,
        'fl_corpus': fl_corpus_score,
        # 'fl_sentence': fl_sentence_score,
        'j_corpus': j_corpus_score,
        # 'j_sentence': j_sentence_score
    }
    
    with open('performance.jsonl', 'a') as f:
        json.dump(performance, f, ensure_ascii=False)
        f.write('\n')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--split_type', type=str, default='model')
    parser.add_argument('--output_file_name', type=str, required=True)
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    main(args)