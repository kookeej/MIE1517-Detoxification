import json
import argparse
import numpy as np
from datetime import datetime

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import corpus_bleu

from wieting_similarity import *

def wieting_sim(inputs, preds):    
    assert len(inputs) == len(preds)

    sim_model = SimilarityEvaluator('./wieting_similarity/sim.pt', './wieting_similarity/sim.sp.30k.model')

    max_sim_scores = []
    avg_sim_scores = []
    for i in tqdm.tqdm(range(len(preds))):
        sim_score_per_one = []
        for j in range(len(inputs[i])):
            s = sim_model.find_similarity([inputs[i][j]], [preds[i]])
            sim_score_per_one.extend(s)
            
        avg_sim_scores.append(sum(sim_score_per_one) / len(sim_score_per_one))
        max_sim_scores.append(max(sim_score_per_one))

    return np.array(avg_sim_scores), np.array(max_sim_scores)

def calc_sta(candidates):
    results = []
    
    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    for i in tqdm.tqdm(range(0, len(candidates), 64)):
        batch = tokenizer(candidates[i:i + 64], return_tensors='pt', padding=True)
        result = model(**batch)['logits'].argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])
    accuracy_by_sent = results
    sta_mean = np.mean(results)
    
    return accuracy_by_sent, sta_mean

def calc_sim(references, candidates):
    """
    Returns:
        similarity_by_sent
        avg_sim_by_sim
    """
    avg_similarity_by_sent, max_similarity_by_sent = wieting_sim(references, candidates)
    avg_avg_sim_by_sent = avg_similarity_by_sent.mean()
    avg_max_sim_by_sent = max_similarity_by_sent.mean()
    
    return avg_similarity_by_sent, avg_avg_sim_by_sent, max_similarity_by_sent, avg_max_sim_by_sent

def calc_fl(candidates):
    # return cola_acc
    fl_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    fl_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA')

    # 입력 토큰화
    fl_inputs = fl_tokenizer(candidates, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # 모델 예측
    with torch.no_grad():
        fl_outputs = fl_model(**fl_inputs)
        predicted_class = fl_outputs.logits.argmax(dim=1)
    # textattack/roberta-base-CoLA에서는 1이 "acceptable"(유창함)
    cola_acc = predicted_class.sum() / len(predicted_class)
    cola_stats = list(1 - predicted_class)
    
    # # 소프트맥스로 확률값도 계산 (J 스코어 계산용)
    # probs = F.softmax(fl_outputs.logits, dim=1)
    # acceptable_prob = probs[0][1]  # 유창함 클래스의 확률
        
    return cola_stats, cola_acc

def calc_j(accuracy_by_sent, similarity_by_sent, cola_stats, candidates):
    return sum(accuracy_by_sent * similarity_by_sent * cola_stats) / len(candidates)


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

        ref_corpus_tok = [[tokenizer.tokenize(sent) for sent in ref] for ref in ref_corpus]
        ref_sentence_tok = [tokenizer.tokenize(sent) for sent in ref_sentence]
        candidates_tok = [tokenizer.tokenize(sent) for sent in candidates]
        
    elif args.split_type == 'split':
        
        ref_corpus_tok = [[sent.split(' ') for sent in ref] for ref in ref_corpus]
        ref_sentence_tok = [sent.split(' ') for sent in ref_sentence]
        candidates_tok = [sent.split(' ') for sent in candidates]

    # Calculate BLEU score
    bleu_corpus_score = corpus_bleu(ref_corpus_tok, candidates_tok)    

    # Calculate STA (style accuracy)
    accuracy_by_sent, sta_score = calc_sta(candidates)

    # Calculate SIM (content preservation)
    avg_similarity_by_sent, avg_avg_sim_by_sent, max_similarity_by_sent, avg_max_sim_by_sent = calc_sim(ref_corpus, candidates)
    
    # Calculate FL (fluency)
    cola_stats, fl_score = calc_fl(candidates)

    # Aggregate all metrics (J)
    avg_j_score = calc_j(accuracy_by_sent, avg_similarity_by_sent, cola_stats, candidates)
    max_j_score = calc_j(accuracy_by_sent, max_similarity_by_sent, cola_stats, candidates)
        
    performance = {
        'base_model_name': args.base_model_name,
        'output_file_name': args.output_file_name,
        'split_type': args.split_type,
        'bleu_corpus': float(bleu_corpus_score),
        'sta': float(sta_score),
        'avg_sim': float(avg_avg_sim_by_sent),
        'max_sim': float(avg_max_sim_by_sent),
        'fl': float(fl_score),
        'avg_j': float(avg_j_score),
        'max_j': float(max_j_score),
    }
    
    print(performance)
    
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