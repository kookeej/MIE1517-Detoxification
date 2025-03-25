from typing import Optional
from transformers import T5Tokenizer
from torch.utils.data import Dataset
import torch

class ParadetoxDatasetForTrain(Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            prompt_type,
            examples=None,
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.prompt_type = prompt_type
        self.examples = examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if isinstance(self.tokenizer, T5Tokenizer):
            inputs = self.tokenizer.encode_plus(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
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

            inputs['labels'] = targets['input_ids']
            tokenized = {k: v.squeeze(0).to(self.device) for k, v in inputs.items()}

            return tokenized

        else:
            toxic = self.data[idx]['toxic']
            neutral = self.data[idx]['neutral']
            shot = build_few_shot_prompt(self.examples[idx], self.prompt_type)

            if self.prompt_type == 'prev':
                prompt = f"Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\n{shot}Toxic comment: {toxic} \nNeutral comment: "
            elif self.prompt_type == "inst":
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an AI assistant tasked with converting toxic comments into polite, neutral sentences.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{shot}Toxic comment: {toxic}\nPlease provide a neutral version of the above comment.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            elif self.prompt_type == "simple":
                prompt = "Toxic comment: " + toxic + "\nNeutral comment: "
            else:
                raise ValueError("prompt_type must be prev, inst or simple")

            full_text = prompt + neutral + self.tokenizer.eos_token

            tokenized = self.tokenizer(
                full_text,
                return_tensors='pt',
            )

            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)

            labels = input_ids.clone()

            prompt_len = len(self.tokenizer(prompt, return_tensors='pt')['input_ids'].squeeze(0))
            labels[:prompt_len] = -100

            tokenized = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            return tokenized

    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        input_ids = left_pad(input_ids, pad_value=self.tokenizer.pad_token_id)
        attention_mask = left_pad(attention_mask, pad_value=0)
        labels = left_pad(labels, pad_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ParadetoxDatasetForEval(Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            prompt_type,
            examples=None,
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.prompt_type = prompt_type
        self.examples = examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if isinstance(self.tokenizer, T5Tokenizer):
            inputs = self.tokenizer.encode_plus(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            raise NotImplementedError
        else:
            toxic = self.data[idx]['toxic']
            shot = build_few_shot_prompt(self.examples[idx], self.prompt_type)

            if self.prompt_type == 'prev':
                prompt = f"Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\n{shot}Toxic comment: {toxic} \nNeutral comment: "
            elif self.prompt_type == "inst":
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an AI assistant tasked with converting toxic comments into polite, neutral sentences.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{shot}Toxic comment: {toxic}\nPlease provide a neutral version of the above comment.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            elif self.prompt_type == "simple":
                prompt = "Toxic comment: " + toxic + "\nNeutral comment: "
            else:
                raise ValueError("prompt_type must be prev, inst or simple")

            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
            )

        tokenized = {k: v.squeeze(0) for k, v in inputs.items()}

        return tokenized

    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        input_ids = left_pad(input_ids, pad_value=self.tokenizer.pad_token_id)
        attention_mask = left_pad(attention_mask, pad_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def build_few_shot_prompt(examples, prompt_type):
    shot = ''
    if examples is None:
        return shot
    for ex in examples:
        shot += f"Toxic comment: {ex['toxic']}\n"
        if prompt_type == 'inst':
            shot += f"### Response: {ex['neutral']}\n"
        else:
            shot += f"Neutral comment: {ex['neutral']}\n"

    return shot



def left_pad(tensors, pad_value):
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        padding = torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)
        padded.append(torch.cat((padding, t), dim=0))
    return torch.stack(padded)