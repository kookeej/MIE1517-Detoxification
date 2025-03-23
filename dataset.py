from typing import Optional
from transformers import T5Tokenizer
from torch.utils.data import Dataset

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
            prompt = f"Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: {toxic} \nNeutral comment: "
            full_text = prompt + neutral + self.tokenizer.eos_token

            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)

            labels = input_ids.clone()

            prompt_len = len(self.tokenizer(prompt, return_tensors='pt')['input_ids'].squeeze(0)) - 1
            labels[:prompt_len] = -100

            tokenized = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device),
                'labels': labels.to(self.device)
            }

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
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            raise NotImplementedError
        else:
            toxic = self.data[idx]['toxic']
            prompt = f"Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: {toxic} \nNeutral comment: "

            inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

        tokenized = {k: v.squeeze(0).to(self.device) for k, v in inputs.items()}

        return tokenized
