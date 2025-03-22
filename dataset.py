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
        else:
            inputs = self.tokenizer(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
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
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

        else:
            inputs = self.tokenizer(
                "Your task is to review the given toxic comment and convert it into a polite, neutral sentence.\nToxic comment: " +
                self.data[idx]['toxic'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

        tokenized = {k: v.squeeze(0).to(self.device) for k, v in inputs.items()}

        return tokenized
