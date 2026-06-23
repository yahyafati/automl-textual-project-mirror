# src/utils.py
import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if isinstance(self.tokenizer, dict):
            tokens = text.split()
            ids = [self.tokenizer.get(tok, 1) for tok in tokens[:self.max_length]]
            ids += [0] * (self.max_length - len(ids))
            return torch.tensor(ids), torch.tensor(label)

        elif TRANSFORMERS_AVAILABLE and self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(label)
            }
        else:
            raise ValueError("Tokenizer not defined or unsupported.")