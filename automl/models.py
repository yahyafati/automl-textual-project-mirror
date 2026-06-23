import numpy as np
import torch
import torch.nn as nn


class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x): return self.model(x)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, **kwargs):
        x = kwargs["input_ids"]
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        logits = self.fc(h[-1])
        return logits