from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
import torch.nn as nn
from ConfigSpace import Configuration
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, TrainResult, PredictionResult


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * directions, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # (B, L, E)
        outputs, (h_n, c_n) = self.lstm(emb)  # h_n: (num_layers*D, B, H)
        # Use last layer’s hidden state, concatenate both directions
        if self.lstm.bidirectional:
            last_fwd = h_n[-2, :, :]  # (B, H)
            last_bwd = h_n[-1, :, :]  # (B, H)
            h = torch.cat([last_fwd, last_bwd], dim=1)
        else:
            h = h_n[-1, :, :]
        h = self.dropout(h)
        logits = self.fc(h)  # (B, num_classes)
        return logits


class TextSequenceDataset(torch.utils.data.Dataset):

    DEFAULT_LABEL_MASK = -100

    def __init__(
        self,
        texts,
        labels,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ):
        self.labels = labels

        # Tokenizer processes the entire list at once during startup
        self.encoded_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

    def __len__(self):
        return (
            len(self.labels)
            if self.labels is not None
            else len(self.encoded_inputs["input_ids"])
        )

    def __getitem__(self, idx):
        x = self.encoded_inputs["input_ids"][idx]
        if self.labels is None:
            return x
        label = self.labels[idx]
        if not pd.isna(label):
            return x, torch.tensor(label, dtype=torch.long)
        return x, torch.tensor(self.DEFAULT_LABEL_MASK, dtype=torch.long)


@register_approach("sequence-dl")
class SequenceDLApproach(Approach[torch.nn.Module, dict]):

    def __init__(
        self,
        config: Union[Configuration, dict],
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(config, num_classes, device, **kwargs)
        num_workers: Optional[int] = kwargs.get("num_workers", None)
        self._num_worker: int = (
            num_workers
            if num_workers is not None
            else self.get_param_value("num_workers")  # FIXME: Not a configuration param
        )
        self.vocab = None
        self.model = None
        self.trainer: Optional[TorchTrainer] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def initialize(self):
        pass

    def prepare_training(self, train: DatasetSplit):
        return train  # handled in prepare()

    def prepare_validation(self, val: DatasetSplit):
        return val

    def prepare(self, train: DatasetSplit, val: DatasetSplit):
        # Hyperparams from config / defaults
        max_seq_len = self.get_param_value("max_seq_length")
        embed_dim = self.get_param_value("seq_embed_dim")
        hidden_dim = self.get_param_value("hidden_dim")
        num_layers = self.get_param_value("seq_num_layers")
        dropout = self.get_param_value("dropout")
        batch_size = self.get_param_value("batch_size", apply_fn=int)

        # Build vocab on train text
        train_texts = train.texts  # adjust column name as needed

        # Build datasets
        train_labels = train.labels
        val_texts = val.texts
        val_labels = val.labels

        # TODO: Add to configspace
        self.tokenizer = AutoTokenizer.from_pretrained("./tokenizers/distilbert-base-uncased")  # type: ignore
        assert self.tokenizer is not None, "Tokenizer is None"

        train_ds = TextSequenceDataset(
            train_texts, train_labels, self.tokenizer, max_seq_len
        )
        val_ds = TextSequenceDataset(val_texts, val_labels, self.tokenizer, max_seq_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
        )

        # Build model
        vocab_size = self.tokenizer.vocab_size
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=self._num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        assert self.model is not None
        self.model.to(self._device)

        # Trainer
        # epochs = self.get_param_value("epochs")

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "vocab": self.vocab,
        }

    def train(
        self,
        prepared_result,
        epochs: int = 10,
        load_path: Optional[Path] = None,
        evaluate_validation=True,
        **kwargs,
    ) -> TrainResult:
        assert self.model is not None

        optimizer_name = self.get_param_value("optimizer")
        lr = self.get_param_value("learning_rate")
        weight_decay = self.get_param_value("weight_decay")

        scheduler = self.get_param_value("scheduler")

        optimizer_args = {"lr": lr, "weight_decay": weight_decay}

        # remove value which are None
        # scheduler_args = {k: v for k, v in scheduler_args.items() if v is not None}

        if self.trainer is None:
            trainer = TorchTrainer(
                model=self.model,
                approach_name=self.name,
                train_loader=prepared_result["train_loader"],
                val_loader=prepared_result["val_loader"],
                device=self._device,
                optimizer=optimizer_name,
                optimizer_args=optimizer_args,
                scheduler=scheduler,
                epochs=epochs,
                evaluate_validation=evaluate_validation,
            )
            self.trainer = trainer

        assert self.trainer is not None
        result = self.trainer.train(
            load_path=load_path,
            save_path=None,  # or some path if you want val-best checkpoint
        )
        return result

    @torch.no_grad()
    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None, "Model is not initialized"
        assert self.tokenizer is not None, "Tokenizer is not initialized"

        max_seq_length = self.get_param_value("max_seq_length")
        batch_size = self.get_param_value("batch_size")
        self.model.eval()

        if isinstance(data, pd.DataFrame):
            texts = data["text"].tolist()
            labels = data["label"].tolist()

            ds = TextSequenceDataset(
                texts,
                labels=labels,
                tokenizer=self.tokenizer,
                max_seq_len=max_seq_length,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self._device.type == "cuda",
            )
        else:
            loader = data

        all_preds = []
        all_labels = []

        for batch in loader:
            x, y = batch
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)

            logits = self.model(x)
            preds = torch.argmax(logits, dim=-1)

            # Keep tensors on GPU
            all_preds.append(preds)
            all_labels.append(y)

        # One GPU -> CPU transfer at the end
        y_pred = torch.cat(all_preds).cpu().numpy()
        y_true = torch.cat(all_labels).cpu().numpy()

        return {
            "y_pred": y_pred,
            "y_true": y_true,
        }
