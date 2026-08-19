from functools import partial
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
import torch.nn as nn
from ConfigSpace import Configuration
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from automl.core.approaches.base_approach import Approach
from automl.core.approaches.text_encoding import (
    TextSequenceDataset,
    collate_sequences,
    encode_texts_cached,
    expand_with_truncation_augmentation,
    get_ellipsis_ids,
    load_tokenizer,
)
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, TrainResult, PredictionResult
from automl.logger import get_logger

logger = get_logger()


class SimpleFFNNClassifier(nn.Module):
    """Bag-of-embeddings baseline: embed each token, mean-pool over the
    non-pad positions to get one fixed-size vector per example, then run
    that vector through a plain feed-forward network.

    Unlike `BiLSTMClassifier`, there is no recurrence/attention here, so
    token order is discarded entirely - this is intentionally the
    "simple" counterpart to `SequenceDLApproach`.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(max(num_layers, 1)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.ffnn = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, input_ids):
        mask = (input_ids != self.padding_idx).unsqueeze(-1).float()  # (B, L, 1)
        emb = self.embedding(input_ids)  # (B, L, E)

        summed = (emb * mask).sum(dim=1)  # (B, E)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1), avoid div-by-zero
        pooled = summed / counts  # masked mean pooling -> (B, E)

        hidden = self.ffnn(pooled)
        logits = self.classifier(hidden)
        return logits


@register_approach("simple")
class SimpleApproach(Approach[torch.nn.Module, dict]):

    TOKENIZERS_DIR = "./tokenizers"
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    MODEL_NAME_CHOICES = (
        "distilbert-base-uncased",
        "bert-base-uncased",
        "google/bert_uncased_L-4_H-512_A-8",
        "microsoft/xtremedistil-l6-h256-uncased",
    )

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
            else int(
                self.get_param_value("num_workers")
            )  # FIXME: Not a configuration param
        )
        self._stochastic_epochs: bool = bool(kwargs.get("stochastic_epochs", False))
        self._stochastic_epoch_fraction: Optional[float] = kwargs.get(
            "stochastic_epoch_fraction", None
        )
        self.model = None
        self.trainer: Optional[TorchTrainer] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._pad_id: int = 0
        self._sep_token_id: Optional[int] = None
        self._model_name: str = self.DEFAULT_MODEL_NAME
        self._tokenizer_path: str = f"{self.TOKENIZERS_DIR}/{self._model_name}"

    def initialize(self):
        pass

    def prepare_training(self, train: DatasetSplit):
        return train  # handled in prepare()

    def prepare_validation(self, val: DatasetSplit):
        return val

    def prepare(self, train: DatasetSplit, val: DatasetSplit):
        # Hyperparams from config / defaults
        max_seq_len = int(self.get_param_value("max_seq_length"))
        embed_dim = int(self.get_param_value("simple_embed_dim"))
        hidden_dim = int(self.get_param_value("simple_hidden_dim"))
        num_layers = int(self.get_param_value("simple_num_layers"))
        dropout = float(self.get_param_value("dropout"))
        batch_size = int(self.get_param_value("batch_size"))
        self._model_name = self.get_param_value("simple_pretrained_model_name")
        self._tokenizer_path = f"{self.TOKENIZERS_DIR}/{self._model_name}"

        logger.debug(
            f"[{self.name}] prepare(): max_seq_len={max_seq_len}, "
            f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, dropout={dropout}, "
            f"batch_size={batch_size}, model_name={self._model_name}, "
            f"num_workers={self._num_worker}."
        )

        train_texts = train.texts
        train_labels = train.labels
        val_texts = val.texts
        val_labels = val.labels

        logger.debug(
            f"[{self.name}] prepare(): {len(train_texts)} train text(s), "
            f"{len(val_texts)} val text(s)."
        )

        self.tokenizer = load_tokenizer(self._tokenizer_path)
        assert self.tokenizer is not None, "Tokenizer is None"
        self._pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        self._sep_token_id = self.tokenizer.sep_token_id

        train_full_ids = encode_texts_cached(
            train_texts, self.tokenizer, self._tokenizer_path
        )
        val_full_ids = encode_texts_cached(
            val_texts, self.tokenizer, self._tokenizer_path
        )

        ellipsis_ids = get_ellipsis_ids(self.tokenizer, self._tokenizer_path)
        train_full_ids, train_labels = expand_with_truncation_augmentation(
            train_texts,
            train_full_ids,
            train_labels,
            max_seq_len,
            self._sep_token_id,
            self._tokenizer_path,
            ellipsis_ids,
        )

        train_ds = TextSequenceDataset(
            train_full_ids, train_labels, max_seq_len, self._sep_token_id
        )
        val_ds = TextSequenceDataset(
            val_full_ids, val_labels, max_seq_len, self._sep_token_id
        )

        collate_fn = partial(collate_sequences, pad_value=self._pad_id)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._num_worker,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_worker > 0,
            collate_fn=collate_fn,
        )

        # Build model - plain embedding table (no pretrained warm-start;
        # that's what keeps this approach "simple" relative to sequence-dl)
        vocab_size = self.tokenizer.vocab_size
        self.model = SimpleFFNNClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=self._num_classes,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=self._pad_id,
        )
        assert self.model is not None
        self.model.to(self._device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(
            f"[{self.name}] Built SimpleFFNNClassifier: vocab_size={vocab_size}, "
            f"num_classes={self._num_classes}, num_params={num_params}, "
            f"device={self._device.type}."
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
        }

    def train(
        self,
        prepared_result,
        epochs: int = 10,
        load_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        evaluate_validation=True,
        max_time_seconds: Optional[float] = None,
        **kwargs,
    ) -> TrainResult:
        assert self.model is not None

        optimizer_name = self.get_param_value("optimizer")
        lr = float(self.get_param_value("learning_rate"))
        weight_decay = float(self.get_param_value("weight_decay"))

        scheduler = self.get_param_value("scheduler")
        warmup_ratio = float(self.get_param_value("warmup_ratio"))
        max_grad_norm = float(self.get_param_value("max_grad_norm"))

        optimizer_args = {"lr": lr, "weight_decay": weight_decay}

        logger.debug(
            f"[{self.name}] train(): optimizer={optimizer_name}, lr={lr}, "
            f"weight_decay={weight_decay}, scheduler={scheduler}, "
            f"warmup_ratio={warmup_ratio}, max_grad_norm={max_grad_norm}, "
            f"epochs={epochs}, evaluate_validation={evaluate_validation}, "
            f"load_path={load_path}."
        )

        if self.trainer is None:
            logger.debug(
                f"[{self.name}] No existing trainer; creating a new TorchTrainer."
            )
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
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                stochastic_epochs=self._stochastic_epochs,
                stochastic_epoch_fraction=self._stochastic_epoch_fraction,
                max_time_seconds=max_time_seconds,
            )
            self.trainer = trainer
        else:
            logger.debug(
                f"[{self.name}] Reusing existing trainer for continued training."
            )

        assert self.trainer is not None
        result = self.trainer.train(
            load_path=load_path,
            save_path=save_path,
        )
        logger.info(f"[{self.name}] train() finished after {epochs} epoch(s).")
        return result

    @torch.no_grad()
    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None, "Model is not initialized"
        assert self.tokenizer is not None, "Tokenizer is not initialized"

        max_seq_length = int(self.get_param_value("max_seq_length"))
        batch_size = int(self.get_param_value("batch_size"))
        self.model.eval()

        if isinstance(data, pd.DataFrame):
            logger.debug(
                f"[{self.name}] predict(): building DataLoader from a "
                f"{len(data)}-row DataFrame (max_seq_length={max_seq_length}, "
                f"batch_size={batch_size})."
            )
            texts = data["text"].tolist()
            labels = data["label"].tolist()

            full_ids = encode_texts_cached(texts, self.tokenizer, self._tokenizer_path)
            ds = TextSequenceDataset(
                full_ids,
                labels=labels,
                max_seq_len=max_seq_length,
                sep_token_id=self._sep_token_id,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self._device.type == "cuda",
                collate_fn=partial(collate_sequences, pad_value=self._pad_id),
            )
        else:
            logger.debug(f"[{self.name}] predict(): using caller-provided DataLoader.")
            loader = data

        all_preds = []
        all_labels = []

        for batch in loader:
            x, y = batch
            x = x.to(self._device, non_blocking=True)

            logits = self.model(x)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        logger.debug(f"[{self.name}] predict(): produced {len(y_pred)} prediction(s).")

        return {
            "y_pred": y_pred,
            "y_true": y_true,
        }
