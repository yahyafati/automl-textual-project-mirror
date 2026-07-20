from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TypedDict

import numpy as np
import pandas as pd
import torch.nn as nn
from ConfigSpace import Configuration
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, PredictionResult, TrainResult
from automl.core.utils import timer
from automl.core.utils.bpe_lstm import make_bpe_collate, BPETokenizedDataset
from automl.core.utils.bpe_tokenizer import BPETokenizer
from automl.core.utils.misc import set_seed
from automl.logger import get_logger

logger = get_logger()


class _PreparationResult(TypedDict):
    model: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader


import torch

_RNN_TYPES = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RNNClassifier(nn.Module):
    """Embedding -> recurrent encoder (RNN | GRU | LSTM) -> linear classifier.

    The last layer's final hidden state is used as the sequence representation.
    Sequences are packed by their true length so trailing padding never leaks
    into that hidden state (important for the ragged, dynamically padded batches).
    ``padding_idx=0`` keeps pad embeddings at zero and out of the gradient.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        rnn_type="lstm",
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in _RNN_TYPES:
            raise ValueError(
                f"rnn_type must be one of {list(_RNN_TYPES)}, got {rnn_type!r}"
            )
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = _RNN_TYPES[rnn_type](
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # inter-layer dropout only applies with >1 layer
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, input_ids, lengths=None, **_):
        emb = self.embed(input_ids)  # [B, L, E]
        if lengths is not None:
            # pack_padded_sequence needs lengths on CPU
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
        else:
            _, hidden = self.rnn(emb)

        # LSTM returns (h_n, c_n); RNN/GRU return h_n. Shape: [layers*dirs, B, H].
        h = hidden[0] if self.rnn_type == "lstm" else hidden
        if self.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=1)  # last layer: forward + backward
        else:
            last = h[-1]  # last layer
        return self.fc(last)


def _make_bpe_collate_inputs_labels(tokenizer: BPETokenizer):
    """
    Wrap the existing make_bpe_collate so that the DataLoader yields
    (input_ids, labels) as a tuple. Sequence lengths are recomputed in
    the model wrapper, so we don't expose them here.
    """
    base_collate = make_bpe_collate(tokenizer.pad_id)

    def collate(examples):
        batch = base_collate(examples)  # dict: {"input_ids", "lengths", "labels"}
        return batch["input_ids"], batch["labels"]

    return collate


class _RNNWithAutoLength(torch.nn.Module):
    """
    Small wrapper around RNNClassifier so that its forward only takes
    input_ids. Sequence lengths are inferred from padding.
    """

    def __init__(self, base_model: RNNClassifier, pad_id: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.pad_id = pad_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq_len]
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids of shape [batch, seq_len], got {input_ids.shape}"
            )
        lengths = (input_ids != self.pad_id).sum(dim=1)
        return self.base_model(input_ids=input_ids, lengths=lengths)


@register_approach("bpe-rnn")
class BpeRNNApproach(Approach[torch.nn.Module, _PreparationResult]):
    """
    RNN text classifier (RNN / GRU / LSTM) over a corpus-trained byte-level BPE vocab,
    implemented as an Approach using base_approach.

    - Trains a fresh BPE tokenizer on the training split.
    - Uses BPETokenizedDataset + dynamic padding.
    - Uses TorchTrainer for training.
    """

    def __init__(
        self,
        config: Configuration | dict,
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, num_classes, device, **kwargs)
        self.tokenizer: Optional[BPETokenizer] = None
        self.trainer: Optional[TorchTrainer] = None

        # Will be filled in initialize()
        self.bpe_vocab_size = int(self.get_param_value("bpe_vocab_size"))
        self.token_length = int(self.get_param_value("token_length"))
        self.rnn_type = self.get_param_value("rnn_type")
        self.emb_dim = int(self.get_param_value("emb_dim"))
        self.hidden_dim = int(self.get_param_value("hidden_dim"))
        self.num_layers = int(self.get_param_value("num_layers"))
        self.bidirectional = bool(self.get_param_value("bidirectional"))
        self.dropout = float(self.get_param_value("dropout"))
        self.batch_size = int(self.get_param_value("batch_size"))

        logger.debug(
            "Initialized BpeRNNApproach with: "
            f"bpe_vocab_size={self.bpe_vocab_size}, token_length={self.token_length}, "
            f"rnn_type={self.rnn_type}, emb_dim={self.emb_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}, "
            f"dropout={self.dropout}, batch_size={self.batch_size}"
        )

        seed = self.get_param_value("seed")
        if seed is not None:
            set_seed(seed)

    # ------------------------------------------------------------------
    # Required by Approach
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set seeds and read all hyperparameters from the config/defaults."""
        pass

    def _build_tokenizer(self, train: DatasetSplit) -> BPETokenizer:
        texts = [str(t) for t in train.texts]
        logger.debug(
            "Training BPE tokenizer on %d texts (vocab_size=%d)",
            len(texts),
            self.bpe_vocab_size,
        )
        tokenizer = BPETokenizer.train(texts, vocab_size=self.bpe_vocab_size)
        return tokenizer

    def prepare_training(self, train: DatasetSplit) -> DataLoader:
        assert (
            self.tokenizer is not None
        ), "Tokenizer must be initialized before prepare_training"
        dataset = BPETokenizedDataset(
            list(map(str, train.texts)),
            list(train.labels),
            self.tokenizer,
            self.token_length,
        )
        collate = _make_bpe_collate_inputs_labels(self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        return loader

    def prepare_validation(self, val: DatasetSplit) -> DataLoader:
        assert (
            self.tokenizer is not None
        ), "Tokenizer must be initialized before prepare_validation"

        if len(val.texts) == 0:
            # Empty validation set; still return a DataLoader for consistency.
            dataset = BPETokenizedDataset([], [], self.tokenizer, self.token_length)
            collate = _make_bpe_collate_inputs_labels(self.tokenizer)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate,
            )

        dataset = BPETokenizedDataset(
            list(map(str, val.texts)),
            list(val.labels),
            self.tokenizer,
            self.token_length,
        )
        collate = _make_bpe_collate_inputs_labels(self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        return loader

    def prepare(self, train: DatasetSplit, val: DatasetSplit) -> _PreparationResult:
        """
        Train the BPE tokenizer, build DataLoaders, and instantiate the RNN model.
        """
        self.initialize()

        with timer.Timer("BPE tokenizer training & data preparation") as t:
            self.tokenizer = self._build_tokenizer(train)
            train_loader = self.prepare_training(train)
            val_loader = self.prepare_validation(val)

        logger.debug(
            "BPE-RNN approach: tokenizer + data loaders built in %.2fs "
            "(train_batches=%d, val_batches=%d)",
            t.execution_time,
            len(train_loader),
            len(val_loader),
        )

        assert self.tokenizer is not None

        base_model = RNNClassifier(
            vocab_size=len(self.tokenizer),
            embedding_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self._num_classes,
            rnn_type=self.rnn_type,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

        model = _RNNWithAutoLength(base_model, pad_id=self.tokenizer.pad_id)
        model.to(self._device)
        self.model = model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug(
            "BPE-RNN model built: total_params=%d, trainable_params=%d",
            total_params,
            trainable_params,
        )

        return {"model": model, "train_loader": train_loader, "val_loader": val_loader}

    def train(
        self,
        prepared_result: _PreparationResult,
        *,
        trainer_load_path: Optional[Path] = None,
        epochs: int = 10,
        **kwargs: Any,
    ) -> TrainResult:
        """
        Train the RNN model using TorchTrainer, analogous to TfidfFFNNApproach.
        """
        logger.debug("Starting BPE-RNN model training...")

        optimizer_name: str = self.get_param_value("optimizer")
        opt_kwargs: dict[str, Any] = {
            "lr": self.get_param_value("learning_rate"),
            "weight_decay": self.get_param_value("weight_decay"),
        }

        if optimizer_name in ["adam", "adamw"]:
            opt_kwargs["betas"] = (
                self.get_param_value("beta1"),
                self.get_param_value("beta2"),
            )
        elif optimizer_name == "sgd":
            opt_kwargs["momentum"] = self.get_param_value("momentum")

        with timer.Timer("BPE-RNN Approach Training") as t:
            trainer = TorchTrainer(
                prepared_result["model"],
                self.name,
                prepared_result["train_loader"],
                prepared_result["val_loader"],
                self._device,
                optimizer=optimizer_name,
                optimizer_args=opt_kwargs,
                epochs=epochs,
            )
            self.trainer = trainer
            result = trainer.train(load_path=trainer_load_path)

        logger.debug(
            "BPE-RNN training completed in %.2fs. Best metric: %.4f",
            t.execution_time,
            result.get("val_accuracy", float("nan")),
        )
        return result

    def _predict_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None

        self.model.eval()
        preds: list[int] = []
        labels: list[int] = []

        with torch.no_grad():
            for input_ids, y in loader:
                input_ids = input_ids.to(self._device)
                logits = self.model(input_ids)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(y.numpy())  # kept on CPU

        return np.asarray(preds), np.asarray(labels)

    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        """
        Run prediction on a DataFrame or DataLoader and return y_pred / y_true.
        """
        assert (
            self.model is not None
        ), "Model is not initialized. Did you call prepare()?"
        assert (
            self.tokenizer is not None
        ), "Tokenizer is not initialized. Did you call prepare()?"

        if isinstance(data, DataLoader):
            y_pred, y_true = self._predict_loader(data)
        else:
            texts = data["text"].astype(str).tolist()
            labels = data["label"].tolist()

            dataset = BPETokenizedDataset(
                texts,
                labels,
                self.tokenizer,
                self.token_length,
            )
            collate = _make_bpe_collate_inputs_labels(self.tokenizer)
            loader = DataLoader(
                dataset,
                batch_size=int(self.get_param_value("batch_size")),
                shuffle=False,
                collate_fn=collate,
            )
            y_pred, y_true = self._predict_loader(loader)

        return {"y_pred": y_pred, "y_true": y_true}
