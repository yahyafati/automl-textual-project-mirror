from functools import lru_cache, partial
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ConfigSpace import Configuration
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, AutoModel

from automl.core.approaches.base_approach import Approach
from automl.core.approaches.text_encoding import (
    TextSequenceDataset,
    collate_sequences,
    encode_texts_cached,
    load_tokenizer,
)
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, TrainResult, PredictionResult
from automl.logger import get_logger

logger = get_logger()


@lru_cache(maxsize=4)
def _load_pretrained_word_embeddings(model_name: str) -> torch.Tensor:
    """Load (and cache) just the pretrained token embedding matrix for
    `model_name`, used to warm-start the BiLSTM's embedding layer instead
    of starting from scratch.

    `model_name` is HPO-tunable (`seq_pretrained_model_name`), so maxsize
    matches `SequenceDLApproach.MODEL_NAME_CHOICES`'s length - otherwise a
    single-slot cache would evict on every model switch, defeating both
    this cache and the ifBO prewarm loop that iterates every choice.
    """
    logger.info(f"Loading pretrained embedding matrix from '{model_name}'.")
    model = AutoModel.from_pretrained(model_name)
    weight = model.get_input_embeddings().weight.detach().clone()
    del model
    logger.debug(
        f"Pretrained embedding matrix for '{model_name}': shape={tuple(weight.shape)}."
    )
    return weight


@lru_cache(maxsize=4)
def _pretrained_svd(model_name: str, vocab_size: int):
    logger.info(
        f"Computing SVD of pretrained embeddings for '{model_name}' "
        f"(vocab_size={vocab_size}); this runs once per (model_name, vocab_size)."
    )
    matrix = _load_pretrained_word_embeddings(model_name)
    if matrix.size(0) >= vocab_size:
        matrix = matrix[:vocab_size]
    else:
        pad = torch.empty(vocab_size - matrix.size(0), matrix.size(1))
        nn.init.normal_(pad, mean=0.0, std=0.02)
        matrix = torch.cat([matrix, pad], dim=0)
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    logger.debug(f"SVD complete for '{model_name}' (vocab_size={vocab_size}).")
    return centered, vt  # computed once, ever, per (model_name, vocab_size)


@lru_cache(maxsize=16)
def _pretrained_embedding_init(
    model_name: str, vocab_size: int, target_dim: int
) -> torch.Tensor:
    """Build an (vocab_size, target_dim) init matrix from `model_name`'s
    pretrained embeddings.

    `target_dim` is the tuned `seq_embed_dim` hyperparameter (32-512) and
    essentially never matches the transformer's native hidden size (768
    for distilbert), so a straight copy isn't possible. Instead, we PCA the
    pretrained matrix down to `target_dim`: this keeps the directions of
    highest variance in the pretrained embedding space, so tokens that are
    semantically close before the projection stay close after it too -
    still a much better starting point than random init, which is what
    makes the LSTM re-learn token semantics from scratch every trial.
    """
    logger.debug(
        f"Building pretrained embedding init for '{model_name}' "
        f"(vocab_size={vocab_size}, target_dim={target_dim})."
    )
    centered, vt = _pretrained_svd(model_name, vocab_size)
    if target_dim >= centered.size(1):
        logger.debug(
            f"target_dim={target_dim} >= native dim={centered.size(1)}; "
            f"padding with {target_dim - centered.size(1)} randomly-initialized dim(s)."
        )
        extra = torch.empty(centered.size(0), target_dim - centered.size(1))
        nn.init.normal_(extra, mean=0.0, std=0.02)
        return torch.cat(
            [centered, extra], dim=1
        )  # note: not mean-restored, matches original behavior
    return centered @ vt[:target_dim].T


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
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
                self.embedding.weight[0].zero_()  # keep padding_idx row zero
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
        # Derive real sequence lengths from padding and pack the batch
        # before running the LSTM. Previously the LSTM ran (and stored
        # activations for backprop) over every padded position too - with
        # a large max_seq_length and mostly-short texts, that's a lot of
        # wasted compute and, more importantly, wasted autograd memory.
        # Packing skips padded positions entirely in both directions.
        lengths = (input_ids != self.embedding.padding_idx).sum(dim=1).clamp(min=1)

        emb = self.embedding(input_ids)  # (B, L, E)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(packed)  # h_n: (num_layers*D, B, H)

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


@register_approach("sequence-dl")
class SequenceDLApproach(Approach[torch.nn.Module, dict]):

    TOKENIZERS_DIR = "./tokenizers"
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    # Vendored locally under TOKENIZERS_DIR (see save_tokenizer.py); the
    # `seq_pretrained_model_name` hyperparameter picks between these.
    # Tokenizer and embedding source are always the same model, since the
    # BiLSTM's vocab indices must line up with whichever embedding matrix
    # warm-starts it.
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
        self.vocab = None
        self.model = None
        # self.label_encoder = LabelEncoder()
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
        embed_dim = int(self.get_param_value("seq_embed_dim"))
        hidden_dim = int(self.get_param_value("hidden_dim"))
        num_layers = int(self.get_param_value("seq_num_layers"))
        dropout = float(self.get_param_value("dropout"))
        batch_size = int(self.get_param_value("batch_size"))
        self._model_name = self.get_param_value("seq_pretrained_model_name")
        self._tokenizer_path = f"{self.TOKENIZERS_DIR}/{self._model_name}"

        logger.debug(
            f"[{self.name}] prepare(): max_seq_len={max_seq_len}, "
            f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, dropout={dropout}, "
            f"batch_size={batch_size}, model_name={self._model_name}, "
            f"num_workers={self._num_worker}."
        )

        # Build vocab on train text
        train_texts = train.texts  # adjust column name as needed

        # Build datasets
        train_labels = train.labels
        val_texts = val.texts
        val_labels = val.labels

        logger.debug(
            f"[{self.name}] prepare(): {len(train_texts)} train text(s), "
            f"{len(val_texts)} val text(s)."
        )

        # train_labels = self.label_encoder.fit_transform(train_labels)
        # val_labels = self.label_encoder.transform(val_labels)

        self.tokenizer = load_tokenizer(self._tokenizer_path)
        assert self.tokenizer is not None, "Tokenizer is None"
        self._pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        self._sep_token_id = self.tokenizer.sep_token_id

        # Encode once per unique text (cached across trials, see
        # `encode_texts_cached`); only truncation to this trial's
        # `max_seq_len` happens below, in `TextSequenceDataset`.
        train_full_ids = encode_texts_cached(
            train_texts, self.tokenizer, self._tokenizer_path
        )
        val_full_ids = encode_texts_cached(
            val_texts, self.tokenizer, self._tokenizer_path
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

        # Build model
        vocab_size = self.tokenizer.vocab_size
        pretrained_embeddings = _pretrained_embedding_init(
            self._model_name, vocab_size, embed_dim
        )
        self.model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=self._num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            pretrained_embeddings=pretrained_embeddings,
        )
        assert self.model is not None
        self.model.to(self._device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.debug(
            f"[{self.name}] Built BiLSTMClassifier: vocab_size={vocab_size}, "
            f"num_classes={self._num_classes}, num_params={num_params}, "
            f"device={self._device.type}."
        )

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
        lr = float(self.get_param_value("learning_rate"))
        weight_decay = float(self.get_param_value("weight_decay"))

        scheduler = self.get_param_value("scheduler")
        warmup_ratio = float(self.get_param_value("warmup_ratio"))
        max_grad_norm = float(self.get_param_value("max_grad_norm"))

        optimizer_args = {"lr": lr, "weight_decay": weight_decay}

        # remove value which are None
        # scheduler_args = {k: v for k, v in scheduler_args.items() if v is not None}

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
            )
            self.trainer = trainer
        else:
            logger.debug(
                f"[{self.name}] Reusing existing trainer for continued training."
            )

        assert self.trainer is not None
        result = self.trainer.train(
            load_path=load_path,
            save_path=None,  # or some path if you want val-best checkpoint
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
            # `y` is only ever compared/concatenated on CPU later - it never
            # needs to touch the GPU at all, so we no longer copy it there.

            logits = self.model(x)
            preds = torch.argmax(logits, dim=-1)

            # Move each batch's predictions to CPU immediately instead of
            # letting a list of GPU-resident tensors grow for the entire
            # pass. For large prediction sets this bounds peak GPU memory
            # to ~one batch instead of the whole dataset.
            all_preds.append(preds.cpu())
            all_labels.append(y)

        # One GPU -> CPU transfer per batch, already done above.
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        # y_pred_orig = self.label_encoder.inverse_transform(y_pred)

        logger.debug(f"[{self.name}] predict(): produced {len(y_pred)} prediction(s).")

        return {
            "y_pred": y_pred,
            "y_true": y_true,
        }
