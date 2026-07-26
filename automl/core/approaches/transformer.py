from functools import partial
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
import torch.nn as nn
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


def _freeze_base_by_ratio(base: nn.Module, freeze_ratio: float) -> None:
    """Freezes the lowest `freeze_ratio` fraction of `base`'s parameters.

    Treats the embeddings and each transformer block as one freezable unit,
    ordered bottom-up (closest to the input first) - lower layers encode
    more generic, transferable features, so they're the ones frozen first.
    `freeze_ratio=1.0` freezes every unit (embeddings + all blocks, i.e.
    linear-probe mode), `0.0` freezes nothing (full fine-tuning), and e.g.
    `0.5` freezes the embeddings plus the lower half of the blocks.
    """
    if freeze_ratio >= 1.0:
        # Freeze every parameter outright, including ones not covered by
        # the unit list below (e.g. BERT's pooler) - "1.0" should mean
        # linear-probing with a fully-fixed base, not "everything the unit
        # list happens to enumerate."
        for param in base.parameters():
            param.requires_grad_(False)
        return

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        blocks = list(base.encoder.layer)  # BERT-style
    elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        blocks = list(base.transformer.layer)  # DistilBERT-style
    else:
        raise ValueError(f"Don't know how to locate transformer blocks on {type(base)}")

    units: list[nn.Module] = [base.embeddings, *blocks]
    num_freeze = round(freeze_ratio * len(units))
    for unit in units[:num_freeze]:
        for param in unit.parameters():
            param.requires_grad_(False)


class TransformerClassifier(nn.Module):
    """A pretrained transformer encoder fine-tuned with a classification
    head on top.

    Where `BiLSTMClassifier` (sequence_dl.py) only ever borrows a pretrained
    *embedding matrix* to warm-start training from scratch, this model keeps
    (and fine-tunes) the transformer itself.

    `forward` takes a single `input_ids` tensor - not an `(input_ids,
    attention_mask)` pair - so it plugs directly into
    `TorchTrainer._compute_loss`, which (like it already does for
    `BiLSTMClassifier`) only ever calls `self.model(x)` with one tensor.
    The attention mask is instead derived here from padding, the same way
    `BiLSTMClassifier.forward` derives its sequence lengths from
    `input_ids != padding_idx`.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pad_token_id: int,
        dropout: float = 0.1,
        freeze_ratio: float = 0.0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.base = AutoModel.from_pretrained(model_name)
        if freeze_ratio > 0.0:
            _freeze_base_by_ratio(self.base, freeze_ratio)
        hidden_size = self.base.config.hidden_size
        # A small MLP head (Linear -> GELU -> LayerNorm -> Dropout ->
        # Linear) instead of a bare `Dropout -> Linear` on top of the CLS
        # token: the extra projection gives the head room to reshape the
        # pretrained representation for the target task, and the LayerNorm
        # keeps that projection's output well-scaled - useful in particular
        # when `freeze_ratio` is high and the head is doing most of the
        # adapting.
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_token_id).long()
        last_hidden_state = self.base(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # CLS-token pooling: the shared tokenizer prepends [CLS] to every
        # sequence (see `text_encoding.truncate_ids`'s docstring), so
        # position 0 always holds it, regardless of truncation.
        cls_hidden = last_hidden_state[:, 0, :]
        pooled = self.layer_norm(self.activation(self.pre_classifier(cls_hidden)))
        return self.classifier(self.dropout(pooled))


@register_approach("transformer")
class TransformerApproach(Approach[TransformerClassifier, dict]):
    """Fine-tunes a pretrained transformer (e.g. DistilBERT/BERT) for text
    classification, as an alternative to `SequenceDLApproach`'s
    from-scratch BiLSTM.
    """

    TOKENIZERS_DIR = "./tokenizers"
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"
    # Vendored locally under TOKENIZERS_DIR (see save_tokenizer.py); the
    # `transformer_model_name` hyperparameter picks between these.
    MODEL_NAME_CHOICES = ("distilbert-base-uncased", "bert-base-uncased")

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
        self._stochastic_epochs: bool = bool(kwargs.get("stochastic_epochs", False))
        self._stochastic_epoch_fraction: Optional[float] = kwargs.get(
            "stochastic_epoch_fraction", None
        )
        self.model: Optional[TransformerClassifier] = None
        self.trainer: Optional[TorchTrainer] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model_name: str = self.DEFAULT_MODEL_NAME
        self._pad_id: int = 0
        self._sep_token_id: Optional[int] = None

    def initialize(self):
        pass

    def prepare_training(self, train: DatasetSplit):
        return train  # handled in prepare()

    def prepare_validation(self, val: DatasetSplit):
        return val

    def _tokenizer_path(self, model_name: str) -> str:
        return f"{self.TOKENIZERS_DIR}/{model_name}"

    def prepare(self, train: DatasetSplit, val: DatasetSplit):
        # Hyperparams from config / defaults
        max_seq_len = int(self.get_param_value("max_seq_length"))
        model_name = self.get_param_value("transformer_model_name")
        classifier_dropout = float(self.get_param_value("dropout"))
        freeze_ratio = float(self.get_param_value("freeze_ratio"))
        batch_size = int(self.get_param_value("batch_size"))

        logger.debug(
            f"[{self.name}] prepare(): max_seq_len={max_seq_len}, "
            f"model_name={model_name}, classifier_dropout={classifier_dropout}, "
            f"freeze_ratio={freeze_ratio}, batch_size={batch_size}, "
            f"num_workers={self._num_worker}."
        )

        self._model_name = model_name
        tokenizer_path = self._tokenizer_path(model_name)

        train_texts = train.texts
        train_labels = train.labels
        val_texts = val.texts
        val_labels = val.labels

        logger.debug(
            f"[{self.name}] prepare(): {len(train_texts)} train text(s), "
            f"{len(val_texts)} val text(s)."
        )

        self.tokenizer = load_tokenizer(tokenizer_path)
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
            train_texts, self.tokenizer, tokenizer_path
        )
        val_full_ids = encode_texts_cached(val_texts, self.tokenizer, tokenizer_path)

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

        self.model = TransformerClassifier(
            model_name=model_name,
            num_classes=self._num_classes,
            pad_token_id=self._pad_id,
            dropout=classifier_dropout,
            freeze_ratio=freeze_ratio,
        )
        self.model.to(self._device)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.debug(
            f"[{self.name}] Built TransformerClassifier from '{model_name}': "
            f"num_classes={self._num_classes}, num_params={num_params}, "
            f"num_trainable={num_trainable}, device={self._device.type}."
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
        evaluate_validation=True,
        **kwargs,
    ) -> TrainResult:
        assert self.model is not None

        optimizer_name = self.get_param_value("optimizer")
        lr = self.get_param_value("learning_rate")
        weight_decay = self.get_param_value("weight_decay")

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
            self.trainer = TorchTrainer(
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
        else:
            logger.debug(
                f"[{self.name}] Reusing existing trainer for continued training."
            )

        assert self.trainer is not None
        result = self.trainer.train(
            load_path=load_path,
            save_path=None,
        )
        logger.info(f"[{self.name}] train() finished after {epochs} epoch(s).")
        return result

    @torch.no_grad()
    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None, "Model is not initialized"
        assert self.tokenizer is not None, "Tokenizer is not initialized"

        max_seq_length = self.get_param_value("max_seq_length")
        batch_size = self.get_param_value("batch_size")
        self.model.eval()

        if isinstance(data, pd.DataFrame):
            logger.debug(
                f"[{self.name}] predict(): building DataLoader from a "
                f"{len(data)}-row DataFrame (max_seq_length={max_seq_length}, "
                f"batch_size={batch_size})."
            )
            texts = data["text"].tolist()
            labels = data["label"].tolist()

            tokenizer_path = self._tokenizer_path(self._model_name)
            full_ids = encode_texts_cached(texts, self.tokenizer, tokenizer_path)
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
