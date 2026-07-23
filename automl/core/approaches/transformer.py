from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from ConfigSpace import Configuration
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
)

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.transformer_trainer import TransformerTrainer
from automl.core.types import (
    DatasetSplit,
    TrainResult,
    PredictionResult,
)
from automl.logger import get_logger

logger = get_logger()


class TransformerTextDataset(Dataset):
    """
    Dataset that tokenizes upfront to avoid multiprocessing deadlocks/crashes.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_length: int,
    ):
        self.labels = labels

        logger.info(f"Pre-tokenizing {len(texts)} texts...")
        logger.debug(
            f"TransformerTextDataset init: max_length={max_length}, "
            f"labels_provided={labels is not None}"
        )

        # Tokenize everything at once
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logger.debug(
            "Finished pre-tokenization. Tensor shapes: "
            + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self.encodings.items())
        )

        if labels is not None:
            if len(labels) != self.encodings["input_ids"].shape[0]:
                logger.warning(
                    "Labels length does not match tokenized input length: "
                    f"len(labels)={len(labels)} vs "
                    f"input_len={self.encodings['input_ids'].shape[0]}"
                )

    def __len__(self) -> int:
        length = (
            len(self.labels)
            if self.labels is not None
            else len(self.encodings["input_ids"])
        )
        logger.debug(f"TransformerTextDataset.__len__ -> {length}")
        return length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Simply slice the pre-computed tensors
        item = {k: v[idx] for k, v in self.encodings.items()}
        logger.debug(
            f"TransformerTextDataset.__getitem__ called for idx={idx}. "
            f"Keys: {list(item.keys())}"
        )

        if self.labels is not None:
            label = self.labels[idx]
            item["labels"] = torch.tensor(label, dtype=torch.long)
            logger.debug(f"TransformerTextDataset.__getitem__ idx={idx}, label={label}")

        return item


@dataclass
class TransformerPreparationResult:
    train_loader: DataLoader
    val_loader: DataLoader


@register_approach("transformer")
class TransformerApproach(
    Approach[AutoModelForSequenceClassification, TransformerPreparationResult]
):
    """
    Modern approach: pretrained transformer with contextualized token-level
    representations, finetuned end-to-end for classification.
    """

    def __init__(
        self,
        config: Configuration,
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        logger.debug(
            f"[transformer] __init__ called with num_classes={num_classes}, "
            f"device={device}, extra_kwargs={list(kwargs.keys())}"
        )
        super().__init__(config, num_classes, device, **kwargs)
        self.tokenizer = None
        self._max_seq_length: int = 128

        num_workers: Optional[int] = kwargs.get("num_workers", None)
        self._num_worker: int = (
            num_workers
            if num_workers is not None
            else self.get_param_value("num_workers")
        )

        logger.info(
            f"[{self.name}] Initialized with num_classes={num_classes}, "
            f"device={device}, num_workers={self._num_worker}"
        )
        # logger.debug(f"[{self.name}] Config space raw: {self.config.get_dictionary()}")

    # ---- helpers ---------------------------------------------------------

    def _get_max_seq_length(self) -> int:
        max_len = int(self.get_param_value("max_seq_length"))
        logger.debug(f"[{self.name}] _get_max_seq_length -> {max_len}")
        return max_len

    def _get_batch_size(self) -> int:
        batch_size = int(self.get_param_value("transformer_batch_size"))
        logger.debug(f"[{self.name}] _get_batch_size -> {batch_size}")
        return batch_size

    # ---- abstract interface implementations ------------------------------

    def initialize(self):
        logger.info(f"[{self.name}] Initializing TransformerApproach components...")
        model_name = self.get_param_value(
            "transformer_model_name",
            default="distilbert-base-uncased",
        )
        self._max_seq_length = self._get_max_seq_length()

        logger.info(f"[{self.name}] Initializing tokenizer for '{model_name}'")
        logger.debug(
            f"[{self.name}] Tokenizer params: model_name={model_name}, "
            f"use_fast=True, max_seq_length={self._max_seq_length}"
        )
        local_path = Path("./tokenizers") / model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(local_path), use_fast=True)

        logger.info(
            f"[{self.name}] Loading model '{model_name}' onto device: {self._device}"
        )
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self._num_classes,
            ).to(self._device)
        )

        # The HPO config space samples max_seq_length independently of the
        # chosen transformer_model_name, so it can exceed the model's
        # positional embedding capacity (e.g. 1024 with a 512-limit model
        # like distilbert/bert-base-uncased). Clamp to what the model can
        # actually handle to avoid a shape-mismatch crash inside the model's
        # embedding layer.
        model_max_len = getattr(self.model.config, "max_position_embeddings", None)
        if model_max_len is not None and self._max_seq_length > model_max_len:
            logger.warning(
                f"[{self.name}] Requested max_seq_length={self._max_seq_length} "
                f"exceeds model '{model_name}' max_position_embeddings="
                f"{model_max_len}. Clamping to {model_max_len}."
            )
            self._max_seq_length = model_max_len

        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.debug(
            f"[{self.name}] Loaded model has {total_params:,} total parameters, "
            f"{trainable_params:,} initially trainable."
        )

        # Optionally freeze base encoder for cheaper finetuning
        freeze = bool(self.get_param_value("freeze_transformer"))
        logger.info(f"[{self.name}] freeze_transformer={freeze}")
        if freeze:
            logger.info(
                f"[{self.name}] Freezing transformer encoder parameters; "
                f"leaving classification head open."
            )
            frozen_count = 0
            trainable_count = 0

            for name, param in self.model.named_parameters():
                if "classifier" in name or "score" in name:
                    param.requires_grad = True
                    trainable_count += param.numel()
                    logger.debug(
                        f"[{self.name}] Keeping trainable: {name} "
                        f"(params={param.numel():,})"
                    )
                else:
                    param.requires_grad = False
                    frozen_count += param.numel()
                    logger.debug(
                        f"[{self.name}] Freezing: {name} " f"(params={param.numel():,})"
                    )

            logger.info(
                f"[{self.name}] Parameter freeze breakdown -> "
                f"Frozen: {frozen_count:,} | Trainable: {trainable_count:,}"
            )
        else:
            logger.info(
                f"[{self.name}] All {total_params:,} parameters are trainable "
                f"(full fine-tuning)."
            )

    def _build_dataloader(
        self,
        split_name: str,
        texts: List[str],
        labels: Optional[List[int]],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        assert self.tokenizer is not None, "Tokenizer must be initialized first."

        num_workers = self._num_worker
        logger.info(
            f"[{self.name}] Creating DataLoader for '{split_name}' split "
            f"({len(texts)} samples, batch_size={batch_size}, "
            f"shuffle={shuffle}, num_workers={num_workers})"
        )
        if labels is not None:
            logger.debug(
                f"[{self.name}] '{split_name}' split has labels. "
                f"Unique labels: {sorted(set(labels)) if len(labels) > 0 else '[]'}"
            )
        else:
            logger.debug(f"[{self.name}] '{split_name}' split is unlabeled.")

        dataset = TransformerTextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self._max_seq_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        logger.debug(
            f"[{self.name}] DataLoader for '{split_name}' ready. "
            f"Estimated batches per epoch: {len(loader)}"
        )
        return loader

    def prepare_training(self, train: DatasetSplit) -> DataLoader:
        logger.info(f"[{self.name}] prepare_training called.")
        if self.model is None or self.tokenizer is None:
            logger.warning(
                f"[{self.name}] Components uninitialized during prepare_training. "
                f"Running initialization..."
            )
            self.initialize()

        logger.debug(
            f"[{self.name}] Training DatasetSplit sizes: "
            f"texts={len(train.texts)}, labels={len(train.labels) if train.labels is not None else 'None'}"
        )
        batch_size = self._get_batch_size()
        loader = self._build_dataloader(
            "train", train.texts, train.labels, batch_size=batch_size, shuffle=True
        )
        logger.info(
            f"[{self.name}] Training DataLoader prepared with {len(loader)} batches."
        )
        return loader

    def prepare_validation(self, val: DatasetSplit) -> DataLoader:
        logger.info(f"[{self.name}] prepare_validation called.")
        if self.model is None or self.tokenizer is None:
            logger.warning(
                f"[{self.name}] Components uninitialized during prepare_validation. "
                f"Running initialization..."
            )
            self.initialize()

        logger.debug(
            f"[{self.name}] Validation DatasetSplit sizes: "
            f"texts={len(val.texts)}, labels={len(val.labels) if val.labels is not None else 'None'}"
        )
        batch_size = self._get_batch_size()
        loader = self._build_dataloader(
            "validation", val.texts, val.labels, batch_size=batch_size, shuffle=False
        )
        logger.info(
            f"[{self.name}] Validation DataLoader prepared with {len(loader)} batches."
        )
        return loader

    def prepare(
        self, train: DatasetSplit, val: DatasetSplit
    ) -> TransformerPreparationResult:
        logger.info(f"[{self.name}] Preparing training and validation data streams.")
        prep_result = {}

        train_loader = self.prepare_training(train)
        prep_result["train_loader_batches"] = len(train_loader)
        val_loader = self.prepare_validation(val)
        prep_result["val_loader_batches"] = len(val_loader)

        logger.debug(f"[{self.name}] Preparation summary: {prep_result}")
        return TransformerPreparationResult(
            train_loader=train_loader, val_loader=val_loader
        )

    def train(
        self,
        prepared_result: TransformerPreparationResult,
        *,
        epochs: int = 10,
        **kwargs,
    ) -> TrainResult:
        assert self.model is not None, "Model not initialized"

        logger.info(f"[{self.name}] train() called for {epochs} epochs.")
        logger.debug(f"[{self.name}] Extra train kwargs: {kwargs if kwargs else '{}'}")

        learning_rate = float(self.get_param_value("transformer_learning_rate"))
        weight_decay = float(self.get_param_value("weight_decay"))
        warmup_ratio = float(self.get_param_value("warmup_ratio"))
        max_grad_norm = float(self.get_param_value("max_grad_norm"))

        train_loader_len = len(prepared_result.train_loader)
        total_steps = epochs * train_loader_len
        num_warmup_steps = int(warmup_ratio * total_steps)

        logger.info(f"[{self.name}] Configuring training pipeline:")
        logger.info(
            f"  -> Total Epochs: {epochs} | Steps per Epoch: {train_loader_len}"
        )
        logger.info(
            f"  -> Total Optimization Steps: {total_steps} | Warmup Steps: {num_warmup_steps}"
        )
        logger.info(
            f"  -> Hyperparams: LR={learning_rate}, Weight Decay={weight_decay}, "
            f"Max Grad Norm={max_grad_norm}, Warmup Ratio={warmup_ratio}"
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.debug(
            f"[{self.name}] Number of trainable parameter tensors: "
            f"{len(trainable_params)}"
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"[{self.name}] Handing over control to TransformerTrainer.")
        trainer = TransformerTrainer(
            approach_name=self.name,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self._device,
            train_loader=prepared_result.train_loader,
            val_loader=prepared_result.val_loader,
            num_epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
        self.trainer = trainer

        logger.debug(f"[{self.name}] Starting training routine in TransformerTrainer.")
        trainer_load_path = kwargs.get("load_path") or kwargs.get("trainer_load_path")
        result = trainer.train(load_path=trainer_load_path)
        logger.info(f"[{self.name}] Training routine finished.")
        logger.debug(
            f"[{self.name}] TrainResult summary: "
            f"best_val_metric={getattr(result, 'best_val_metric', None)}, "
            f"epochs_trained={getattr(result, 'epochs_trained', None)}"
        )
        return result

    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None, "Model not initialized"
        assert self.tokenizer is not None, "Tokenizer not initialized"

        logger.info(f"[{self.name}] predict() called.")
        if isinstance(data, DataLoader):
            logger.info(f"[{self.name}] Prediction called with an explicit DataLoader.")
            loader = data
            logger.debug(
                f"[{self.name}] Provided DataLoader has {len(loader)} batches."
            )
        else:
            logger.debug(
                f"[{self.name}] Prediction called with DataFrame of shape {data.shape}."
            )
            assert "text" in data.columns, "DataFrame must contain a 'text' column"
            texts = data["text"].tolist()
            labels = data["label"].tolist() if "label" in data.columns else None

            logger.info(
                f"[{self.name}] Building temporary loader from DataFrame with "
                f"{len(texts)} rows."
            )

            if labels is not None:
                logger.debug(
                    f"[{self.name}] DataFrame includes labels. "
                    f"Unique labels: {sorted(set(labels)) if len(labels) > 0 else '[]'}"
                )
            else:
                logger.debug(
                    f"[{self.name}] DataFrame does not include labels; "
                    f"inference will be unlabeled."
                )

            dataset = TransformerTextDataset(
                texts=texts,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self._max_seq_length,
            )
            loader = DataLoader(
                dataset,
                batch_size=self._get_batch_size(),
                shuffle=False,
                num_workers=self.get_param_value("num_workers"),
                pin_memory=torch.cuda.is_available(),
            )
            logger.debug(
                f"[{self.name}] Temporary prediction DataLoader has {len(loader)} batches."
            )

        logger.info(f"[{self.name}] Starting model inference pass (eval mode)...")
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            total_batches = len(loader)
            logger.debug(f"[{self.name}] Total inference batches: {total_batches}")
            for batch_idx, batch in enumerate(loader):
                logger.debug(
                    f"[{self.name}] Inference batch {batch_idx + 1}/{total_batches} "
                    f"received with keys: {list(batch.keys())}"
                )

                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                logger.debug(
                    f"[{self.name}] Batch {batch_idx + 1}: input_ids.shape="
                    f"{tuple(input_ids.shape)}, attention_mask.shape="
                    f"{tuple(attention_mask.shape)}"
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.append(preds)
                logger.debug(
                    f"[{self.name}] Batch {batch_idx + 1}: logits.shape="
                    f"{tuple(logits.shape)}, preds.shape={preds.shape}"
                )

                if "labels" in batch:
                    labels_np = batch["labels"].cpu().numpy()
                    all_labels.append(labels_np)
                    logger.debug(
                        f"[{self.name}] Batch {batch_idx + 1}: labels.shape={labels_np.shape}"
                    )

                if (batch_idx + 1) % max(1, total_batches // 5) == 0 or (
                    batch_idx + 1
                ) == total_batches:
                    logger.info(
                        f"[{self.name}] Inference Progress: "
                        f"Batch {batch_idx + 1}/{total_batches}"
                    )

        y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        logger.debug(f"[{self.name}] Concatenated predictions shape: {y_pred.shape}")
        if all_labels:
            y_true = np.concatenate(all_labels, axis=0)
            logger.debug(f"[{self.name}] Concatenated labels shape: {y_true.shape}")
            logger.info(
                f"[{self.name}] Inference completed. Processed {len(y_pred)} records "
                f"(Ground truth labels included)."
            )
        else:
            y_true = np.array([])
            logger.info(
                f"[{self.name}] Inference completed. Processed {len(y_pred)} records "
                f"(Unlabeled data)."
            )

        return PredictionResult(y_pred=y_pred, y_true=y_true)
