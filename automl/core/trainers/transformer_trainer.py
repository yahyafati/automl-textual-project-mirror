from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from automl.core.trainers.base_trainer import Trainer
from automl.core.types import TrainResult, EpochResult, ApproachName
from automl.logger import get_logger

logger = get_logger()


class TransformerTrainer(Trainer):
    def __init__(
        self,
        approach_name: ApproachName,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(approach_name)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.criterion = nn.CrossEntropyLoss()
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self._current_epoch = 0

        logger.info(
            f"[TransformerTrainer] Initialized for approach='{approach_name}' "
            f"on device={device}, num_epochs={num_epochs}, "
            f"max_grad_norm={max_grad_norm}"
        )
        logger.debug(
            f"[TransformerTrainer] Train loader batches={len(train_loader)}, "
            f"Val loader batches={len(val_loader)}, "
            f"Scheduler={'present' if scheduler is not None else 'none'}"
        )

    def train(self, load_path: Optional[Path] = None) -> TrainResult:
        logger.info("[TransformerTrainer] Starting training loop.")
        if load_path is not None:
            logger.info(
                f"[TransformerTrainer] load_path provided. "
                f"Loading checkpoint from {load_path} before training."
            )
            self.load(load_path)

        best_val_acc = self.best_val_acc
        best_state_dict = (
            {k: v.cpu() for k, v in self.model.state_dict().items()}
            if best_val_acc > 0.0
            else None
        )

        total_train_batches = len(self.train_loader)
        logger.debug(
            f"[TransformerTrainer] Total epochs={self.num_epochs}, "
            f"batches per epoch={total_train_batches}"
        )

        for epoch_idx in range(self.start_epoch, self.num_epochs):
            epoch = epoch_idx + 1
            self._current_epoch = epoch_idx
            logger.info(
                f"[TransformerTrainer] ===== Epoch {epoch}/{self.num_epochs} START ====="
            )
            self.model.train()
            total_loss = 0.0
            total_batches = 0

            for batch_idx, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logger.debug(
                    f"[TransformerTrainer] Epoch {epoch} "
                    f"Batch {batch_idx}/{total_train_batches}: "
                    f"input_ids.shape={tuple(input_ids.shape)}, "
                    f"attention_mask.shape={tuple(attention_mask.shape)}, "
                    f"labels.shape={tuple(labels.shape)}"
                )

                # HF models return loss when labels are given
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                logger.debug(
                    f"[TransformerTrainer] Epoch {epoch} "
                    f"Batch {batch_idx}: loss={loss.item():.6f}"
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_batches += 1

                if (batch_idx % max(1, total_train_batches // 5) == 0) or (
                    batch_idx == total_train_batches
                ):
                    logger.info(
                        f"[TransformerTrainer] Epoch {epoch} Progress: "
                        f"{batch_idx}/{total_train_batches} batches "
                        f"({(batch_idx / max(1, total_train_batches)) * 100:.1f}%)"
                    )

            avg_loss = total_loss / max(total_batches, 1)
            logger.info(
                f"[TransformerTrainer] Epoch {epoch} training complete. "
                f"Average train_loss={avg_loss:.4f}. "
                "Starting validation."
            )

            val_acc = self.evaluate()
            logger.info(
                f"[TransformerTrainer] Epoch {epoch}/{self.num_epochs} "
                f"train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}"
            )

            self._history.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=avg_loss,
                    val_accuracy=val_acc,
                )
            )

            if val_acc > best_val_acc:
                logger.info(
                    f"[TransformerTrainer] New best validation accuracy at epoch "
                    f"{epoch}: {val_acc:.4f} (prev best={best_val_acc:.4f}). "
                    f"Saving best state dict to memory."
                )
                best_val_acc = val_acc
                # Store on CPU to avoid GPU-only checkpoint
                best_state_dict = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }

            logger.info(
                f"[TransformerTrainer] ===== Epoch {epoch}/{self.num_epochs} END ====="
            )

        self.best_val_acc = best_val_acc

        if best_state_dict is not None:
            logger.info(
                "[TransformerTrainer] Loading best model state dict "
                f"(best_val_acc={best_val_acc:.4f}) into model."
            )
            self.model.load_state_dict(best_state_dict)
        else:
            logger.warning(
                "[TransformerTrainer] No best_state_dict recorded; "
                "model remains in last-epoch state."
            )

        logger.info(
            f"[TransformerTrainer] Training finished. Best val_accuracy={best_val_acc:.4f}"
        )

        return TrainResult(
            val_accuracy=float(best_val_acc),
            history=self.history,
        )

    def evaluate(self) -> float:
        logger.debug("[TransformerTrainer] Starting evaluation on validation set.")
        self.model.eval()
        correct = 0
        total = 0

        total_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                batch_correct = (preds == labels).sum().item()
                batch_total = labels.size(0)

                correct += batch_correct
                total += batch_total

                logger.debug(
                    f"[TransformerTrainer] Eval batch {batch_idx}/{total_batches}: "
                    f"batch_correct={batch_correct}, batch_total={batch_total}"
                )

        val_acc = correct / total if total > 0 else 0.0
        logger.info(
            f"[TransformerTrainer] Evaluation complete. "
            f"Total samples={total}, correct={correct}, "
            f"val_accuracy={val_acc:.4f}"
        )
        return val_acc

    def save(self, path: Path, **kwargs) -> None:
        logger.info(
            f"[TransformerTrainer] Saving model checkpoint to {path} "
            f"(extra_kwargs={list(kwargs.keys()) if kwargs else []})."
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        epoch = max(
            self.start_epoch,
            self._current_epoch + 1 if self._history else 0,
        )
        state = {
            "model_state_dict": {
                k: v.cpu() for k, v in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "history": self._history,
        }
        torch.save(state, path)
        logger.info(f"[TransformerTrainer] Saved model checkpoint to {path}")

    def load(self, path: Path) -> None:
        logger.info(f"[TransformerTrainer] Loading model checkpoint from {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_epoch = state.get("epoch", 0)
        self.best_val_acc = state.get("best_val_acc", 0.0)
        self._history = state.get("history", [])
        logger.info(f"[TransformerTrainer] Loaded model checkpoint from {path}")
