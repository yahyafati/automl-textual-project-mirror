from pathlib import Path
from typing import Optional, TypedDict, Any, Union

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from automl.core.trainers.base_trainer import Trainer
from automl.core.types import TrainResult, ApproachName, EpochResult
from automl.logger import get_logger

logger = get_logger()


class Checkpoint(TypedDict):
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    epoch: int


class TorchTrainer(Trainer):

    OPTIMIZER_MAPPING: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    SCHEDULER_MAPPING: dict[str, type[_LRScheduler]] = {
        "steplr": StepLR,
        "cosineannealinglr": CosineAnnealingLR,
        "exponentiallr": ExponentialLR,
        "reducelronplateau": ReduceLROnPlateau,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        approach_name: ApproachName,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        *,
        optimizer: torch.optim.Optimizer | str = "adamw",
        optimizer_args: Optional[dict] = None,
        scheduler: _LRScheduler | str | None = None,
        scheduler_args: Optional[dict] = None,
        epochs: int = 5,
    ):
        super().__init__(approach_name)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

        self.optimizer = self.create_optimizer(self.model, optimizer, optimizer_args)
        self.scheduler = self.create_scheduler(
            self.optimizer, scheduler, scheduler_args, epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self._current_epoch = 0

        logger.debug(
            f"Initialized TorchTrainer for '{self.approach_name}' on device: {self.device}. "
            f"Total epochs planned: {self.epochs}"
        )

    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer: str | torch.optim.Optimizer,
        optimizer_args: Optional[dict] = None,
    ):
        if isinstance(optimizer, str):
            opt_class = TorchTrainer.OPTIMIZER_MAPPING.get(optimizer.lower())
            if opt_class is None:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            optimizer_args = optimizer_args or {}
            return opt_class(model.parameters(), **optimizer_args)

        return optimizer

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler: Union[str, _LRScheduler, None],
        scheduler_args: Optional[dict] = None,
        epochs: Optional[int] = None,
    ) -> (
        None
        | _LRScheduler
        | StepLR
        | CosineAnnealingLR
        | ExponentialLR
        | ReduceLROnPlateau
    ):
        if scheduler is None:
            return None

        # already-constructed scheduler
        if not isinstance(scheduler, str):
            return scheduler

        name = scheduler.lower()
        scheduler_args = scheduler_args or {}

        if name == "steplr":
            # Allowed args: step_size (required), gamma=0.1, last_epoch=-1, verbose=False
            step_size = scheduler_args.get("step_size", 1)
            gamma = scheduler_args.get("gamma", 0.1)
            last_epoch = scheduler_args.get("last_epoch", -1)
            return StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
                last_epoch=last_epoch,
            )

        elif name == "cosineannealinglr":
            # Allowed args: T_max (required), eta_min=0, last_epoch=-1, verbose=False
            # use epochs as default T_max if provided
            T_max = scheduler_args.get("T_max", epochs)
            if T_max is None:
                raise ValueError(
                    "CosineAnnealingLR requires 'T_max' or 'epochs' to be set."
                )
            eta_min = scheduler_args.get("eta_min", 0.0)
            last_epoch = scheduler_args.get("last_epoch", -1)
            return CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )

        elif name == "exponentiallr":
            gamma = scheduler_args.get("gamma", 0.1)
            last_epoch = scheduler_args.get("last_epoch", -1)
            return ExponentialLR(
                optimizer,
                gamma=gamma,
                last_epoch=last_epoch,
            )

        elif name == "reducelronplateau":
            # Allowed args (common ones): mode, factor, patience, threshold, threshold_mode,
            # cooldown, min_lr, eps, verbose
            mode = scheduler_args.get("mode", "min")
            factor = scheduler_args.get("factor", 0.1)
            patience = scheduler_args.get("patience", 10)
            threshold = scheduler_args.get("threshold", 1e-4)
            threshold_mode = scheduler_args.get("threshold_mode", "rel")
            cooldown = scheduler_args.get("cooldown", 0)
            min_lr = scheduler_args.get("min_lr", 0.0)
            eps = scheduler_args.get("eps", 1e-8)
            return ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
            )

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    def _compute_loss(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            x = batch["input_ids"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        logits = self.model(x)
        return self.criterion(logits, y)

    def _train_step(self, batch) -> float:
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            total_loss += self._train_step(batch)

        return total_loss / num_batches if num_batches > 0 else 0.0

    def load(self, path: Optional[Path]) -> None:
        if path is None:
            logger.debug("No load path provided. Training from scratch.")
            return

        ckpt_file = Path(path)
        if not ckpt_file.exists():
            logger.warning(
                f"Checkpoint file not found at {ckpt_file}. Starting from scratch."
            )
            return

        logger.debug(f"Found checkpoint at {ckpt_file}. Loading state...")
        checkpoint = torch.load(ckpt_file, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.approach_name = checkpoint["approach_name"]
        self.optimizer = self.create_optimizer(
            self.model,
            checkpoint["optimizer"]["name"],
            checkpoint["optimizer"]["args"],
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"]["state_dict"])

        scheduler_cfg = checkpoint.get("scheduler", None)
        if scheduler_cfg is not None:
            self.scheduler = self.create_scheduler(
                self.optimizer,
                scheduler_cfg["name"],
                scheduler_cfg.get("args"),
                epochs=self.epochs,
            )
            self.scheduler.load_state_dict(scheduler_cfg["state_dict"])
        else:
            self.scheduler = None

        logger.debug(
            f"Successfully resumed from checkpoint. Next epoch: {self.start_epoch + 1}"
        )

    def save(self, path: Path, **kwargs) -> None:
        if path is None:
            logger.debug("No save path provided. Skipping checkpoint saving.")
            return

        epoch = self._current_epoch
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        optimizer_state_dict = self.optimizer.state_dict()
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer": {
                "name": self.optimizer.__class__.__name__.lower(),
                "args": self.optimizer.defaults,
                "state_dict": optimizer_state_dict,
            },
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "approach_name": self.approach_name,
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = {
                "name": self.scheduler.__class__.__name__.lower(),
                # many schedulers store init kwargs in .state_dict() only;
                # if you store them separately, pass them in when you build.
                "args": getattr(self.scheduler, "defaults", {}),
                "state_dict": self.scheduler.state_dict(),
            }

        path = Path(path)
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint successfully saved to {path}")

    def train(
        self,
        load_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
    ) -> TrainResult:
        logger.debug(f"Starting training pipeline (epochs={self.epochs})...")
        if load_path:
            self.load(load_path)

        if self.start_epoch >= self.epochs:
            logger.warning(
                f"Start epoch ({self.start_epoch}) is greater than or equal to total epochs ({self.epochs}). "
                f"Skipping training loop."
            )

        try:
            for epoch in range(self.start_epoch, self.epochs):
                logger.debug(f"--- Epoch {epoch + 1}/{self.epochs} ---")
                self._current_epoch = epoch
                avg_loss = self._run_epoch()

                logger.debug(f"Epoch {epoch + 1} complete. Train Loss: {avg_loss:.4f}")

                val_acc = None
                if self.val_loader is not None:
                    val_acc = self.evaluate()
                    logger.debug(
                        f"Epoch {epoch + 1} complete. Val Accuracy: {val_acc:.4f}"
                    )

                    if val_acc > self.best_val_acc:
                        logger.debug(
                            f" New best validation accuracy achieved: {val_acc:.4f} (was {self.best_val_acc:.4f})"
                        )
                        self.best_val_acc = val_acc
                        if save_path:
                            self.save(save_path)

                self._history.append(
                    EpochResult(
                        epoch=epoch + 1,
                        train_loss=avg_loss,
                        val_accuracy=float(val_acc) if val_acc is not None else None,
                    )
                )

                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        metric = val_acc if val_acc is not None else avg_loss
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()

            logger.info(
                f"Training completed. Best Validation Accuracy: {self.best_val_acc:.4f}"
            )
        except KeyboardInterrupt as err:
            logger.warning(
                f"Training interrupted. Best Validation Accuracy: {self.best_val_acc:.4f}"
            )
            raise
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared.")

        return {"val_accuracy": self.best_val_acc, "history": self.history}

    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0

        logger.debug("Running evaluation on validation set...")
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    x, y = batch["input_ids"].to(self.device), batch["labels"].to(
                        self.device
                    )
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")
                logits = self.model(x)
                batch_labels = y
                batch_preds = torch.argmax(logits, dim=-1)
                preds.extend(batch_preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        return float(accuracy_score(labels, preds, normalize=True))
