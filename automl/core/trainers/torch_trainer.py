import itertools
import uuid
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

    OPTIMIZER_MAPPING = {
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
        evaluate_validation: bool = True,
        max_grad_norm: Optional[float] = None,
        warmup_ratio: float = 0.0,
        stochastic_epochs: bool = False,
        stochastic_epoch_fraction: Optional[float] = None,
    ):
        super().__init__(approach_name)
        self.trainer_id = uuid.uuid4().hex[:8]
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.evaluate_validation = evaluate_validation
        self.max_grad_norm = max_grad_norm

        # Stochastic epochs: each "epoch" trains on a random fraction of
        # the training data instead of a full pass, so an epoch-budgeted
        # fidelity (as used by ifBO's freeze-thaw scheduler) buys cheaper,
        # more numerous gradient-update steps rather than being dominated
        # by full-dataset passes. See `_run_epoch`.
        self.stochastic_epochs = stochastic_epochs
        if self.stochastic_epochs:
            fraction = (
                stochastic_epoch_fraction
                if stochastic_epoch_fraction is not None
                else 0.25
            )
            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    "stochastic_epoch_fraction must be in (0, 1], got " f"{fraction}"
                )
            self.stochastic_epoch_fraction = fraction
        else:
            self.stochastic_epoch_fraction = None

        self.optimizer = self.create_optimizer(self.model, optimizer, optimizer_args)
        # Snapshot the target LR per param group *before* warmup starts
        # scaling it down, so we know what to warm up towards/back to.
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.warmup_epochs = (
            int(round(warmup_ratio * epochs)) if warmup_ratio > 0 else 0
        )
        self.scheduler = self.create_scheduler(
            self.optimizer, scheduler, scheduler_args, epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self._current_epoch = 0

        logger.debug(
            f"Initialized TorchTrainer for '{self.approach_name}' on device: {self.device.type}. "
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
            lr = optimizer_args.get("lr", 1e-3)
            weight_decay = optimizer_args.get("weight_decay", 1e-4)
            momentum = optimizer_args.get("momentum", 0.9)
            if opt_class == torch.optim.SGD:
                return torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                )
            return opt_class(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        logger.debug(
            f"Starting epoch {self._current_epoch + 1} with {num_batches} batches."
        )

        if self.stochastic_epochs and num_batches > 0:
            # Take only the first `steps_per_epoch` batches of a *freshly
            # shuffled* pass over train_loader (it reshuffles on every new
            # iteration) instead of the whole dataset. That's a different
            # random subsample each call, so a fixed epoch budget still
            # covers the full dataset in expectation over several epochs,
            # while each individual epoch is proportionally cheaper.
            steps_per_epoch = max(
                1, round(num_batches * self.stochastic_epoch_fraction)
            )
            batch_iter = itertools.islice(self.train_loader, steps_per_epoch)
            logger.debug(f"Using stochastic epoch with {steps_per_epoch} batches.")
        else:
            batch_iter = self.train_loader

        num_seen = 0
        for batch in batch_iter:
            total_loss += self._train_step(batch)
            num_seen += 1

            if num_seen % 50 == 0:
                logger.debug(
                    f"Epoch {self._current_epoch + 1}, Batch {num_seen}: Loss = {total_loss / num_seen:.4f}"
                )

        return total_loss / num_seen if num_seen > 0 else 0.0

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
        self._history = checkpoint.get("history", [])
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

        epoch = max(
            self.start_epoch,
            self._current_epoch + 1 if self._history else 0,
        )
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
            "history": self._history,
            "trainer_id": self.trainer_id,
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
                logger.debug(
                    f"--- [{self.trainer_id}] Epoch {epoch + 1}/{self.epochs} ---"
                )
                self._current_epoch = epoch

                in_warmup = epoch < self.warmup_epochs
                if in_warmup:
                    warmup_factor = (epoch + 1) / self.warmup_epochs
                    for group, base_lr in zip(
                        self.optimizer.param_groups, self._base_lrs
                    ):
                        group["lr"] = base_lr * warmup_factor

                avg_loss = self._run_epoch()

                logger.debug(
                    f"[{self.trainer_id}]: Epoch {epoch + 1} complete. Train Loss: {avg_loss:.4f}"
                )

                val_acc = None
                if self.evaluate_validation and self.val_loader is not None:
                    val_acc = self.evaluate()
                    logger.debug(
                        f"[{self.trainer_id}]: Epoch {epoch + 1} complete. Val Accuracy: {val_acc:.4f}"
                    )

                    if val_acc > self.best_val_acc:
                        logger.debug(
                            f"[{self.trainer_id}]: New best validation accuracy achieved: {val_acc:.4f} (was {self.best_val_acc:.4f})"
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

                # Hold the main scheduler off until warmup has finished so it
                # doesn't immediately override the warmed-up LR.
                if self.scheduler is not None and not in_warmup:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        metric = val_acc if val_acc is not None else avg_loss
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()

            logger.info(
                f"[{self.trainer_id}]: Training completed. Best Validation Accuracy: {self.best_val_acc:.4f}"
            )
        except KeyboardInterrupt as err:
            logger.warning(
                f"[{self.trainer_id}]: Training interrupted. Best Validation Accuracy: {self.best_val_acc:.4f}"
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
