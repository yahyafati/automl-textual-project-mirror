from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from automl.core.trainers.base_trainer import Trainer
from automl.core.types import TrainResult, ApproachName, EpochResult
from automl.logger import get_logger

logger = get_logger()


class SklearnTrainer(Trainer):
    def __init__(
        self,
        model,
        approach_name: ApproachName,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ):
        super().__init__(approach_name)
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.best_val_acc = 0.0
        self.classes_ = np.unique(y_train)

    def load(self, path: Optional[Path]) -> None:
        if path is None:
            logger.debug("No load path provided. Training from scratch.")
            return

        if not path.exists():
            logger.warning(
                f"Checkpoint file not found at {path}. Training from scratch."
            )
            return

        checkpoint = joblib.load(path)
        self.model = checkpoint["model"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self._history = checkpoint.get("history", [])

    def save(self, path: Path, **kwargs) -> None:
        if path is None:
            logger.debug("No save path provided. Skipping checkpoint saving.")
            return

        checkpoint = {
            "model": self.model,
            "best_val_acc": self.best_val_acc,
            "history": self._history,
        }

        joblib.dump(checkpoint, path)

    def train(self, load_path: Optional[Path] = None) -> TrainResult:
        self.load(load_path)
        # Use partial_fit for epoch-like behavior
        self.model.fit(self.X_train, self.y_train)
        train_cost = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(self.X_train)
            # Log loss heavily penalizes confident but incorrect predictions,
            # making it a much better metric for probabilistic models than raw accuracy.
            train_cost = log_loss(self.y_train, probs)

        val_acc = None
        if self.X_val is not None:
            val_acc = self.evaluate()

        # Since we are doing a single full fit, history just gets one final entry
        self._history.append(
            EpochResult(epoch=0, train_loss=train_cost, val_accuracy=val_acc)
        )

        # This is more for the Trainer's structural similarity with other trainers that train over epochs
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # Note: ensure you return self._history if that is the actual attribute name
        return {"val_accuracy": self.best_val_acc, "history": self._history}

    def evaluate(self) -> float:
        if self.X_val is None:
            return 0.0
        preds = self.model.predict(self.X_val)
        return accuracy_score(self.y_val, preds)
