from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Optional

import numpy as np
import pandas as pd
import torch
from ConfigSpace import Configuration
from scipy.sparse import csr_array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.sklearn_trainer import SklearnTrainer
from automl.core.types import DatasetSplit, PredictionResult, TrainResult
from automl.core.utils import timer
from automl.core.utils.misc import get_device
from automl.logger import get_logger

logger = get_logger()


class _PreparationResult(TypedDict):
    X_train: csr_array
    y_train: np.ndarray
    X_val: csr_array
    y_val: np.ndarray


@register_approach("tfidf-linear")
class TfidfLinearApproach(Approach[SGDClassifier, _PreparationResult]):
    """
    Cheap, sparse-native linear baseline. Useful as:
    (a) a strong reference point TF-IDF+FFNN should beat, and
    (b) a low-fidelity proxy in multi-fidelity HPO -- an SGDClassifier
        epoch (`max_iter`) is orders of magnitude cheaper than a FFNN
        epoch, so it's a good way to screen representation/preprocessing
        choices before committing budget to the neural arm.
    """

    def __init__(
        self,
        config: Configuration,
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        device = device or get_device()
        super().__init__(config, num_classes, device, **kwargs)
        self.vectorizer: TfidfVectorizer | None = None
        self.trainer: Optional[SklearnTrainer] = None

    def initialize(self):
        max_features = self.get_param_value("vocab_size")
        ngram_range = tuple((1, self.get_param_value("ngram_max")))
        alpha = self.get_param_value("alpha")
        max_iter = self.get_param_value("epochs")  # reuse "epochs" as the fidelity dim
        class_balance = self.get_param_value("class_balance")
        seed = self.get_param_value("seed")

        # Multiplying by some number here, because it kept stopping before convergence.
        #  Also, since this is a Linear Model, much faster NN, I think it is only fair.
        max_iter = max(max_iter, 1) * 40

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=self.get_param_value("min_df"),
            max_df=self.get_param_value("max_df"),
            sublinear_tf=self.get_param_value("sublinear_tf"),
        )

        # TODO: Maybe add loss to `configspace`
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            class_weight="balanced" if class_balance else None,
            random_state=seed,
            n_jobs=-1,
        )

    def prepare_training(self, train: DatasetSplit):
        assert self.vectorizer is not None
        X_train = self.vectorizer.fit_transform(train.texts)
        y_train = np.array(train.labels, dtype=np.int64)

        return X_train.tocsr(), y_train

    def prepare_validation(self, val: DatasetSplit):
        assert self.vectorizer is not None
        X_val = self.vectorizer.transform(val.texts)
        y_val = np.array(val.labels, dtype=np.int64)

        return X_val.tocsr(), y_val

    def prepare(self, train: DatasetSplit, val: DatasetSplit) -> _PreparationResult:
        self.initialize()

        with timer.Timer("TFIDF Linear Vectorization") as t:
            X_train, y_train = self.prepare_training(train)
            X_val, y_val = self.prepare_validation(val)

        logger.debug(
            "Linear approach: training vectorization took %.2fs", t.execution_time
        )

        return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}

    def train(
        self,
        prepared_result: _PreparationResult,
        load_path: Optional[Path] = None,
        **kwargs,
    ) -> TrainResult:
        assert (
            self.model is not None
        ), "Model is not initialized, call self.initialize() first"

        with timer.Timer("TFIDF Linear Approach Training") as t:
            trainer = SklearnTrainer(
                self.model,
                self.name,
                prepared_result["X_train"],
                prepared_result["y_train"],
                prepared_result["X_val"],
                prepared_result["y_val"],
            )
            self.trainer = trainer
            result = trainer.train(load_path)

        val_accuracy = self.model.score(
            prepared_result["X_val"],
            prepared_result["y_val"],
        )
        logger.debug(
            "Linear approach trained in %.2fs, val_accuracy=%.4f",
            t.execution_time,
            val_accuracy,
        )

        return TrainResult(val_accuracy=result["val_accuracy"], history=trainer.history)

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if isinstance(data, pd.DataFrame):
            assert (
                self.vectorizer is not None
            ), "self.vectorizer is None, Did you run self.train()?"
            X = self.vectorizer.transform(data["text"].tolist())
            y_true = np.array(data["label"].tolist(), dtype=np.int64)
        else:
            # Fall back to whatever was bundled at prepare-time.
            X, y_true = data["X_val"], data["y_val"]

        assert self.model is not None
        y_pred = self.model.predict(X)
        return {"y_pred": y_pred, "y_true": y_true}
