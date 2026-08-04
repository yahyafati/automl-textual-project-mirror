from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TypedDict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from ConfigSpace import Configuration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from automl.core.approaches.base_approach import Approach
from automl.core.registry import register_approach
from automl.core.trainers.torch_trainer import TorchTrainer
from automl.core.types import DatasetSplit, PredictionResult, TrainResult
from automl.core.utils import timer
from automl.core.utils.misc import set_seed
from automl.logger import get_logger

logger = get_logger()


class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, dropout: float = 0):
        super().__init__()
        hidden = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class _SparseIndexDataset(Dataset):
    """Dataset that yields row indices only; densification happens per-batch
    in the collate_fn. Avoids ever materializing the full dense matrix
    (X_train.toarray() on a 10k+ vocab can be the single biggest memory
    spike in this pipeline)."""

    def __init__(self, n_rows: int) -> None:
        self.n_rows = n_rows

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, idx: int) -> int:
        return idx


def _make_sparse_collate(X: sp.csr_matrix, y: np.ndarray):
    def collate(indices: list[int]):
        idx = np.asarray(indices)
        X_batch = X[idx].toarray()
        y_batch = y[idx]
        return (
            torch.tensor(X_batch, dtype=torch.float32),
            torch.tensor(y_batch, dtype=torch.long),
        )

    return collate


class _PreparationResult(TypedDict):
    model: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader


@register_approach("tfidf-ffnn")
class TfidfFFNNApproach(Approach[torch.nn.Module, _PreparationResult]):

    def __init__(
        self,
        config: Configuration,
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, num_classes, device, **kwargs)
        logger.debug("Initializing TfidfApproach...")
        self.vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.representation: str = "word"
        self.trainer: Optional[TorchTrainer] = None

    @property
    def use_char_ngrams(self):
        return self.representation in ["char", "hybrid"]

    def initialize(self):
        seed = self.get_param_value("seed")
        set_seed(seed)

        max_features = self.get_param_value("vocab_size")
        ngram_range = tuple((1, self.get_param_value("ngram_max")))
        analyzer = self.get_param_value("analyzer")
        stop_words = self.get_param_value("stop_words")
        min_df = self.get_param_value("min_df")
        max_df = self.get_param_value("max_df")
        sublinear_tf = self.get_param_value("sublinear_tf")
        use_idf = self.get_param_value("use_idf")
        norm = self.get_param_value("norm")

        self.representation = self.get_param_value("representation")
        logger.debug(f"self.representation set to {self.representation}")

        char_ngram_range = (
            self.get_param_value("char_ngram_min"),
            self.get_param_value("char_ngram_max"),
        )
        # When combining word + char features, split the feature budget so
        # the two vectorizers don't independently blow past vocab_size.
        # Checkout notes.md#1

        word_max_features = max_features
        char_max_features = max_features
        match self.representation:
            case "word":
                word_max_features = max_features
                char_max_features = 0
            case "char":
                word_max_features = 0
                char_max_features = max_features
            case "hybrid":
                word_max_features = max_features // 2
                char_max_features = max_features - word_max_features

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=word_max_features,
            ngram_range=ngram_range,
            analyzer=analyzer,
            # TODO: There is apparently an issue with commonly used stop-words as indicated:
            #  https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            norm=norm,
        )

        if self.use_char_ngrams:
            self.char_vectorizer = TfidfVectorizer(
                lowercase=True,
                max_features=char_max_features,
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
                use_idf=use_idf,
                norm=norm,
            )

    def prepare_training(self, train: DatasetSplit):
        assert self.vectorizer is not None
        X_train = self.vectorizer.fit_transform(train.texts)
        if self.use_char_ngrams:
            assert self.char_vectorizer is not None
            X_train_char = self.char_vectorizer.fit_transform(train.texts)
            X_train = sp.hstack([X_train, X_train_char])

        X_train = X_train.tocsr()
        y_train = np.array(train.labels, dtype=np.int64)
        return X_train, y_train

    def prepare_validation(self, val: DatasetSplit):
        assert self.vectorizer is not None
        X_val = self.vectorizer.transform(val.texts)
        if self.use_char_ngrams:
            assert self.char_vectorizer is not None
            X_val_char = self.char_vectorizer.transform(val.texts)
            X_val = sp.hstack([X_val, X_val_char])

        X_val = X_val.tocsr()
        y_val = np.array(val.labels, dtype=np.int64)
        return X_val, y_val

    def prepare(self, train: DatasetSplit, val: DatasetSplit) -> _PreparationResult:
        max_features = self.get_param_value("vocab_size")
        batch_size = int(self.get_param_value("batch_size"))
        check_balance = self.get_param_value("class_balance")
        dropout = self.get_param_value("dropout")
        hidden_dim = self.get_param_value("hidden_dim")

        self.initialize()
        assert self.vectorizer is not None

        with timer.Timer("TFIDF Linear Vectorization") as t:
            X_train, y_train = self.prepare_training(train)
            X_val, y_val = self.prepare_validation(val)
        logger.debug(
            "FFNN approach: training vectorization took %.2fs", t.execution_time
        )

        actual_features = X_train.shape[1]
        logger.debug(
            "TF-IDF fitted: actual_features=%d (requested max=%d), sparsity=%.4f%%",
            actual_features,
            max_features,
            100.0 * (1.0 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])),
        )
        logger.debug(
            "Labels: train unique=%s, val unique=%s",
            np.unique(y_train),
            np.unique(y_val),
        )

        train_dataset = _SparseIndexDataset(X_train.shape[0])
        val_dataset = _SparseIndexDataset(X_val.shape[0])

        sampler = None
        shuffle = True
        if check_balance:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(y_train),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler
            logger.debug("Class balancing enabled via WeightedRandomSampler")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=_make_sparse_collate(X_train, y_train),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_make_sparse_collate(X_val, y_val),
        )

        logger.debug(
            "Data loaders built: train=%d batches, val=%d batches (batch_size=%d)",
            len(train_loader),
            len(val_loader),
            batch_size,
        )

        model = SimpleFFNN(
            actual_features,
            output_dim=self._num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model.to(self._device)
        self.model = model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug(
            "Model parameters: total=%d, trainable=%d", total_params, trainable_params
        )

        return {"model": model, "train_loader": train_loader, "val_loader": val_loader}

    def train(
        self,
        prepared_result: _PreparationResult,
        *,
        trainer_load_path: Optional[Path] = None,
        epochs: int = 50,
        **kwargs,
    ) -> TrainResult:
        logger.debug("Starting TF-IDF model training...")
        optimizer_name: str = self.get_param_value("optimizer")

        opt_kwargs: dict[str, Any] = {
            "lr": self.get_param_value("learning_rate"),
            "weight_decay": self.get_param_value("weight_decay"),
        }

        if optimizer_name in ["adam", "adamw"]:
            b1 = self.get_param_value("beta1")
            b2 = self.get_param_value("beta2")
            opt_kwargs["betas"] = (b1, b2)
        elif optimizer_name == "sgd":
            opt_kwargs["momentum"] = self.get_param_value("momentum")

        with timer.Timer("TFIDF FFNN Approach Training") as t:
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
            "Training completed in %.2fs. Best metric: %.4f",
            t.execution_time,
            result.get("val_accuracy", float("nan")),
        )
        return result

    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        assert self.model is not None

        val_loader = data
        if isinstance(val_loader, pd.DataFrame):
            assert self.vectorizer is not None, "Vectorizer is not initialized."
            X_val = self.vectorizer.transform(val_loader["text"].tolist())
            if self.use_char_ngrams and self.char_vectorizer is not None:
                X_val_char = self.char_vectorizer.transform(val_loader["text"].tolist())
                X_val = sp.hstack([X_val, X_val_char]).tocsr()
            else:
                X_val = X_val.tocsr()
            y_val = np.array(val_loader["label"].tolist(), dtype=np.int64)
            batch_size = self.get_param_value("batch_size")
            val_loader = DataLoader(
                _SparseIndexDataset(X_val.shape[0]),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_make_sparse_collate(X_val, y_val),
            )

        logger.debug(
            "Starting prediction: %d batches on device=%s",
            len(val_loader),
            self._device,
        )
        self.model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(self._device), batch[1].to(self._device)
                logits = self.model(x)
                labels.extend(y.cpu().numpy())

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        logger.debug(f"Prediction loop finished. Evaluated {len(preds)} samples.")
        y_pred, y_true = np.array(preds), np.array(labels)
        logger.debug("Prediction completed: %d predictions generated", len(y_pred))
        return {"y_pred": y_pred, "y_true": y_true}
