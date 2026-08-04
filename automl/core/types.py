from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Literal, List, Optional, Any

import numpy as np
import pandas as pd

ApproachName = Literal["sequence-dl", "transformer", "tfidf-ffnn"]


@dataclass
class DatasetSplit:
    texts: List[str]
    labels: List[int]

    @property
    def size(self) -> int:
        assert len(self.texts) == len(
            self.labels
        ), "The sizes of the text and labels is different"
        return len(self.texts)

    @staticmethod
    def from_df(df: pd.DataFrame) -> "DatasetSplit":
        assert {"text", "label"}.issubset(
            df.columns
        ), f"Missing required columns. Expected: 'text', 'label'. Found: {list(df.columns)}"

        return DatasetSplit(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
        )


class EpochResult(TypedDict):
    epoch: int
    train_loss: Optional[float]
    val_accuracy: Optional[float]


class PredictionResult(TypedDict):
    y_pred: np.ndarray
    y_true: np.ndarray


class TrainResult(TypedDict):
    val_accuracy: float
    history: List[EpochResult]


class EvaluationResult(TypedDict):
    train_result: TrainResult
    prediction_result: PredictionResult


class TrialResult(TypedDict):
    config: dict[str, Any]
    seed: int
    budget: float
    trialNo: int
    execution_time: float
    timestamp: str
    val_error: float
    best_so_far: bool  # is_best_yet
    epoch_history: list[EpochResult]
