from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar, Generic, Literal, Optional, Any, Generator, Union, Callable

import joblib
import pandas as pd
import torch
from ConfigSpace import Configuration
from torch.utils.data import DataLoader

from automl.core.approaches.constants import get_approach_defaults
from automl.core.trainers.base_trainer import Trainer
from automl.core.types import (
    ApproachName,
    DatasetSplit,
    TrainResult,
    PredictionResult,
)
from automl.core.utils.misc import get_device, numpy_and_config_encoder
from automl.logger import get_logger

TModel = TypeVar("TModel")
TPreparationResult = TypeVar("TPreparationResult")

Mode = Literal["train", "eval"]

logger = get_logger()


class Approach(ABC, Generic[TModel, TPreparationResult]):
    """
    Strategy interface for an individual text classification approach.
    Implementations are stateless or lightly stateful wrappers around
    model construction, data preparation, training, and prediction.
    """

    name: ApproachName

    def __init__(
        self,
        config: Union[Configuration, dict],
        num_classes: int,
        device: Optional[torch.device] = None,
        approach_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self._mode: Mode = "train"
        self._device: torch.device = device or get_device()
        self._params: dict[str, Any] = dict(config)
        self._approach_params: dict[str, Any] = dict(approach_params or {})
        self._params.update(self._approach_params)
        self._num_classes: int = num_classes
        self.model: Optional[TModel] = None
        self.trainer: Optional[Trainer] = None

        self._default_params: dict[str, Any] = dict(get_approach_defaults(self.name))
        self._kwargs = kwargs

    def get_param_value(
        self,
        param_name: str,
        default: Any = None,
        apply_fn: Optional[Callable[[Any], Any]] = None,
    ):
        if param_name in self._params:
            value = self._params[param_name]
        elif param_name in self._approach_params:
            value = self._approach_params[param_name]
        elif param_name in self._default_params:
            logger.debug(f"Using default value for parameter '{param_name}': {default}")
            value = self._default_params.get(param_name, default)
        else:
            value = default
            logger.warning(
                f"Parameter '{param_name}' is not specified and has no default value."
            )
        if apply_fn:
            value = apply_fn(value)
        return value

    @contextmanager
    def with_mode(self, mode: Mode) -> Generator["Approach", Any, None]:
        old_mode = self._mode
        self._mode = mode
        try:
            yield self
        finally:
            self._mode = old_mode

    @abstractmethod
    def initialize(self):
        """ """
        ...

    @abstractmethod
    def prepare_training(self, train: DatasetSplit):
        """ """
        ...

    @abstractmethod
    def prepare_validation(self, val: DatasetSplit):
        """ """
        ...

    @abstractmethod
    def prepare(self, train: DatasetSplit, val: DatasetSplit) -> TPreparationResult:
        """
        Build the model and corresponding DataLoaders for train/val.
        """
        ...

    @abstractmethod
    def train(
        self,
        prepared_result: TPreparationResult,
        **kwargs,
    ) -> TrainResult:
        """
        Run the training loop and return metrics (e.g. best val accuracy).
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame | DataLoader) -> PredictionResult:
        """
        Run prediction on a DataLoader and return predictions and labels.
        """
        ...

    @staticmethod
    def load(checkpoint_dir_path: str | Path) -> "Approach":
        """
        Static constructor to load an Approach from a checkpoint directory.

        This restores:
          - the approach type (via the registry)
          - the hyperparameter configuration (`config`)

        It does NOT:
          - rebuild the model or DataLoaders (call `prepare(...)` yourself)
          - automatically restore the trainer (you do that after you have
            model + loaders and have attached a Trainer).

        Usage:
            from src.automl.registry import register_all_approaches

            register_all_approaches()
            approach = Approach.load("/path/to/checkpoint_dir")
            # later, with data:
            prep = approach.prepare(train_split, val_split)
            # then build a trainer and resume from trainer.pth if desired.
        """
        from automl.core.registry import get_approach  # local import to avoid cycles

        checkpoint_dir_path = Path(checkpoint_dir_path)

        approach_pth = checkpoint_dir_path / "approach.pth"

        if not approach_pth.exists():
            raise FileNotFoundError(
                f"No approach checkpoint found in {checkpoint_dir_path}. "
                f"Expected 'approach.pth' or 'approach.json'."
            )

        payload: dict[str, Any] | None = None

        if approach_pth.exists():
            try:
                payload = joblib.load(approach_pth)
                logger.debug(f"Loaded approach metadata from {approach_pth}")
            except Exception as e:
                logger.error(
                    f"Failed to load approach metadata from {approach_pth}: {e}"
                )

        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid approach payload: expected dict, got {type(payload)}"
            )

        name = payload.get("approach")
        config = payload.get("config", {})
        num_classes = int(payload.get("num_classes"))  # type: ignore
        device = payload.get("device", get_device())
        kwargs = payload.get("kwargs", {})

        device = torch.device(device) if isinstance(device, str) else device

        if name is None:
            raise ValueError("Checkpoint payload missing 'approach' field")

        if num_classes is None:
            raise ValueError("Checkpoint payload missing 'num_classes' field")

        # Lookup concrete Approach class from registry
        ApproachCls = get_approach(name)  # type: ignore[type-abstract]

        approach: Approach = ApproachCls(
            config=config,
            num_classes=num_classes,
            device=device,
            **kwargs,
        )
        logger.debug(
            f"Instantiated approach '{name}' from checkpoint with "
            f"{len(config)} hyperparameters."
        )

        return approach

    def save(self, checkpoint_dir_path: str | Path, replace_best: bool = False):
        checkpoint_dir_path = Path(checkpoint_dir_path) / self.name
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "approach": self.name,
            "class": self.__class__.__name__,
            "num_classes": self._num_classes,
            "kwargs": self._kwargs,
            "config": dict(self._params),
            "device": str(self._device),
        }
        joblib.dump(payload, checkpoint_dir_path / "approach.pth")
        logger.debug(f"Approach saved to {checkpoint_dir_path / 'approach.pth'}")

        # try saving it as json to
        try:
            with open(checkpoint_dir_path / "approach.json", "w") as f:
                json.dump(payload, f, default=numpy_and_config_encoder)
            logger.debug(
                f"Approach saved as JSON to {checkpoint_dir_path / 'approach.json'}"
            )
        except Exception as e:
            logger.error(f"Failed to save approach as JSON: {e}")

        if self.trainer is not None:
            self.trainer.save(checkpoint_dir_path / "trainer.pth")

        if replace_best:
            (checkpoint_dir_path.parent / "best").write_text(
                str(checkpoint_dir_path.name)
            )

        logger.debug(f"Approach saved to {checkpoint_dir_path}")
