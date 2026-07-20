from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

from automl.core.types import TrainResult, ApproachName, EpochResult
from automl.logger import get_logger

logger = get_logger()


class Trainer(ABC):

    def __init__(self, approach_name: ApproachName):
        self.approach_name: ApproachName = approach_name
        self._history: List[EpochResult] = []

    @property
    def history(self) -> List[EpochResult]:
        return deepcopy(self._history)

    @abstractmethod
    def train(self, load_path: Optional[Path] = None) -> TrainResult: ...

    @abstractmethod
    def evaluate(self) -> float: ...

    @abstractmethod
    def save(self, path: Path, **kwargs) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
