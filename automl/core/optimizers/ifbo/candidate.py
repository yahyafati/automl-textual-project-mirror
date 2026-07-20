"""
Internal candidate representation for ifBO in this AutoML setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from ConfigSpace import Configuration


@dataclass
class IfBOCandidate:
    config: Configuration
    z: torch.Tensor  # encoded hyperparameters in [0, 1]^d
    steps_done: int = 0  # how many freeze-thaw "steps" already evaluated
    ts: list[float] = field(default_factory=list)  # normalized time steps for FT-PFN
    ys: list[float] = field(
        default_factory=list
    )  # observed performance (accuracy in [0,1])
