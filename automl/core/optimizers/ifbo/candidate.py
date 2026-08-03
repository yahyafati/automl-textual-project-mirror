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
    z: torch.Tensor
    steps_done: int = 0
    ts: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    uid: int = -1
