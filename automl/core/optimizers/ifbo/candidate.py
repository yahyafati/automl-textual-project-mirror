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
    # Fixed at creation and reused for every freeze-thaw step of this
    # candidate, so the train/val split stays constant across resumes of
    # the same checkpoint. Must stay separate from the per-step training
    # seed, which is free to vary.
    data_seed: int = 0
