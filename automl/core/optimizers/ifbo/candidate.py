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
    # Plateau tracking (see IfboOptimizer._update_plateau_state): best
    # observed accuracy and how many consecutive steps since it last
    # improved by more than ifbo_min_delta.
    best_y: float = float("-inf")
    no_improve_steps: int = 0
    # Set once this candidate plateaus; excluded from future selection in
    # IfboOptimizer._select_next_candidate but remains eligible as an
    # incumbent based on its best observed accuracy.
    stopped: bool = False
