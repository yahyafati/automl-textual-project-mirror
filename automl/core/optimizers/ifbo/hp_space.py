"""
Small hyperparameter-space abstraction.

Encodes ConfigSpace-style hyperparameters into a normalized [0, 1] scalar
(and a full configuration into a [0, 1]^d vector) so it can be consumed by
the FT-PFN surrogate.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import torch

# FT-PFN surrogate can only consume this many hyperparameter dimensions.
MAX_HYPERPARAMETERS = 10

logger = logging.getLogger()


class HPSpec:
    """Base class for a single hyperparameter's type/range."""

    def encode(self, value: Any) -> float:
        raise NotImplementedError

    def decode(self, u: float) -> Any:
        raise NotImplementedError

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError


class Float(HPSpec):
    """Continuous hyperparameter, optionally encoded on a log scale."""

    def __init__(self, low: float, high: float, log: bool = False):
        self.low = float(low)
        self.high = float(high)
        self.log = bool(log)

    def encode(self, value: float) -> float:
        lo, hi = (
            (math.log(self.low), math.log(self.high))
            if self.log
            else (self.low, self.high)
        )
        v = math.log(value) if self.log else float(value)
        return min(1.0, max(0.0, (v - lo) / (hi - lo)))

    def decode(self, u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(lo + u * (hi - lo))
        return self.low + u * (self.high - self.low)

    def sample(self, rng: random.Random) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(rng.uniform(lo, hi))
        return rng.uniform(self.low, self.high)


class Integer(HPSpec):
    """Integer hyperparameter, optionally encoded on a log scale."""

    def __init__(self, low: int, high: int, log: bool = False):
        self.low = int(low)
        self.high = int(high)
        self.log = bool(log)

    def encode(self, value: int) -> float:
        lo, hi = (
            (math.log(self.low), math.log(self.high))
            if self.log
            else (self.low, self.high)
        )
        v = math.log(value) if self.log else float(value)
        return min(1.0, max(0.0, (v - lo) / (hi - lo)))

    def decode(self, u: float) -> int:
        u = min(1.0, max(0.0, float(u)))
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            val = math.exp(lo + u * (hi - lo))
        else:
            val = self.low + u * (self.high - self.low)
        return int(round(min(self.high, max(self.low, val))))

    def sample(self, rng: random.Random) -> int:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            val = math.exp(rng.uniform(lo, hi))
        else:
            val = rng.uniform(self.low, self.high)
        return int(round(min(self.high, max(self.low, val))))


class Categorical(HPSpec):
    """Unordered categorical hyperparameter, encoded into equal-width bins."""

    def __init__(self, choices: tuple[Any, ...]):
        if not choices:
            raise ValueError("Categorical needs at least one choice.")
        self.choices = tuple(choices)

    def encode(self, value: Any) -> float:
        idx = self.choices.index(value)
        # Centre of the idx-th bin to avoid exact 0/1
        return (idx + 0.5) / len(self.choices)

    def decode(self, u: float) -> Any:
        u = float(u)
        idx = min(len(self.choices) - 1, max(0, int(u * len(self.choices))))
        return self.choices[idx]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)


class HyperparameterSpace:
    """
    Ordered collection of HPSpecs; encodes ConfigSpace configurations into [0,1]^d
    for FT-PFN. At most 10 dimensions (limitation of current FTPFN versions).
    """

    def __init__(self, **specs: HPSpec):
        if len(specs) == 0:
            raise ValueError("HyperparameterSpace needs at least one hyperparameter.")

        all_names = list(specs.keys())

        # `model_type` is dropped unconditionally regardless of how much
        # headroom the space has - it's a `ConfigSpace.Constant`, constant
        # for the whole run, and so carries zero information for the
        # surrogate no matter the dimension budget. In this project it's
        # already filtered out earlier too (see `_build_hp_space`); this is
        # a defensive second filter for any other caller of this class.
        all_names = [name for name in all_names if name != "model_type"]

        # Everything else is only dropped if the space still doesn't fit
        # MAX_HYPERPARAMETERS afterwards, and then only as many - in this
        # order, least-useful-first - as needed to fit. This is a priority
        # list, not an unconditional drop list: e.g. `transformer`'s space
        # already fits within the cap without dropping `warmup_ratio`, so it
        # keeps it, while `sequence-dl` needs both entries dropped to fit.
        # See `docs/IFBO_METHOD.md` §3 for the reasoning behind this
        # ordering (a judgment call, not a rigorous ablation).
        drop_priority = ["warmup_ratio", "seq_num_layers"]

        dropped_names: list[str] = []
        for name in drop_priority:
            if len(all_names) <= MAX_HYPERPARAMETERS:
                break
            if name in all_names:
                all_names.remove(name)
                dropped_names.append(name)

        if len(all_names) > MAX_HYPERPARAMETERS:
            kept_names = all_names[:MAX_HYPERPARAMETERS]
            overflow_names = all_names[MAX_HYPERPARAMETERS:]
            dropped_names += overflow_names
            logger.warning(
                f"FT-PFN surrogate supports at most {MAX_HYPERPARAMETERS} "
                f"hyperparameters; after priority-dropping {drop_priority}, "
                f"{len(all_names)} remain. Keeping the first "
                f"{MAX_HYPERPARAMETERS} (in `ConfigurationSpace.get_hyperparameters()` "
                f"order, which is alphabetical - not by importance): {kept_names}. "
                f"Dropping: {overflow_names}. Add these to `drop_priority` above "
                f"instead of relying on this fallback.",
                stacklevel=2,
            )
        else:
            kept_names = all_names

        self.names = kept_names
        self.specs = {name: specs[name] for name in kept_names}
        # Kept around for introspection/debugging, not used by encode().
        self.dropped_names = dropped_names

    @property
    def dim(self) -> int:
        return len(self.names)

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        """
        Encode a ConfigSpace Configuration (as dict) into a vector in [0,1]^d.

        Inactive hyperparameters (value is None or missing) are mapped to 0.5.
        Only ``self.names`` (at most ``MAX_HYPERPARAMETERS``, chosen by
        priority order in __init__) are ever encoded; any keys in ``config``
        that aren't in ``self.names`` (including dropped hyperparameters)
        are silently ignored.
        """
        assert len(self.names) <= MAX_HYPERPARAMETERS, (
            f"HyperparameterSpace.names exceeds {MAX_HYPERPARAMETERS} dims; "
            "this should be impossible after __init__ truncation."
        )
        values: list[float] = []
        for name in self.names:
            val = config.get(name, None)
            if val is None:
                values.append(0.5)
            else:
                values.append(self.specs[name].encode(val))
        return torch.tensor(values, dtype=torch.float32)
