"""
ifBO (In-Context Freeze-Thaw Bayesian Optimization) optimizer.

This package preserves the public API of the original single-file
``ifbo_opt.py`` module, so existing imports such as

    from src.automl.optimizers.ifbo_opt import IfboOptimizer

continue to work unchanged.
"""

from .candidate import IfBOCandidate
from .hp_space import Categorical, Float, HPSpec, HyperparameterSpace, Integer
from .optimizer import IfboOptimizer

__all__ = [
    "IfboOptimizer",
    "HyperparameterSpace",
    "HPSpec",
    "Float",
    "Integer",
    "Categorical",
    "IfBOCandidate",
]
