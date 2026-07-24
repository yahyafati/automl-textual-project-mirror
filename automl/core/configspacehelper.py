from __future__ import annotations

import itertools
from typing import Optional

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
    InCondition,
    ForbiddenLessThanRelation,
    Constant,
)
from ConfigSpace.conditions import Condition
from ConfigSpace.hyperparameters import Hyperparameter

from automl.core import registry
from automl.logger import get_logger

logger = get_logger()


def build_config_space(
    seed: int = 42, fixed_model_type: str = "sequence-dl"
) -> ConfigurationSpace:
    # TODO: Remove this
    cs = ConfigurationSpace(seed=seed)

    model_type = Constant("model_type", "sequence-dl")
    # --- NN ---
    hidden_dim = Categorical("hidden_dim", [32, 64, 128, 256], default=128)
    dropout = Float("dropout", (0.0, 0.5), default=0.1)
    learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
    optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adam")
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="steplr",
    )
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-4, log=True)
    batch_size = Categorical("batch_size", [32, 64, 128, 256], default=64)

    # TODO: Check this out
    #  If you want to enforce exact powers of 2 (which align best with CUDA memory management and PyTorch tensor cores),
    #      you can also define it as a Categorical hyperparameter:
    max_seq_length = Categorical("max_seq_length", [128, 256, 512, 1024], default=256)

    warmup_ratio = Float("warmup_ratio", (0.0, 0.2), default=0.1)

    # --- sequence-dl (LSTM/GRU/CNN) hyperparameters ---
    seq_embed_dim = Integer("seq_embed_dim", (32, 512), default=128, log=True)
    seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)

    cs.add(
        [
            model_type,
            max_seq_length,  # sequence-dl and transformer only
            weight_decay,  # sequence-dl and transformer only
            batch_size,
            warmup_ratio,
            seq_embed_dim,
            seq_num_layers,
            hidden_dim,
            dropout,
            learning_rate,
            optimizer,
            scheduler,
        ]
    )

    conditions: list[Condition] = []

    cs.add(conditions)

    return cs
