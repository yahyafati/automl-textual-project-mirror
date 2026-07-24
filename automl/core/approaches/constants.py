from typing import Any, Literal

from automl.core.types import ApproachName

COMMON_CONFIG: dict[str, Any] = {
    # --- reproducibility ---
    "seed": 42,
    "scheduler_step_size": None,
    "scheduler": None,
    "scheduler_gamma": None,
}

SEQUENCE_DL_DEFAULT_CONFIG = {
    "max_seq_length": 128,
    "seq_embed_dim": 128,
    "hidden_dim": 128,
    "seq_num_layers": 1,
    "dropout": 0.5,
    "batch_size": 64,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
}


TRANSFORMER_DEFAULT_CONFIG = {
    "transformer_model_name": "distilbert-base-uncased",
    "max_seq_length": 128,
    "dropout": 0.1,
    "freeze_base": False,
    "batch_size": 32,
    "epochs": 5,
    "optimizer": "adamw",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
}


DEFAULTS: dict[ApproachName | Literal["common"], dict[str, Any]] = {
    "common": COMMON_CONFIG,
    "sequence-dl": SEQUENCE_DL_DEFAULT_CONFIG,
    "transformer": TRANSFORMER_DEFAULT_CONFIG,
}


def get_approach_defaults(approach_name: ApproachName) -> dict[str, Any]:
    defaults = DEFAULTS.get("common", {})
    if approach_name != "common":
        defaults = {**defaults, **DEFAULTS.get(approach_name, {})}
    return defaults


__all__ = ["get_approach_defaults"]
