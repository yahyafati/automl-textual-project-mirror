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
    "seq_arch": "bilstm",
    "max_vocab_size": 20000,
    "max_seq_length": 128,
    "seq_embed_dim": 128,
    "hidden_dim": 128,
    "seq_num_layers": 1,
    "dropout": 0.5,
    "batch_size": 64,
    "epochs": 5,
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "seq_num_filters": 1,
    "seq_kernel_pattern": "3",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
}


DEFAULTS: dict[ApproachName | Literal["common"], dict[str, Any]] = {
    "common": COMMON_CONFIG,
    "sequence-dl": SEQUENCE_DL_DEFAULT_CONFIG,
}


def get_approach_defaults(approach_name: ApproachName) -> dict[str, Any]:
    defaults = DEFAULTS.get("common", {})
    if approach_name != "common":
        defaults = {**defaults, **DEFAULTS.get(approach_name, {})}
    return defaults


__all__ = ["get_approach_defaults"]
