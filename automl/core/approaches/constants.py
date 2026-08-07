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
    "class_balance": False,
}


TRANSFORMER_DEFAULT_CONFIG = {
    "transformer_model_name": "distilbert-base-uncased",
    "max_seq_length": 128,
    "dropout": 0.1,
    "freeze_ratio": 0.0,
    "batch_size": 32,
    "epochs": 5,
    "optimizer": "adamw",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
}


TFIDF_FFNN_DEFAULT_CONFIG = {
    "max_grad_norm": 1.0,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "beta1": 0.9,
    "beta2": 0.999,
    "momentum": 0.9,
    "vocab_size": 10_000,
    "ngram_max": 1,
    "analyzer": "word",
    "stop_words": None,
    "min_df": 1,
    "max_df": 1.0,
    "sublinear_tf": False,
    "use_idf": True,
    "norm": "l2",
    "representation": "word",
    "char_ngram_min": 2,
    "char_ngram_max": 5,
    "class_balance": False,
}


SIMPLE_DEFAULT_CONFIG = {
    "max_seq_length": 128,
    "simple_pretrained_model_name": "distilbert-base-uncased",
    "simple_embed_dim": 100,
    "simple_hidden_dim": 64,
    "simple_num_layers": 1,
    "dropout": 0.2,
    "batch_size": 64,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
}


DEFAULTS: dict[ApproachName | Literal["common"], dict[str, Any]] = {
    "common": COMMON_CONFIG,
    "sequence-dl": SEQUENCE_DL_DEFAULT_CONFIG,
    "transformer": TRANSFORMER_DEFAULT_CONFIG,
    "tfidf-ffnn": TFIDF_FFNN_DEFAULT_CONFIG,
    "simple": SIMPLE_DEFAULT_CONFIG,
}


def get_approach_defaults(approach_name: ApproachName) -> dict[str, Any]:
    defaults = DEFAULTS.get("common", {})
    if approach_name != "common":
        defaults = {**defaults, **DEFAULTS.get(approach_name, {})}
    return defaults


__all__ = ["get_approach_defaults"]