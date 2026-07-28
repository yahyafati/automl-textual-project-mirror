from __future__ import annotations

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
)
from ConfigSpace.hyperparameters import Hyperparameter

from automl.logger import get_logger

logger = get_logger()


def build_config_space(
    seed: int = 42, fixed_model_type: str = "sequence-dl"
) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)

    model_type = Constant("model_type", fixed_model_type)

    # --- hyperparameters shared by every approach (consumed generically by
    dropout = Float("dropout", (0.0, 0.5), default=0.2)
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-4, log=True)
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="cosineannealinglr",
    )
    batch_size = Integer("batch_size", (32, 512), log=True, default=64)

    max_seq_length = Integer("max_seq_length", (64, 256), log=True, default=128)

    warmup_ratio = Float("warmup_ratio", (0.0, 0.2), default=0.1)

    hyperparams: list[Hyperparameter] = [
        model_type,
        max_seq_length,
        weight_decay,
        batch_size,
        warmup_ratio,
        dropout,
        scheduler,
    ]

    if fixed_model_type == "sequence-dl":
        hidden_dim = Integer("hidden_dim", (32, 256), log=True, default=128)
        learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
        optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adamw")
        seq_embed_dim = Integer("seq_embed_dim", (32, 512), log=True, default=128)
        seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)
        seq_pretrained_model_name = Categorical(
            "seq_pretrained_model_name",
            [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "google/bert_uncased_L-4_H-512_A-8",
                "microsoft/xtremedistil-l6-h256-uncased",
            ],
            default="distilbert-base-uncased",
        )

        hyperparams += [
            hidden_dim,
            learning_rate,
            optimizer,
            seq_embed_dim,
            seq_num_layers,
            seq_pretrained_model_name,
        ]

    elif fixed_model_type == "transformer":
        learning_rate = Float("learning_rate", (1e-5, 5e-5), default=2e-5, log=True)
        optimizer = Categorical("optimizer", ["adamw", "adam"], default="adamw")
        transformer_model_name = Categorical(
            "transformer_model_name",
            [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "google/bert_uncased_L-4_H-512_A-8",
                "microsoft/xtremedistil-l6-h256-uncased",
            ],
            default="google/bert_uncased_L-4_H-512_A-8",
        )
        freeze_ratio = Float("freeze_ratio", (0.75, 1.0), default=0.75)

        hyperparams += [
            learning_rate,
            optimizer,
            transformer_model_name,
            freeze_ratio,
        ]

    else:
        raise ValueError(
            f"Unknown fixed_model_type for config space: {fixed_model_type!r}"
        )

    cs.add(hyperparams)

    return cs
