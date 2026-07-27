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
    cs = ConfigurationSpace(seed=seed)

    model_type = Constant("model_type", fixed_model_type)

    # --- hyperparameters shared by every approach (consumed generically by
    # TorchTrainer / Approach.get_param_value, regardless of architecture) ---
    dropout = Float("dropout", (0.0, 0.5), default=0.2)
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-4, log=True)
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="cosineannealinglr",
    )
    batch_size = Categorical("batch_size", [32, 64, 128, 256], default=64)

    # TODO: Check this out
    #  If you want to enforce exact powers of 2 (which align best with CUDA memory management and PyTorch tensor cores),
    #      you can also define it as a Categorical hyperparameter:
    # Capped at 256 (was up to 1024): packed-sequence LSTM cost scales
    # ~linearly with token count, so trials sampling 1024 cost 4-8x a trial
    # sampling 128 for little accuracy gain on star-rating classification,
    # where most signal is in the first ~256 tokens of a review.
    max_seq_length = Categorical("max_seq_length", [64, 128, 256], default=128)

    warmup_ratio = Float("warmup_ratio", (0.0, 0.2), default=0.1)

    hyperparams: list[Hyperparameter] = [
        model_type,
        max_seq_length,  # sequence-dl and transformer
        weight_decay,  # sequence-dl and transformer
        batch_size,
        warmup_ratio,
        dropout,
        scheduler,
    ]

    if fixed_model_type == "sequence-dl":
        # --- sequence-dl (BiLSTM, trained from scratch) hyperparameters ---
        hidden_dim = Categorical("hidden_dim", [32, 64, 128, 256], default=128)
        learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
        optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adamw")
        seq_embed_dim = Categorical(
            "seq_embed_dim", [32, 64, 128, 256, 512], default=128
        )
        seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)
        # Tokenizer + pretrained-embedding source used to warm-start the
        # BiLSTM's embedding layer (see `_pretrained_embedding_init` in
        # approaches/sequence_dl.py). Tokenizer and embedding always come
        # from the same model, so this single choice drives both.
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
        # --- transformer (fine-tuned pretrained encoder) hyperparameters ---
        # Fine-tuning needs a much smaller LR than training the BiLSTM from
        # scratch: sequence-dl's 1e-4 to 1e-2 range would wreck the
        # pretrained weights within a handful of steps, so this uses the
        # standard BERT-family fine-tuning range instead.
        learning_rate = Float("learning_rate", (1e-5, 5e-5), default=2e-5, log=True)
        optimizer = Categorical("optimizer", ["adamw", "adam"], default="adamw")
        transformer_model_name = Categorical(
            "transformer_model_name",
            [
                # "distilbert-base-uncased",
                # "bert-base-uncased",
                "google/bert_uncased_L-4_H-512_A-8",
                "microsoft/xtremedistil-l6-h256-uncased",
            ],
            default="google/bert_uncased_L-4_H-512_A-8",
        )
        # Fraction of the pretrained base's layers to freeze, ordered
        # bottom-up (embeddings first): 0.0 fine-tunes the whole base
        # (default), 1.0 is linear-probe mode (only the classification head
        # trains), and values in between gradually unfreeze the top layers
        # while keeping the more generic lower layers fixed. See
        # `TransformerClassifier`/`_freeze_base_by_ratio` in
        # `approaches/transformer.py`.
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

    conditions: list[Condition] = []

    cs.add(conditions)

    return cs
