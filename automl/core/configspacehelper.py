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
)
from ConfigSpace.conditions import Condition
from ConfigSpace.hyperparameters import Hyperparameter

from automl.core import registry
from automl.logger import get_logger

logger = get_logger()


def build_config_space(
    seed: int = 42,
    fixed_model_type: Optional[str] = None,
) -> ConfigurationSpace:
    # TODO: Remove this
    cs = ConfigurationSpace(seed=seed)

    # --- top-level branch: representation + model family are coupled ---
    all_model_types = list(registry.register_all_approaches())
    if not all_model_types:
        logger.warning("No valid model types found.")

    model_type: Hyperparameter
    if fixed_model_type is None:
        logger.info("No fixed model type specified.")
        model_type = Categorical("model_type", all_model_types, default="tfidf-ffnn")
        allowed_model_types = set(all_model_types)
    else:
        logger.info(f"Fixed model type specified: {fixed_model_type}")
        if fixed_model_type not in all_model_types:
            logger.error(f"Unknown model_type: {fixed_model_type}")
            raise ValueError(f"Unknown model_type: {fixed_model_type}")
        model_type = Categorical(
            "model_type", [fixed_model_type], default=fixed_model_type
        )
        allowed_model_types = {fixed_model_type}

    # This is only really used by the TF-IDF approaches
    representation = Categorical(
        "representation", ["word", "char", "hybrid"], default="word"
    )

    # --- representation hyperparameters (for TF-IDF models) ---
    vocab_size = Integer("vocab_size", (1_000, 50_000), default=10_000, log=True)
    ngram_max = Integer("ngram_max", (1, 3), default=1)  # ngram_range = (1, ngram_max)
    char_ngram_min = Integer("char_ngram_min", (2, 4), default=3)
    char_ngram_max = Integer("char_ngram_max", (4, 6), default=5)
    min_df = Float("min_df", (0.0, 0.05), default=0.02)
    max_df = Float("max_df", (0.7, 1.0), default=0.8)
    sublinear_tf = Categorical("sublinear_tf", [True, False], default=True)

    # --- NN ---
    hidden_dim = Integer("hidden_dim", (32, 512), default=128, log=True)
    dropout = Float("dropout", (0.0, 0.5), default=0.1)
    learning_rate = Float("learning_rate", (1e-5, 1e-3), default=1e-3, log=True)
    optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adam")
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="steplr",
    )
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-5, log=True)
    batch_size = Integer("batch_size", (32, 256), default=64, log=True)

    # --- linear-only hyperparameters ---
    alpha = Float("alpha", (1e-6, 1e-1), default=1e-4, log=True)
    max_seq_length = Integer("max_seq_length", (64, 256), default=128)

    # --- transformer-only hyperparameters ---
    transformer_model_name = Categorical(
        "transformer_model_name",
        ["distilbert-base-uncased", "bert-base-uncased"],
        default="distilbert-base-uncased",
    )
    transformer_learning_rate = Float(
        "transformer_learning_rate",
        (1e-6, 5e-5),
        default=2e-5,
        log=True,
    )
    transformer_batch_size = Categorical(
        "transformer_batch_size",
        [8, 16, 32],
        default=16,
    )
    freeze_transformer = Categorical(
        "freeze_transformer",
        [True, False],
        default=False,
    )
    warmup_ratio = Float("warmup_ratio", (0.0, 0.2), default=0.1)

    # --- sequence-dl (LSTM/GRU/CNN) hyperparameters ---
    # seq_arch = Categorical("seq_arch", ["bilstm", "cnn"], default="bilstm")
    seq_embed_dim = Integer("seq_embed_dim", (32, 512), default=128, log=True)
    seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)

    # CNN-specific
    # seq_num_filters = Integer("seq_num_filters", (50, 300), default=100, log=True)
    # kernel_possibilities = ["3", "4", "5"]
    # combinations = [
    #     ",".join(combo)
    #     for i in range(1, len(kernel_possibilities) + 1)
    #     for combo in itertools.combinations(kernel_possibilities, i)
    # ]
    # seq_kernel_pattern = Categorical(
    #     "seq_kernel_pattern", combinations, default=",".join(kernel_possibilities)
    # )

    # --- BPE-RNN (byte-level BPE + RNN) hyperparameters ---
    bpe_vocab_size = Integer(
        "bpe_vocab_size",
        (2_000, 50_000),
        default=8_000,
        log=True,
    )
    token_length = Integer(
        "token_length",
        (64, 512),
        default=256,
    )
    rnn_type = Categorical(
        "rnn_type",
        ["lstm", "gru"],
        default="lstm",
    )
    emb_dim = Integer(
        "emb_dim",
        (32, 512),
        default=128,
        log=True,
    )
    num_layers = Integer(
        "num_layers",
        (1, 3),
        default=1,
    )
    bidirectional = Categorical(
        "bidirectional",
        [True, False],
        default=False,
    )

    cs.add(
        [
            model_type,
            max_seq_length,  # sequence-dl and transformer only
            weight_decay,  # sequence-dl and transformer only
            batch_size,
        ]
    )

    conditions: list[Condition] = []

    if any(
        [_allowed_model.startswith("tfidf") for _allowed_model in allowed_model_types]
    ):
        cs.add(
            [
                representation,
                vocab_size,
                ngram_max,
                char_ngram_min,
                char_ngram_max,
                min_df,
                max_df,
                sublinear_tf,
                alpha,
            ]
        )

        conditions += [
            # Representation-specific for TF-IDF models
            # InCondition(representation, model_type, ["tfidf-ffnn", "tfidf-linear"]),
            InCondition(vocab_size, model_type, ["tfidf-ffnn", "tfidf-linear"]),
            InCondition(ngram_max, representation, ["word", "hybrid"]),
            InCondition(char_ngram_min, representation, ["char", "hybrid"]),
            InCondition(char_ngram_max, representation, ["char", "hybrid"]),
        ]

        # Linear branch
        if "tfidf-linear" in allowed_model_types:
            conditions.append(EqualsCondition(alpha, model_type, "tfidf-linear"))

        cs.add(
            [
                ForbiddenLessThanRelation(char_ngram_max, char_ngram_min),
                ForbiddenLessThanRelation(max_df, min_df),
            ]
        )

    if "transformer" in allowed_model_types:
        cs.add(
            [
                transformer_model_name,
                transformer_learning_rate,
                transformer_batch_size,
                freeze_transformer,
                warmup_ratio,
            ]
        )

    if "sequence-dl" in allowed_model_types:
        cs.add(
            [
                seq_embed_dim,
                seq_num_layers,
                # seq_arch,
                # seq_num_filters,
                # seq_kernel_pattern,
            ]
        )

    if "bpe-rnn" in allowed_model_types:
        cs.add(
            [
                # BPE-RNN specific
                bpe_vocab_size,
                token_length,
                rnn_type,
                emb_dim,
                num_layers,
                bidirectional,
            ]
        )

    # FFNN + BPE-RNN branch (shared hyperparameters)
    ffnn_like_models = [
        m for m in ["tfidf-ffnn", "bpe-rnn", "sequence-dl"] if m in allowed_model_types
    ]
    if ffnn_like_models:
        cs.add(
            [
                hidden_dim,
                dropout,
                learning_rate,
                optimizer,
                scheduler,
            ]
        )
        conditions.extend(
            [
                InCondition(max_seq_length, model_type, ffnn_like_models),
                InCondition(hidden_dim, model_type, ffnn_like_models),
                InCondition(dropout, model_type, ffnn_like_models),
                InCondition(learning_rate, model_type, ffnn_like_models),
                InCondition(optimizer, model_type, ffnn_like_models),
                InCondition(weight_decay, model_type, ffnn_like_models),
                InCondition(batch_size, model_type, ffnn_like_models),
                InCondition(scheduler, model_type, ffnn_like_models),
            ]
        )

    # Transformer branch
    if "transformer" in allowed_model_types:
        conditions.extend(
            [
                EqualsCondition(transformer_model_name, model_type, "transformer"),
                EqualsCondition(max_seq_length, model_type, "transformer"),
                EqualsCondition(transformer_learning_rate, model_type, "transformer"),
                EqualsCondition(transformer_batch_size, model_type, "transformer"),
                EqualsCondition(freeze_transformer, model_type, "transformer"),
                EqualsCondition(warmup_ratio, model_type, "transformer"),
            ]
        )

    # Sequence-dl branch
    if "sequence-dl" in allowed_model_types:
        conditions.extend(
            [
                EqualsCondition(seq_embed_dim, model_type, "sequence-dl"),
                EqualsCondition(seq_num_layers, model_type, "sequence-dl"),
                # EqualsCondition(seq_arch, model_type, "sequence-dl"),
                # EqualsCondition(seq_arch, model_type, "sequence-dl"),
                # EqualsCondition(seq_num_filters, seq_arch, "cnn"),
                # EqualsCondition(seq_kernel_pattern, seq_arch, "cnn"),
            ]
        )

    # BPE-RNN branch
    if "bpe-rnn" in allowed_model_types:
        conditions.extend(
            [
                EqualsCondition(bpe_vocab_size, model_type, "bpe-rnn"),
                EqualsCondition(token_length, model_type, "bpe-rnn"),
                EqualsCondition(rnn_type, model_type, "bpe-rnn"),
                EqualsCondition(emb_dim, model_type, "bpe-rnn"),
                EqualsCondition(num_layers, model_type, "bpe-rnn"),
                EqualsCondition(bidirectional, model_type, "bpe-rnn"),
            ]
        )

    cs.add(conditions)

    return cs
