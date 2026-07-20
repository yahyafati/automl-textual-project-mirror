from __future__ import annotations

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

from automl.core import registry
from automl.logger import get_logger

logger = get_logger()


def build_sequence_dl_config_space(seed: int = 42) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)

    # Architecture: BiLSTM or CNN
    seq_arch = Categorical("seq_arch", ["bilstm", "cnn"], default="bilstm")

    # Shared sequence-DL hyperparameters
    seq_embed_dim = Integer("seq_embed_dim", (32, 512), default=128, log=True)
    seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)
    hidden_dim = Integer("hidden_dim", (32, 512), default=128, log=True)
    dropout = Float("dropout", (0.0, 0.5), default=0.1)

    # Data-related
    vocab_size = Integer("vocab_size", (1_000, 50_000), default=10_000, log=True)
    max_seq_length = Integer("max_seq_length", (64, 256), default=128)

    # CNN-specific: filters and kernel-size pattern
    seq_num_filters = Integer("seq_num_filters", (50, 300), default=100, log=True)

    # Single parameter for kernel size combinations
    # Each choice encodes which kernel sizes (3, 4, 5) are active.
    # You can map this back to booleans when building the model.
    seq_kernel_pattern = Categorical(
        "seq_kernel_pattern",
        [
            "3",
            "4",
            "5",
            "3,4",
            "3,5",
            "4,5",
            "3,4,5",
        ],
        default="3,4,5",
    )

    # Optimizer / training hyperparameters
    learning_rate = Float("learning_rate", (1e-4, 1e-1), default=1e-3, log=True)
    optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adam")
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-5, log=True)
    batch_size = Categorical("batch_size", [32, 64, 128, 256], default=64)
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="steplr",
    )
    scheduler_step_size = Integer("scheduler_step_size", (1, 10), default=5)
    scheduler_gamma = Float("scheduler_gamma", (0.1, 0.9), default=0.1)

    cs.add(
        [
            # seq_arch,
            # seq_embed_dim,
            # seq_num_layers,
            hidden_dim,
            dropout,
            vocab_size,
            # max_seq_length,
            seq_num_filters,
            seq_kernel_pattern,
            learning_rate,
            optimizer,
            weight_decay,
            batch_size,
            # scheduler,
            # scheduler_step_size,
            # scheduler_gamma,
        ]
    )

    return cs


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
    # ["tfidf-ffnn", "tfidf-linear", "transformer"]

    if fixed_model_type is None:
        logger.info("No fixed model type specified.")
        model_type = Categorical(
            "model_type",
            all_model_types,
            default="tfidf-ffnn",
        )
        allowed_model_types = set(all_model_types)
    else:
        logger.info(f"Fixed model type specified: {fixed_model_type}")
        if fixed_model_type not in all_model_types:
            logger.error(f"Unknown model_type: {fixed_model_type}")
            raise ValueError(f"Unknown model_type: {fixed_model_type}")
        model_type = Categorical(
            "model_type",
            [fixed_model_type],
            default=fixed_model_type,
        )
        allowed_model_types = {fixed_model_type}

    # This is only really used by the TF-IDF approaches
    representation = Categorical(
        "representation",
        ["word", "char", "hybrid"],
        default="word",
    )

    # --- representation hyperparameters (for TF-IDF models) ---
    vocab_size = Integer("vocab_size", (1_000, 50_000), default=10_000, log=True)
    ngram_max = Integer("ngram_max", (1, 3), default=1)  # ngram_range = (1, ngram_max)
    char_ngram_min = Integer("char_ngram_min", (2, 4), default=3)
    char_ngram_max = Integer("char_ngram_max", (4, 6), default=5)
    min_df = Float("min_df", (0.0, 0.05), default=0.02)
    max_df = Float("max_df", (0.7, 1.0), default=0.8)
    sublinear_tf = Categorical("sublinear_tf", [True, False], default=True)

    # --- FFNN-only + BPE-RNN hyperparameters ---
    hidden_dim = Integer("hidden_dim", (32, 512), default=128, log=True)
    dropout = Float("dropout", (0.0, 0.5), default=0.1)
    learning_rate = Float("learning_rate", (1e-4, 1e-1), default=1e-3, log=True)
    optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adam")
    scheduler = Categorical(
        "scheduler",
        ["steplr", "cosineannealinglr", "exponentiallr", "reducelronplateau"],
        default="steplr",
    )
    scheduler_step_size = Integer("scheduler_step_size", (1, 10), default=5)
    scheduler_gamma = Float("scheduler_gamma", (0.1, 0.9), default=0.1)
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-5, log=True)
    batch_size = Categorical("batch_size", [32, 64, 128, 256], default=64)

    # Optimizer-specific hyperparameters
    momentum = Float("momentum", bounds=(0.0, 0.99), default=0.9)
    beta1 = Float("beta1", bounds=(0.8, 0.999), default=0.9)
    beta2 = Float("beta2", bounds=(0.9, 0.9999), default=0.999)

    # --- linear-only hyperparameters ---
    alpha = Float("alpha", (1e-6, 1e-1), default=1e-4, log=True)

    # --- transformer-only hyperparameters ---
    transformer_model_name = Categorical(
        "transformer_model_name",
        [
            "distilbert-base-uncased",
            "bert-base-uncased",
        ],
        default="distilbert-base-uncased",
    )
    max_seq_length = Integer("max_seq_length", (64, 256), default=128)
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
    seq_arch = Categorical("seq_arch", ["bilstm", "cnn"], default="bilstm")
    seq_embed_dim = Integer("seq_embed_dim", (32, 512), default=128, log=True)
    seq_num_layers = Integer("seq_num_layers", (1, 3), default=1)

    # CNN-specific
    seq_num_filters = Integer("seq_num_filters", (50, 300), default=100, log=True)
    seq_kernel_size_3 = Categorical("seq_kernel_size_3", [True, False], default=True)
    seq_kernel_size_4 = Categorical("seq_kernel_size_4", [True, False], default=True)
    seq_kernel_size_5 = Categorical("seq_kernel_size_5", [True, False], default=True)

    # --- cross-cutting ---
    class_balance = Categorical("class_balance", [True, False], default=False)
    seed_hp = Integer("seed", (0, 2**16), default=42)

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
            representation,
            vocab_size,
            ngram_max,
            char_ngram_min,
            char_ngram_max,
            min_df,
            max_df,
            sublinear_tf,
            hidden_dim,
            dropout,
            learning_rate,
            optimizer,
            max_seq_length,
            scheduler,
            scheduler_step_size,
            scheduler_gamma,
            weight_decay,
            batch_size,
            alpha,
            class_balance,
            seed_hp,
            momentum,
            beta1,
            beta2,
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
                seq_arch,
                seq_num_filters,
                seq_kernel_size_3,
                seq_kernel_size_4,
                seq_kernel_size_5,
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

    # --- conditionals: only sample/apply a hyperparameter when it's relevant ---
    conditions = [
        # Representation-specific for TF-IDF models
        # InCondition(representation, model_type, ["tfidf-ffnn", "tfidf-linear"]),
        InCondition(ngram_max, representation, ["word", "hybrid"]),
        InCondition(char_ngram_min, representation, ["char", "hybrid"]),
        InCondition(char_ngram_max, representation, ["char", "hybrid"]),
        # Optimizer-specific hyperparameters (always valid; parent is 'optimizer')
        EqualsCondition(momentum, optimizer, "sgd"),
        InCondition(beta1, optimizer, ["adam", "adamw"]),
        InCondition(beta2, optimizer, ["adam", "adamw"]),
    ]

    # FFNN + BPE-RNN branch (shared hyperparameters)
    ffnn_like_models = [
        m for m in ["tfidf-ffnn", "bpe-rnn", "sequence-dl"] if m in allowed_model_types
    ]
    if ffnn_like_models:
        conditions.extend(
            [
                InCondition(vocab_size, model_type, ffnn_like_models),
                InCondition(max_seq_length, model_type, ffnn_like_models),
                InCondition(hidden_dim, model_type, ffnn_like_models),
                InCondition(dropout, model_type, ffnn_like_models),
                InCondition(learning_rate, model_type, ffnn_like_models),
                InCondition(optimizer, model_type, ffnn_like_models),
                InCondition(scheduler, model_type, ffnn_like_models),
                InCondition(scheduler_step_size, scheduler, ["steplr"]),
                InCondition(scheduler_gamma, scheduler, ["exponentiallr"]),
                InCondition(weight_decay, model_type, ffnn_like_models),
                InCondition(batch_size, model_type, ffnn_like_models),
            ]
        )

    # Linear branch
    if "tfidf-linear" in allowed_model_types:
        conditions.append(EqualsCondition(alpha, model_type, "tfidf-linear"))

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
                EqualsCondition(seq_arch, model_type, "sequence-dl"),
                EqualsCondition(seq_arch, model_type, "sequence-dl"),
                EqualsCondition(seq_num_filters, seq_arch, "cnn"),
                EqualsCondition(seq_kernel_size_3, seq_arch, "cnn"),
                EqualsCondition(seq_kernel_size_4, seq_arch, "cnn"),
                EqualsCondition(seq_kernel_size_5, seq_arch, "cnn"),
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

    # --- forbidden: ---
    cs.add(
        [
            ForbiddenLessThanRelation(char_ngram_max, char_ngram_min),
            ForbiddenLessThanRelation(max_df, min_df),
        ]
    )
    return cs
