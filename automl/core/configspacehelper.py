from __future__ import annotations

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
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
    batch_size = Categorical("batch_size", [32, 64, 128, 256, 512], default=64)

    warmup_ratio = Float("warmup_ratio", (0.0, 0.2), default=0.1)

    hyperparams: list[Hyperparameter] = [
        model_type,
        weight_decay,
        batch_size,
        warmup_ratio,
        dropout,
    ]

    conditions: list = []

    if fixed_model_type == "sequence-dl":
        hidden_dim = Categorical("hidden_dim", [32, 64, 128, 256], default=128)
        learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
        seq_embed_dim = Categorical("seq_embed_dim", [32, 48, 64, 128, 256, 512], default=128)
        seq_num_layers = Integer("seq_num_layers", (1, 5), default=1)

        hyperparams += [
            hidden_dim,
            learning_rate,
            seq_embed_dim,
            seq_num_layers,
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

    elif fixed_model_type == "tfidf-ffnn":
        hidden_dim = Integer("hidden_dim", (32, 256), log=True, default=128)
        learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
        optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adamw")

        # optimizer-specific params, only active for the optimizers that use them
        beta1 = Float("beta1", (0.8, 0.999), default=0.9)
        beta2 = Float("beta2", (0.9, 0.9999), default=0.999)
        momentum = Float("momentum", (0.0, 0.99), default=0.9)

        # --- TF-IDF vectorization params ---
        vocab_size = Integer("vocab_size", (1_000, 50_000), log=True, default=10_000)
        ngram_max = Integer("ngram_max", (1, 3), default=1)
        analyzer = Categorical("analyzer", ["word", "char", "char_wb"], default="word")
        stop_words = Categorical("stop_words", [None, "english"], default=None)
        min_df = Integer("min_df", (1, 10), default=1)
        max_df = Float("max_df", (0.5, 1.0), default=1.0)
        sublinear_tf = Categorical("sublinear_tf", [True, False], default=False)
        use_idf = Categorical("use_idf", [True, False], default=True)
        norm = Categorical("norm", [None, "l1", "l2"], default="l2")
        representation = Categorical(
            "representation", ["word", "char", "hybrid"], default="word"
        )
        # ranges chosen so char_ngram_min <= char_ngram_max always holds
        char_ngram_min = Integer("char_ngram_min", (2, 3), default=2)
        char_ngram_max = Integer("char_ngram_max", (3, 6), default=5)
        class_balance = Categorical("class_balance", [True, False], default=False)

        hyperparams += [
            hidden_dim,
            learning_rate,
            optimizer,
            beta1,
            beta2,
            momentum,
            vocab_size,
            ngram_max,
            analyzer,
            stop_words,
            min_df,
            max_df,
            sublinear_tf,
            use_idf,
            norm,
            representation,
            char_ngram_min,
            char_ngram_max,
            class_balance,
        ]

        conditions += [
            InCondition(beta1, optimizer, ["adam", "adamw"]),
            InCondition(beta2, optimizer, ["adam", "adamw"]),
            EqualsCondition(momentum, optimizer, "sgd"),
        ]

    elif fixed_model_type == "simple":
        hidden_dim = Integer("simple_hidden_dim", (16, 256), log=True, default=64)
        learning_rate = Float("learning_rate", (1e-4, 1e-2), default=1e-3, log=True)
        optimizer = Categorical("optimizer", ["adam", "adamw", "sgd"], default="adamw")
        simple_embed_dim = Integer("simple_embed_dim", (32, 300), log=True, default=100)
        simple_num_layers = Integer("simple_num_layers", (1, 3), default=1)
        simple_pretrained_model_name = Categorical(
            "simple_pretrained_model_name",
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
            simple_embed_dim,
            simple_num_layers,
            simple_pretrained_model_name,
        ]

    else:
        raise ValueError(
            f"Unknown fixed_model_type for config space: {fixed_model_type!r}"
        )

    cs.add(hyperparams)

    if conditions:
        cs.add(conditions)

    return cs