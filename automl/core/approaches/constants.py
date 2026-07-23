from typing import Any, Literal

from automl.core.types import ApproachName

COMMON_CONFIG: dict[str, Any] = {
    # --- reproducibility ---
    "seed": 42,
    "scheduler_step_size": None,
    "scheduler": None,
    "scheduler_gamma": None,
}

DEFAULT_TFIDF_FFNN_CONFIG: dict[str, Any] = {
    # --- vectorizer: word-level ---
    "vocab_size": 10_000,
    "ngram_max": 1,
    "analyzer": "word",  # "word" | "char" | "char_wb"
    "stop_words": None,  # None | "english"
    "min_df": 0.02,  # float (fraction) or int
    "max_df": 0.8,  # float (fraction) or int
    "sublinear_tf": True,
    "use_idf": True,
    "norm": "l2",  # "l1" | "l2" | None
    # --- optional char n-grams ---
    "use_char_ngrams": False,
    "char_ngram_min": 3,  # IMPORTANT: should be int, not float
    "char_ngram_max": 5,  # IMPORTANT: should be int, not float
    # --- model architecture ---
    "hidden_dim": 128,
    "dropout": 0.0,
    # --- training ---
    "batch_size": 64,
    "class_balance": False,
    "epochs": 50,
    # --- optimizer ---
    "optimizer": "adam",  # "adam" | "adamw" | "sgd"
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    # --- optimizer: adam/adamw ---
    "beta1": 0.9,
    "beta2": 0.999,
    # --- optimizer: sgd ---
    "momentum": 0.0,
}

TFIDF_LINEAR_DEFAULT_CONFIG: dict[str, Any] = {
    # --- reproducibility ---
    "seed": 42,
    # --- TF-IDF ---
    "vocab_size": 20_000,
    "ngram_max": 2,
    "min_df": 0.02,
    "max_df": 0.8,
    "sublinear_tf": True,
    # --- SGDClassifier ---
    "alpha": 1e-4,
    "loss": "log_loss",
    # --- fidelity / training budget ---
    # NOTE: reused as proxy for SGD max_iter scaling
    "epochs": 20,
    "max_iter_multiplier": 40,
    # --- class handling ---
    "class_balance": False,
}

TRANSFORMER_DEFAULT_CONFIG: dict[str, Any] = {
    # --- pretrained model & tokenizer ---
    "transformer_model_name": "distilbert-base-uncased",
    "max_seq_length": 128,
    # --- training ---
    "transformer_batch_size": 16,
    "epochs": 5,  # only used when not overridden by the optimizer
    "class_balance": False,
    # --- optimizer / scheduler ---
    "transformer_learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "freeze_transformer": False,
    "max_grad_norm": 1.0,
    # --- dataloader ---
    "num_workers": 4,
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

DEFAULT_BPE_LSTM = {
    "bpe_vocab_size": 8000,
    "token_length": 256,
    "rnn_type": "lstm",
    "emb_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "bidirectional": False,
    "dropout": 0.0,
    "epochs": 5,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
}


DEFAULTS: dict[ApproachName | Literal["common"], dict[str, Any]] = {
    "common": COMMON_CONFIG,
    "tfidf-ffnn": DEFAULT_TFIDF_FFNN_CONFIG,
    "tfidf-linear": TFIDF_LINEAR_DEFAULT_CONFIG,
    "transformer": TRANSFORMER_DEFAULT_CONFIG,
    "sequence-dl": SEQUENCE_DL_DEFAULT_CONFIG,
    "bpe-rnn": DEFAULT_BPE_LSTM,
}


def get_approach_defaults(approach_name: ApproachName) -> dict[str, Any]:
    defaults = DEFAULTS.get("common", {})
    if approach_name != "common":
        defaults = {**defaults, **DEFAULTS.get(approach_name, {})}
    return defaults


__all__ = ["get_approach_defaults"]
