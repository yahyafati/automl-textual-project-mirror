from pathlib import Path
from typing import TypedDict, Optional

from automl.core.types import ApproachName


class RuntimeConfigDict(TypedDict):
    """Custom TypedDict mapping to configuration parameters."""

    runtime_id: str
    dataset: str
    output_path: Path
    load_path: Optional[Path]
    data_path: Path
    seed: int
    approach: ApproachName
    vocab_size: int
    token_length: int

    evaluation_budget: int
    max_budget: int
    min_budget: int
    n_trials: int

    max_num_rows: int

    batch_size: int
    lr: float
    weight_decay: float
    lstm_emb_dim: int
    lstm_hidden_dim: int
    ffnn_hidden_layer_dim: int
    data_fraction: float

    enable_jsonl_history: bool
    num_workers: int
    optimizer: str
    max_trainers_in_memory: int
