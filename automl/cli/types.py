import dataclasses
from pathlib import Path
from typing import TypedDict, Optional

from automl.core.types import ApproachName


class RuntimeConfig(TypedDict):
    """Custom TypedDict mapping to configuration parameters."""

    runtime_id: str
    device: str
    dataset: str
    output_path: Path
    load_path: Optional[Path]
    data_path: Path
    seed: int
    approach: ApproachName

    evaluation_budget: int
    max_budget: int
    min_budget: int
    n_trials: int

    max_num_rows: int
    val_size: float

    enable_jsonl_history: bool
    num_workers: int
    optimizer: str

    # Number of ifBO trials to run concurrently (one per device, or
    # round-robin across devices if this exceeds the number of visible
    # GPUs). Defaults to 1 = today's fully sequential behavior.
    num_parallel_trials: int

    log_level: str

    use_random_selection: bool
    ifbo_greedy_candidate_selection: bool
    ifbo_incumbent_ensemble_top_k: int
    ifbo_incumbent_ensemble_accuracy_threshold: float
