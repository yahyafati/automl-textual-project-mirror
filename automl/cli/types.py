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

    max_trial_time_seconds: Optional[float]

    max_num_rows: int
    val_size: float

    enable_jsonl_history: bool
    num_workers: int
    optimizer: str

    evaluate_incumbent: bool

    stochastic_epochs: bool
    stochastic_epoch_fraction: float

    num_parallel_trials: int

    log_level: str

    ifbo_use_random_selection: bool
    ifbo_greedy_candidate_selection: bool
    ifbo_incumbent_ensemble_top_k: int
    ifbo_incumbent_ensemble_accuracy_threshold: float
    # Wall-clock cap (seconds) on a single ifBO freeze-thaw step.
    ifbo_thaw_step: float
