import dataclasses
from pathlib import Path
from typing import TypedDict, Optional

from automl.core.types import ApproachName


class RuntimeConfig(TypedDict):
    """Custom TypedDict mapping to configuration parameters."""

    runtime_id: str
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

    enable_jsonl_history: bool
    num_workers: int
    optimizer: str

    log_level: str

    use_random_selection: bool
    ifbo_greedy_candidate_selection: bool
