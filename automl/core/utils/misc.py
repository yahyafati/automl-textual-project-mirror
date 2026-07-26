import json
import os
from pathlib import Path
from typing import Union, TypedDict, Any

from ConfigSpace import Configuration

from automl.core.types import TrainResult, EvaluationResult
from automl.logger import get_logger

logger = get_logger()


def atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    """Writes `obj` via `torch.save` without ever leaving a truncated file at
    `path`.

    `torch.save` writes directly to its target file, so a process killed
    mid-write (Ctrl-C, OOM-kill, crash) leaves a corrupt, half-written
    checkpoint - `torch.load` on that file later fails with a miniz "failed
    finding central directory" error, indistinguishable from real disk
    corruption. Saving to a temp file in the same directory and
    `os.replace`-ing it into place makes the swap atomic: `path` always
    either holds the previous complete checkpoint or the new one, never a
    partial write.
    """
    import torch

    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(verbose: bool = False):
    import torch

    if torch.cuda.is_available():
        if verbose:
            logger.info(f"Using CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        if verbose:
            logger.info("Using MPS")
        return torch.device("mps")

    if verbose:
        logger.info("Using CPU")

    return torch.device("cpu")


def numpy_and_config_encoder(obj):
    import numpy as np

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_config(config: Configuration) -> dict:
    """
    Converts a single configuration (e.g., ConfigSpace.Configuration, dict, etc.)
    into a JSON-serializable dict.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return dict(config)


class SavedIncumbent(TypedDict):
    incumbent: Configuration
    evaluation_result: EvaluationResult


def save_incumbent(
    incumbent: SavedIncumbent | list[SavedIncumbent],
    output_path: Union[str, Path],
    filename: str = "incumbent.json",
    ensemble_evaluation_result: TrainResult | None = None,
) -> Path:
    """
    Saves an SMAC incumbent (single config or list of configs) to disk.
    """

    output_path = Path(output_path)
    filename = Path(filename)
    output_path.mkdir(parents=True, exist_ok=True)

    # Normalize to list
    data: dict[str, Any] | list[dict[str, Any]]
    if isinstance(incumbent, (list, tuple)):
        incumbent_data = [
            {
                "incumbent": _serialize_config(config["incumbent"]),
                "evaluation_result": config["evaluation_result"],
            }
            for config in incumbent
        ]
        if ensemble_evaluation_result is None:
            data = incumbent_data
        else:
            data = {
                "incumbents": incumbent_data,
                "ensemble_evaluation_result": ensemble_evaluation_result,
            }
    else:
        data = {
            "incumbent": _serialize_config(incumbent["incumbent"]),
            "evaluation_result": incumbent["evaluation_result"],
        }

    out_file = output_path / filename

    with open(out_file, "w") as f:
        json.dump(data, f, indent=4, default=numpy_and_config_encoder)

    return out_file
