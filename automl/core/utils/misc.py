import json
from pathlib import Path
from typing import Union, TypedDict, Any

from ConfigSpace import Configuration

from automl.core.types import TrainResult
from automl.logger import get_logger

logger = get_logger()


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
    evaluation_result: TrainResult


def save_incumbent(
    incumbent: SavedIncumbent | list[SavedIncumbent],
    output_path: Union[str, Path],
    filename: str = "incumbent.json",
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
        data = [
            {
                "incumbent": _serialize_config(config["incumbent"]),
                "evaluation_result": config["evaluation_result"],
            }
            for config in incumbent
        ]
    else:
        data = {
            "incumbent": _serialize_config(incumbent["incumbent"]),
            "evaluation_result": incumbent["evaluation_result"],
        }

    out_file = output_path / filename

    with open(out_file, "w") as f:
        json.dump(data, f, indent=4, default=numpy_and_config_encoder)

    return out_file
