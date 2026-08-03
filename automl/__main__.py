from __future__ import annotations

import json
import logging

from automl.core.registry import register_all_approaches
from .core import optimizers
from .core.utils import timer
from .core.utils.misc import set_seed
from .cli import load_runtime_config, RuntimeConfig
from .environment.device_info import get_device_info, save_device_info
from .environment.save_requirements import save_requirements
from .logger import setup_logging


def main(config: RuntimeConfig):
    register_all_approaches()
    device_info = get_device_info()
    save_device_info(device_info, config["output_path"] / "device_info.json")
    save_requirements(config["output_path"] / "requirements.txt")
    with open(config["output_path"] / "runtime_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    set_seed(config["seed"])
    optimizer_classes = {
        "smac": optimizers.SmacOptimizer,
        "random": optimizers.RandomSearch,
        "ifbo": optimizers.IfboOptimizer,
    }
    optimizer_name = config.get("optimizer", "smac")
    optimizer = optimizer_classes[optimizer_name](config)

    logger.info("Starting main optimization...")
    with timer.Timer("Main Optimization") as t:
        optimizer.run()
    logger.info(f"Main optimization completed in {t.formatted_execution_time}.")
    logger.info(f"Results saved at '{config['output_path']}'")


def run():
    """Convenience function if this module is imported and executed programmatically."""
    runtime_config = load_runtime_config()
    main(runtime_config)


if __name__ == "__main__":
    _runtime_config = load_runtime_config()
    setup_logging(
        output_path=_runtime_config["output_path"] / "app.log",
        level=_runtime_config["log_level"],
    )
    logger = logging.getLogger()
    main(_runtime_config)
else:
    logger = logging.getLogger()
