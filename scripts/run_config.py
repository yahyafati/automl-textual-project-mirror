from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from automl.core import registry
from automl.core.datasets import get_dataset_class
from automl.core.types import DatasetSplit, TrainResult
from automl.core.utils.misc import get_device, numpy_and_config_encoder
from automl.logger import get_logger

logger = get_logger("")


def load_config(path: Path, model_type_override: Optional[str]) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")

    with path.open() as f:
        payload = json.load(f)

    # Accept either a flat {hyperparam: value, ...} dict, or an
    # incumbent.json-style {"incumbent": {...}} wrapper.
    config = dict(payload["incumbent"] if "incumbent" in payload else payload)

    if model_type_override:
        config["model_type"] = model_type_override

    if "model_type" not in config:
        raise KeyError(
            f"Config at {path} has no 'model_type' key; pass --model-type to set one."
        )

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single hyperparameter configuration for a fixed "
        "number of epochs on one dataset, and report per-epoch train/val metrics."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to a config JSON file (either a flat {hyperparam: value, "
        "...} dict with a 'model_type' key, or an incumbent.json with an "
        "{'incumbent': {...}} wrapper).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (ag_news, imdb, amazon, dbpedia, yelp)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        required=True,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override/supply the approach name if not present in the config JSON",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Root path where dataset data is stored (default: ./data)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation size fraction (default: 0.2)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to keep, stratified by label (default: 1.0)",
    )
    parser.add_argument(
        "--max-num-rows",
        type=int,
        default=None,
        help="Cap total training rows, sampled uniformly across classes "
        "(default: no cap). Useful for a quick smoke run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save the train_result JSON to",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_size <= 0:
        raise ValueError(
            "--val-size must be > 0: this script reports per-epoch validation "
            "metrics, which need a held-out split."
        )

    logger.info("Registering all approaches")
    registry.register_all_approaches()

    config = load_config(args.config, args.model_type)
    model_type = config["model_type"]
    logger.info(f"Loaded config for model_type={model_type} from {args.config}")

    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class(args.data_path)

    logger.info(
        f"Creating dataloaders (val_size={args.val_size}, "
        f"train_fraction={args.train_fraction}, "
        f"max_num_rows={args.max_num_rows}, seed={args.seed})"
    )
    data_info = dataset.create_dataloaders(
        val_size=args.val_size,
        random_state=args.seed,
        train_fraction=args.train_fraction,
        max_num_rows=args.max_num_rows,
    )
    train_df, val_df = data_info["train_df"], data_info["val_df"]
    logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    train_split = DatasetSplit(
        texts=train_df["text"].tolist(), labels=train_df["label"].tolist()
    )
    val_split = DatasetSplit(
        texts=val_df["text"].tolist(), labels=val_df["label"].tolist()
    )

    ApproachCls = registry.get_approach(model_type)
    approach = ApproachCls(
        config,
        data_info["num_classes"],
        get_device(),
        num_workers=args.num_workers,
    )

    logger.info("Preparing approach (preprocessing, model init, etc.)")
    prepared = approach.prepare(train_split, val_split)

    logger.info(f"Training for {args.epochs} epochs")
    train_result: TrainResult = approach.train(prepared, epochs=args.epochs)

    logger.info(f"Final validation accuracy: {train_result['val_accuracy']:.4f}")
    for epoch in train_result["history"]:
        logger.info(
            f"  epoch {epoch['epoch']}: "
            f"train_loss={epoch.get('train_loss')} "
            f"val_accuracy={epoch.get('val_accuracy')}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(train_result, f, indent=2, default=numpy_and_config_encoder)
        logger.info(f"Saved train_result to {args.output}")


if __name__ == "__main__":
    main()
