from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union, Any

import matplotlib.pyplot as plt

from automl.core import registry
from automl.core.approaches.base_approach import Approach
from automl.core.datasets import get_dataset_class, BaseTextDataset
from automl.core.types import DatasetSplit, TrainResult
from automl.core.utils.misc import get_device, SavedIncumbent
from automl.core.utils.misc import numpy_and_config_encoder
from automl.logger import get_logger

logger = get_logger("")


def load_incumbent(path: Union[str, Path]):
    path = Path(path)
    logger.info(f"Loading incumbent configuration from {path}")

    if not path.exists():
        logger.error(f"Incumbent file does not exist: {path}")
        raise ValueError(f"Incumbent file does not exist: {path}")

    if not path.is_file():
        logger.error(f"Incumbent path is not a file: {path}")
        raise ValueError(f"Incumbent path is not a file: {path}")

    try:
        with path.open() as f:
            incumbent: SavedIncumbent = json.load(f)
    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse JSON from incumbent file: {path}")
        raise

    logger.debug(f"Loaded incumbent keys: {list(incumbent.keys())}")
    return incumbent["incumbent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training for an incumbent configuration and save results."
    )
    parser.add_argument(
        "--incumbent",
        "-i",
        type=Path,
        required=True,
        help="Path to incumbent.json",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (used with get_dataset_class)",
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
        default=0.8,
        help="Fraction of data used for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("incumbent_results"),
        help="Directory where model, results and plots are saved (default: ./incumbent_results)",
    )
    args = parser.parse_args()

    logger.info(
        "Parsed arguments: "
        f"incumbent={args.incumbent}, "
        f"dataset={args.dataset}, "
        f"data_path={args.data_path}, "
        f"val_size={args.val_size}, "
        f"train_fraction={args.train_fraction}, "
        f"seed={args.seed}, "
        f"epochs={args.epochs}, "
        f"output_dir={args.output_dir}"
    )

    return args


def evaluate_incumbent(
    incumbent: dict[str, Any], epochs: int, dataset: BaseTextDataset, seed: int
):
    """Same evaluation protocol as SmacOptimizer."""

    logger.info(
        f"[{__file__}] Retraining incumbent on full train data (epochs={epochs})..."
    )

    data_info = dataset.create_dataloaders(
        val_size=0.0,
        random_state=seed,
    )
    train_df, test_df = data_info["train_df"], data_info["test_df"]

    train_split = DatasetSplit(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
    )
    test_split = DatasetSplit(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
    )

    approach: Approach = registry.get_approach(incumbent.get("model_type"))(
        incumbent,
        data_info["num_classes"],
        get_device(),
        num_workers=0,  # TODO: Get from runtime config
    )

    with approach.with_mode("eval") as _approach:
        prepared = _approach.prepare(train_split, test_split)
        train_result = _approach.train(prepared, epochs=epochs)

    logger.info(
        f"[{__file__}] Final Held-Out Test Accuracy: "
        f"{train_result['val_accuracy']:.4f}"
    )

    return train_result


def plot_history(train_result: TrainResult, output_dir: Path, model_type: str) -> None:
    """
    Generate and save training/validation curves to files.

    Expects:
        train_result["history"] to be a list of EpochResult dicts.
    """
    logger.info("Generating training history plots")
    history = train_result.get("history", [])
    if not history:
        logger.warning("No history found in train_result; skipping plots.")
        return

    epochs = [ep["epoch"] for ep in history]

    train_losses = [ep.get("train_loss") for ep in history]
    val_accuracies = [ep.get("val_accuracy") for ep in history]

    # Filter None if necessary (e.g., some epochs may not have metrics)
    def _filter_none(xs, ys):
        return [(x, y) for x, y in zip(xs, ys) if y is not None]

    # Plot training loss
    ep_loss = _filter_none(epochs, train_losses)
    if ep_loss:
        x_loss, y_loss = zip(*ep_loss)
        plt.figure()
        plt.plot(x_loss, y_loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title(f"Training Loss - {model_type}")
        plt.grid(True)
        loss_path = output_dir / f"train_loss_{model_type}.png"
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved training loss plot to {loss_path}")
    else:
        logger.warning("No valid train_loss values found; skipping loss plot.")

    # Plot validation accuracy
    ep_acc = _filter_none(epochs, val_accuracies)
    if ep_acc:
        x_acc, y_acc = zip(*ep_acc)
        plt.figure()
        plt.plot(x_acc, y_acc, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(f"Validation Accuracy - {model_type}")
        plt.grid(True)
        acc_path = output_dir / f"val_accuracy_{model_type}.png"
        plt.savefig(acc_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved validation accuracy plot to {acc_path}")
    else:
        logger.warning("No valid val_accuracy values found; skipping accuracy plot.")


def main():
    args = parse_args()
    logger.info("Starting incumbent training run")

    logger.info("Registering all approaches")
    registry.register_all_approaches()

    incumbent = load_incumbent(args.incumbent)

    if "model_type" not in incumbent:
        logger.error("Incumbent configuration missing 'model_type' key")
        raise KeyError("Incumbent configuration missing 'model_type' key")

    model_type = incumbent.get("model_type", "<Unspecified>")
    logger.info(f"Using model_type={model_type}")

    dataset_name = args.dataset
    logger.info(f"Loading dataset '{dataset_name}' from {args.data_path}")
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(args.data_path)

    logger.info(
        f"Creating dataloaders with val_size={args.val_size}, "
        f"train_fraction={args.train_fraction}, seed={args.seed}"
    )
    data_info = dataset.create_dataloaders(
        val_size=args.val_size,
        random_state=args.seed,
        train_fraction=args.train_fraction,
    )
    train_df, val_df = data_info["train_df"], data_info["val_df"]
    logger.info(f"Train size: {len(train_df)}, " f"Validation size: {len(val_df)}")

    train_split = DatasetSplit(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
    )
    val_split = DatasetSplit(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
    )

    logger.info(f"Initializing approach for model_type={model_type}")
    ApproachCls = registry.get_approach(model_type)
    approach = ApproachCls(incumbent, dataset.get_num_classes())

    logger.info("Preparing approach (preprocessing, model init, etc.)")
    prep_result = approach.prepare(train_split, val_split)

    logger.info(f"Starting training for {args.epochs} epochs")
    train_result: TrainResult = approach.train(prep_result, epochs=args.epochs)
    logger.info("Training finished")

    logger.info(f"Retraining on all data")
    eval_result = evaluate_incumbent(incumbent, args.epochs, dataset, args.seed)
    logger.info(f"Retraining evaluation result: {eval_result}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    # Save model/checkpoint
    logger.info("Saving model/checkpoint")
    approach.save(output_dir)

    # Save train_result JSON
    train_result_path = output_dir / "train_result.json"
    logger.info(f"Saving training result JSON to {train_result_path}")
    with train_result_path.open("w") as f:
        json.dump(train_result, f, default=numpy_and_config_encoder)
    logger.info("Training result JSON saved successfully")

    # Generate and save plots
    plot_history(train_result, output_dir, model_type)

    final_val_acc = train_result["val_accuracy"]
    logger.info(f"Final validation accuracy: {final_val_acc}")


if __name__ == "__main__":
    main()
