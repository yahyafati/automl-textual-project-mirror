from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from automl.core import registry
from automl.core.approaches.base_approach import Approach
from automl.core.datasets import get_dataset_class
from automl.core.types import (
    DatasetSplit,
    EvaluationResult,
    PredictionResult,
    TrainResult,
)
from automl.core.utils.misc import (
    SavedIncumbent,
    get_device,
    numpy_and_config_encoder,
    save_incumbent,
    set_seed,
)
from automl.logger import get_logger

logger = get_logger("")

MANIFEST_FILENAME = "manifest.json"
HISTORY_COPY_FILENAME = "history.log.jsonl"
HELDOUT_LABELS_FILENAME = "heldout_labels.npy"

DEFAULT_TOP_K = 5
DEFAULT_EPOCHS = 50
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 2
DEFAULT_DATA_FRACTION = 1.0
DEFAULT_DEVICE = "auto"
DEFAULT_DATA_PATH = Path("data")
DEFAULT_STOCHASTIC_EPOCHS = False
DEFAULT_STOCHASTIC_EPOCH_FRACTION = 0.25

# Standard Unix convention: 128 + SIGINT(2).
SIGINT_EXIT_CODE = 130


def manifest_path_for(output_dir: Path) -> Path:
    return output_dir / MANIFEST_FILENAME


def load_manifest(output_dir: Path) -> Optional[dict[str, Any]]:
    path = manifest_path_for(output_dir)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path_for(output_dir)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, default=numpy_and_config_encoder)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read trial configs from a history.log.jsonl file (as produced by "
            "the optimizers under automl/core/optimizers), retrain the top-k "
            "configs by validation accuracy on the full training set, save "
            "each trained model, and evaluate a majority-vote ensemble of the "
            "incumbents on the held-out test set (same protocol as "
            "Optimizer._finalize_optimization in base_optimizer.py). "
            "Re-running with the same --output-dir resumes an interrupted "
            "run instead of starting over: already-trained incumbents are "
            "loaded from disk instead of being retrained. Ctrl-C is handled "
            "safely - whatever has finished so far (and, where possible, the "
            "in-flight incumbent's partial training progress) is saved "
            "before exiting."
        )
    )
    parser.add_argument(
        "--history",
        "-H",
        type=Path,
        default=None,
        help="Path to a history.log.jsonl (or history.log.ignore.jsonl) file. "
        "Required to start a fresh run. When resuming (an existing "
        "--output-dir with a manifest.json), this is ignored in favor of "
        "the configs recorded in that manifest unless explicitly passed, "
        "in which case the top-k configs are re-selected from this file "
        "(already-completed incumbents whose config is unchanged are still "
        "reused).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        choices=["ag_news", "imdb", "amazon", "dbpedia", "yelp"],
        help="Required to start a fresh run. Falls back to the value "
        "recorded in manifest.json when resuming.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Root path where dataset data is stored (default: ./data, or "
        "the value recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of distinct configs to retrain and ensemble, ranked by "
        "their best validation accuracy in the history file (default: 5, "
        "or the value recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to retrain each selected config on the full "
        "training set (default: 50, or the value recorded in manifest.json "
        "when resuming). Raising this on a resumed run continues training "
        "each already-completed incumbent for the extra epochs instead of "
        "retraining from scratch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for data splitting/training (default: 42, or the "
        "value recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers (default: 2, or the value "
        "recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Fraction of the full training set to retrain each incumbent "
        "on, stratified by label (default: 1.0, i.e. all training data, or "
        "the value recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'auto', 'cpu', 'cuda', 'cuda:0', 'mps', ... (default: auto, "
        "or the value recorded in manifest.json when resuming)",
    )
    parser.add_argument(
        "--stochastic-epochs",
        action="store_true",
        default=None,
        help="Sample a random fraction of the training batches each epoch "
        "instead of iterating the full dataset, same as torch_trainer.py's "
        "TorchTrainer (see --stochastic-epoch-fraction). Default: False, "
        "or the value recorded in manifest.json when resuming.",
    )
    parser.add_argument(
        "--stochastic-epoch-fraction",
        type=float,
        default=None,
        help="Fraction of training batches to draw per epoch when "
        "--stochastic-epochs is set, in (0, 1] (default: 0.25, or the "
        "value recorded in manifest.json when resuming).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("topk_results"),
        help="Directory where models, predictions, incumbent.json and "
        "manifest.json are saved (default: ./topk_results). Pointing this "
        "at an existing run's directory resumes that run.",
    )
    args = parser.parse_args()

    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.data_fraction is not None and not 0.0 < args.data_fraction <= 1.0:
        parser.error("--data-fraction must be in (0, 1]")
    if args.stochastic_epoch_fraction is not None and not (
        0.0 < args.stochastic_epoch_fraction <= 1.0
    ):
        parser.error("--stochastic-epoch-fraction must be in (0, 1]")

    manifest = load_manifest(args.output_dir)
    if manifest is None:
        if args.history is None:
            parser.error(
                "--history is required to start a fresh run (no "
                f"{MANIFEST_FILENAME} found in --output-dir)"
            )
        if args.dataset is None:
            parser.error(
                "--dataset is required to start a fresh run (no "
                f"{MANIFEST_FILENAME} found in --output-dir)"
            )

    logger.info(
        "Parsed arguments: "
        f"history={args.history}, dataset={args.dataset}, data_path={args.data_path}, "
        f"top_k={args.top_k}, epochs={args.epochs}, seed={args.seed}, "
        f"num_workers={args.num_workers}, device={args.device}, output_dir={args.output_dir}, "
        f"data_fraction={args.data_fraction}, stochastic_epochs={args.stochastic_epochs}, "
        f"stochastic_epoch_fraction={args.stochastic_epoch_fraction}"
    )
    return args


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"History file does not exist: {path}")

    trials: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON on line {line_no} of {path}")
    return trials


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def select_top_k_configs(
    trials: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """
    Deduplicate trials by configuration (keeping the entry with the lowest
    val_error seen for each distinct config, since the same config may have
    been evaluated multiple times at different budgets) and return the
    top_k configs ranked by validation accuracy (1 - val_error), best first.
    """
    best_by_config: dict[str, dict[str, Any]] = {}
    for trial in trials:
        config = trial.get("config")
        val_error = trial.get("val_error")
        if not config or val_error is None:
            continue
        if isinstance(val_error, float) and np.isnan(val_error):
            continue

        key = _config_key(config)
        current_best = best_by_config.get(key)
        if current_best is None or val_error < current_best["val_error"]:
            best_by_config[key] = trial

    ranked = sorted(best_by_config.values(), key=lambda t: t["val_error"])
    return ranked[:top_k]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device(verbose=True)
    return torch.device(device_arg)


def save_test_predictions(
    predictions: np.ndarray, output_dir: Path, filename: str
) -> Path:
    """Mirrors Optimizer._save_test_predictions in base_optimizer.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / filename
    np.save(predictions_path, predictions)
    logger.info(f"Saved test predictions to {predictions_path}")
    return predictions_path


def _majority_vote(labels: np.ndarray):
    classes, counts = np.unique(labels, return_counts=True)
    return classes[int(np.argmax(counts))]


def save_ensemble_predictions(
    incumbent_predictions: list[np.ndarray],
    heldout_labels: np.ndarray,
    output_dir: Path,
    filename: str = "predictions.npy",
) -> TrainResult:
    """
    Persist a deterministic majority-vote ensemble and compute held-out
    accuracy. Mirrors Optimizer._save_ensemble_predictions in
    base_optimizer.py (ties broken by np.unique's natural sort order).
    """
    if not incumbent_predictions:
        raise ValueError("Cannot ensemble an empty prediction list.")

    prediction_shapes = {pred.shape for pred in incumbent_predictions}
    if len(prediction_shapes) != 1:
        raise ValueError(
            "Cannot ensemble incumbent predictions with different shapes: "
            f"{sorted(prediction_shapes)}"
        )

    stacked_predictions = np.stack(incumbent_predictions, axis=0)
    ensemble_predictions = np.apply_along_axis(
        _majority_vote, axis=0, arr=stacked_predictions
    )
    save_test_predictions(ensemble_predictions, output_dir, filename)

    if ensemble_predictions.shape != heldout_labels.shape:
        raise ValueError(
            "Cannot compute ensemble accuracy because predictions and labels "
            f"have different shapes: {ensemble_predictions.shape} vs "
            f"{heldout_labels.shape}."
        )

    ensemble_accuracy = float(np.mean(ensemble_predictions == heldout_labels))
    logger.info(
        f"Saved majority-vote ensemble from {len(incumbent_predictions)} "
        f"incumbents with held-out accuracy {ensemble_accuracy:.4f}."
    )
    return TrainResult(val_accuracy=ensemble_accuracy, history=[])


def incumbent_trainer_checkpoint_path(
    incumbent_checkpoint_dir: Path, model_type: str
) -> Path:
    """Where Approach.save()/approach.save() writes trainer weights for a
    given incumbent (see Approach.save in base_approach.py: it nests output
    under a `self.name` subfolder)."""
    return incumbent_checkpoint_dir / model_type / "trainer.pth"


def load_trained_approach(
    checkpoint_dir: Path,
    train_split: DatasetSplit,
    test_split: DatasetSplit,
    device: torch.device,
    num_workers: int = 2,
) -> Approach:
    """
    Fully restore a previously-saved incumbent: reload its config via
    Approach.load(), rebuild the model/dataloaders via prepare() (weights
    are randomly initialized again at this point), then load the trained
    weights from the checkpoint's trainer.pth on top. The returned Approach
    is ready for `.predict(...)` without any further training.
    """
    approach = Approach.load(checkpoint_dir)
    with approach.with_mode("eval") as _approach:
        prepared = _approach.prepare(train_split, test_split)
        trainer_checkpoint = checkpoint_dir / "trainer.pth"
        if _approach.trainer is not None and trainer_checkpoint.exists():
            _approach.trainer.load(trainer_checkpoint)
    return approach


def _save_partial_checkpoint(approach: Approach, checkpoint_dir: Path) -> None:
    """Best-effort checkpoint save used when training is interrupted
    mid-run: never let a secondary failure here mask the interrupt."""
    if approach.trainer is None:
        logger.warning(
            "Interrupted before any trainer state existed; nothing to checkpoint."
        )
        return
    try:
        approach.save(checkpoint_dir)
        logger.warning(f"Saved partial model checkpoint to {checkpoint_dir}")
    except Exception:
        logger.exception(f"Failed to save partial checkpoint to {checkpoint_dir}")


def train_and_evaluate_config(
    config: dict[str, Any],
    epochs: int,
    dataset,
    seed: int,
    device: torch.device,
    num_workers: int,
    output_dir: Path,
    predictions_filename: str,
    checkpoint_dir: Path,
    data_fraction: float = 1.0,
    resume_checkpoint: Optional[Path] = None,
    stochastic_epochs: bool = DEFAULT_STOCHASTIC_EPOCHS,
    stochastic_epoch_fraction: float = DEFAULT_STOCHASTIC_EPOCH_FRACTION,
) -> EvaluationResult:
    """
    Retrain a config on the (optionally subsampled) training set and
    evaluate it on the held-out test set. Mirrors Optimizer.evaluate_incumbent
    in base_optimizer.py (same data split, same train/predict calls), but
    additionally persists the trained model to `checkpoint_dir` as soon as
    training finishes (before prediction), and - if training itself is
    interrupted (KeyboardInterrupt) - saves whatever partial progress the
    approach's trainer has made so far before re-raising, so a Ctrl-C never
    throws away completed epochs.

    If `resume_checkpoint` points at an existing trainer checkpoint (e.g.
    from a previous, interrupted run of this script), it is handed to the
    approach's train() call the same way base_optimizer.py resumes trials
    (`load_path`/`trainer_load_path`): approaches that track an epoch
    counter (the torch-based ones) will pick up where they left off, or
    skip training entirely if `epochs` was already reached.
    """
    model_type = config["model_type"]
    logger.info(
        f"Retraining config (model_type={model_type}) on {data_fraction:.2%} "
        f"of the train data (epochs={epochs})..."
    )

    data_info = dataset.create_dataloaders(
        val_size=0.0, random_state=seed, train_fraction=data_fraction
    )
    train_df, test_df = data_info["train_df"], data_info["test_df"]
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    train_split = DatasetSplit(
        texts=train_df["text"].tolist(), labels=train_df["label"].tolist()
    )
    test_split = DatasetSplit(
        texts=test_df["text"].tolist(), labels=test_df["label"].tolist()
    )

    should_evaluate = not pd.isna(test_split.labels[0])

    approach: Approach = registry.get_approach(model_type)(
        config,
        data_info["num_classes"],
        device,
        num_workers=num_workers,
        stochastic_epochs=stochastic_epochs,
        stochastic_epoch_fraction=stochastic_epoch_fraction,
    )

    load_kwargs: dict[str, Path] = {}
    if resume_checkpoint is not None and resume_checkpoint.exists():
        logger.info(
            f"Found existing checkpoint at {resume_checkpoint}; resuming "
            "training from it instead of starting from scratch."
        )
        # Different approaches name this kwarg differently; passing both is
        # harmless since each train() only reads the one it declares (see
        # Optimizer._trainer_load_kwargs in base_optimizer.py).
        load_kwargs = {
            "load_path": resume_checkpoint,
            "trainer_load_path": resume_checkpoint,
        }

    set_seed(seed)
    with approach.with_mode("eval") as _approach:
        prepared = _approach.prepare(train_split, test_split)
        try:
            train_result = _approach.train(
                prepared,
                epochs=epochs,
                evaluate_validation=should_evaluate,
                **load_kwargs,
            )
        except KeyboardInterrupt:
            logger.warning(
                f"Training interrupted for config (model_type={model_type}); "
                "saving whatever progress was made before exiting."
            )
            _save_partial_checkpoint(_approach, checkpoint_dir)
            raise

        logger.info(f"Saving trained model to {checkpoint_dir}")
        _approach.save(checkpoint_dir)

        try:
            logger.info("Predicting for test set")
            prediction_result = _approach.predict(test_df)
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted while predicting on the test set; the trained "
                f"model was already saved to {checkpoint_dir}."
            )
            raise

    logger.info(f"Held-out test accuracy: {train_result['val_accuracy']:.4f}")
    save_test_predictions(
        prediction_result["y_pred"], output_dir, filename=predictions_filename
    )

    evaluation_result = EvaluationResult(
        train_result=train_result, prediction_result=prediction_result
    )
    return evaluation_result


def _pick(cli_value: Any, manifest: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if manifest is not None and key in manifest:
        return manifest[key]
    return default


def main():
    args = parse_args()

    logger.info("Registering all approaches")
    registry.register_all_approaches()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_manifest = load_manifest(output_dir)
    resuming = old_manifest is not None
    if resuming:
        logger.info(
            f"Found existing {MANIFEST_FILENAME} in {output_dir}; resuming previous run."
        )

    dataset_name: str = _pick(args.dataset, old_manifest, "dataset", None)
    data_path = Path(_pick(args.data_path, old_manifest, "data_path", DEFAULT_DATA_PATH))
    top_k: int = _pick(args.top_k, old_manifest, "top_k", DEFAULT_TOP_K)
    epochs: int = _pick(args.epochs, old_manifest, "epochs", DEFAULT_EPOCHS)
    seed: int = _pick(args.seed, old_manifest, "seed", DEFAULT_SEED)
    num_workers: int = _pick(args.num_workers, old_manifest, "num_workers", DEFAULT_NUM_WORKERS)
    data_fraction: float = _pick(
        args.data_fraction, old_manifest, "data_fraction", DEFAULT_DATA_FRACTION
    )
    device_arg: str = _pick(args.device, old_manifest, "device", DEFAULT_DEVICE)
    device = resolve_device(device_arg)
    stochastic_epochs: bool = _pick(
        args.stochastic_epochs, old_manifest, "stochastic_epochs", DEFAULT_STOCHASTIC_EPOCHS
    )
    stochastic_epoch_fraction: float = _pick(
        args.stochastic_epoch_fraction,
        old_manifest,
        "stochastic_epoch_fraction",
        DEFAULT_STOCHASTIC_EPOCH_FRACTION,
    )

    # --- select (or reuse) the top-k configs to train ---
    if resuming and args.history is None:
        logger.info(
            "No --history passed; reusing the configs recorded in "
            f"{MANIFEST_FILENAME}."
        )
        top_trials: list[dict[str, Any]] = old_manifest["incumbents"]
        history_path_for_record = old_manifest.get("history_path")
    else:
        if resuming:
            logger.info(
                "--history passed while resuming; re-selecting the top-k "
                "configs (already-completed incumbents whose config is "
                "unchanged will still be reused)."
            )
        trials = load_history(args.history)
        logger.info(f"Loaded {len(trials)} trial(s) from {args.history}")
        raw_top_trials = select_top_k_configs(trials, top_k)
        if not raw_top_trials:
            raise ValueError(
                "No valid trials with a config and val_error found in history file."
            )
        top_trials = [
            {
                "config": t["config"],
                "val_error": t["val_error"],
                "trial_no": t.get("trialNo"),
                "budget": t.get("budget"),
            }
            for t in raw_top_trials
        ]
        history_path_for_record = str(args.history)

    logger.info(f"Selected top {len(top_trials)} config(s) by validation accuracy:")
    for i, trial in enumerate(top_trials):
        logger.info(
            f"  [{i}] val_accuracy={1.0 - trial['val_error']:.4f} "
            f"(trial_no={trial.get('trial_no')}, budget={trial.get('budget')}) "
            f"model_type={trial['config'].get('model_type')}"
        )

    dataset = get_dataset_class(dataset_name)(data_path)

    checkpoints_dir = output_dir / "checkpoints"
    heldout_labels_path = output_dir / HELDOUT_LABELS_FILENAME
    heldout_labels: Optional[np.ndarray] = (
        np.load(heldout_labels_path) if heldout_labels_path.exists() else None
    )

    has_multiple_incumbents = len(top_trials) > 1
    saved_incumbents: list[SavedIncumbent] = []
    incumbent_predictions: list[np.ndarray] = []
    incumbent_records: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "history_path": history_path_for_record,
        "dataset": dataset_name,
        "data_path": str(data_path),
        "top_k": top_k,
        "epochs": epochs,
        "seed": seed,
        "num_workers": num_workers,
        "data_fraction": data_fraction,
        "device": device_arg,
        "stochastic_epochs": stochastic_epochs,
        "stochastic_epoch_fraction": stochastic_epoch_fraction,
        "has_multiple_incumbents": has_multiple_incumbents,
        "incumbents": incumbent_records,
        "ensemble_completed": False,
        "ensemble_val_accuracy": None,
    }
    # In case of an interruption before the very first incumbent finishes,
    # make sure a manifest still exists on disk recording this run's args.
    save_manifest(manifest, output_dir)

    try:
        for idx, trial in enumerate(top_trials):
            config = trial["config"]
            model_type = config["model_type"]
            predictions_filename = (
                f"predictions_incumbent_{idx}.npy"
                if has_multiple_incumbents
                else "predictions.npy"
            )
            predictions_path = output_dir / predictions_filename
            incumbent_checkpoint_dir = checkpoints_dir / f"incumbent_{idx}"
            trainer_checkpoint = incumbent_trainer_checkpoint_path(
                incumbent_checkpoint_dir, model_type
            )

            prior_record: Optional[dict[str, Any]] = None
            if old_manifest is not None:
                old_incumbents = old_manifest.get("incumbents", [])
                if (
                    idx < len(old_incumbents)
                    and old_incumbents[idx].get("config") == config
                ):
                    prior_record = old_incumbents[idx]

            # The predictions filename depends on has_multiple_incumbents,
            # which can flip between runs (e.g. --top-k grows from 1 to
            # 2+), so look for the cached file under whatever name it was
            # saved as, not just the name this run would use.
            cached_predictions_path: Optional[Path] = None
            if prior_record is not None and prior_record.get("predictions_file"):
                candidate = output_dir / prior_record["predictions_file"]
                if candidate.exists():
                    cached_predictions_path = candidate

            already_done = (
                prior_record is not None
                and prior_record.get("status") == "completed"
                and prior_record.get("epochs_trained") == epochs
                and cached_predictions_path is not None
                and trainer_checkpoint.exists()
            )

            if already_done:
                logger.info(
                    f"Incumbent {idx} already trained for {epochs} epoch(s) in a "
                    f"previous run; loading cached predictions/model from "
                    f"{incumbent_checkpoint_dir} instead of retraining."
                )
                if heldout_labels is None:
                    raise RuntimeError(
                        f"Cannot resume: {cached_predictions_path} exists but no "
                        f"cached held-out labels were found at {heldout_labels_path}."
                    )
                y_pred = np.load(cached_predictions_path)
                if cached_predictions_path != predictions_path:
                    # Naming convention changed since the cached file was
                    # written (e.g. top-k grew past 1); keep the on-disk
                    # layout consistent with this run's convention.
                    np.save(predictions_path, y_pred)
                evaluation_result = EvaluationResult(
                    train_result=TrainResult(
                        val_accuracy=prior_record["train_val_accuracy"],
                        history=prior_record.get("history", []),
                    ),
                    prediction_result=PredictionResult(
                        y_pred=y_pred, y_true=heldout_labels
                    ),
                )
                record = dict(prior_record)
                record["predictions_file"] = predictions_filename
            else:
                # SklearnTrainer.train() always refits from scratch
                # regardless of load_path (see sklearn_trainer.py), and
                # tfidf_linear.py's own self.model isn't re-synced when the
                # trainer swaps its model reference on load - handing it a
                # stale checkpoint gains nothing and leaves self.model
                # unfitted. Only pass a resume checkpoint to approaches
                # that can actually use one.
                can_resume_training = model_type != "tfidf-linear"
                evaluation_result = train_and_evaluate_config(
                    config=config,
                    epochs=epochs,
                    dataset=dataset,
                    seed=seed,
                    device=device,
                    num_workers=num_workers,
                    output_dir=output_dir,
                    predictions_filename=predictions_filename,
                    checkpoint_dir=incumbent_checkpoint_dir,
                    data_fraction=data_fraction,
                    resume_checkpoint=(
                        trainer_checkpoint
                        if can_resume_training and trainer_checkpoint.exists()
                        else None
                    ),
                    stochastic_epochs=stochastic_epochs,
                    stochastic_epoch_fraction=stochastic_epoch_fraction,
                )

                labels = evaluation_result["prediction_result"]["y_true"]
                if heldout_labels is None:
                    heldout_labels = labels
                    np.save(heldout_labels_path, heldout_labels)
                    logger.info(f"Saved held-out labels to {heldout_labels_path}")
                elif not np.array_equal(heldout_labels, labels):
                    raise ValueError(
                        "Cannot compute ensemble accuracy because incumbent "
                        "evaluations used different held-out labels."
                    )

                record = {
                    "idx": idx,
                    "config": config,
                    "val_error": trial.get("val_error"),
                    "trial_no": trial.get("trial_no"),
                    "budget": trial.get("budget"),
                    "model_type": model_type,
                    "checkpoint_dir": str(
                        incumbent_checkpoint_dir.relative_to(output_dir)
                    ),
                    "predictions_file": predictions_filename,
                    "status": "completed",
                    "epochs_trained": epochs,
                    "train_val_accuracy": evaluation_result["train_result"][
                        "val_accuracy"
                    ],
                    "history": evaluation_result["train_result"]["history"],
                }

            incumbent_records.append(record)
            if has_multiple_incumbents:
                incumbent_predictions.append(
                    evaluation_result["prediction_result"]["y_pred"]
                )

            saved_incumbents.append(
                SavedIncumbent(incumbent=config, evaluation_result=evaluation_result)
            )

            # Persist progress after every incumbent, so an interruption
            # only costs the in-flight incumbent, not everything completed
            # so far.
            save_manifest(manifest, output_dir)

        ensemble_evaluation_result: Optional[TrainResult] = None
        if has_multiple_incumbents:
            if heldout_labels is None:
                raise ValueError(
                    "Cannot compute ensemble accuracy without held-out labels."
                )
            ensemble_evaluation_result = save_ensemble_predictions(
                incumbent_predictions, heldout_labels, output_dir=output_dir
            )

        incumbent_path = save_incumbent(
            incumbent=saved_incumbents,
            output_path=output_dir,
            ensemble_evaluation_result=ensemble_evaluation_result,
        )
        logger.info(f"Saved incumbent metadata to {incumbent_path}")

        manifest["ensemble_completed"] = True
        manifest["ensemble_val_accuracy"] = (
            ensemble_evaluation_result["val_accuracy"]
            if ensemble_evaluation_result is not None
            else None
        )
        save_manifest(manifest, output_dir)

        # Keep a copy of whichever history file was used to select these
        # configs, so the run directory is self-contained and reproducible.
        if args.history is not None:
            history_copy_path = output_dir / HISTORY_COPY_FILENAME
            shutil.copyfile(args.history, history_copy_path)
            logger.info(f"Saved a copy of the history file to {history_copy_path}")

        if ensemble_evaluation_result is not None:
            logger.info(
                f"Final ensemble held-out accuracy: {ensemble_evaluation_result['val_accuracy']:.4f}"
            )
        else:
            acc = saved_incumbents[0]["evaluation_result"]["train_result"]["val_accuracy"]
            logger.info(f"Final held-out accuracy: {acc:.4f}")

    except KeyboardInterrupt:
        completed = sum(
            1 for r in incumbent_records if r.get("status") == "completed"
        )
        logger.warning(
            f"Interrupted by user. {completed}/{len(top_trials)} incumbent(s) "
            f"finished and were saved under {output_dir}. Re-run with "
            f"--output-dir {output_dir} to resume."
        )
        sys.exit(SIGINT_EXIT_CODE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Safety net for an interrupt landing outside main()'s own handler
        # (e.g. during argument parsing or approach registration, before
        # there is anything to save).
        logger.warning("Interrupted by user before any work could be saved.")
        sys.exit(SIGINT_EXIT_CODE)
