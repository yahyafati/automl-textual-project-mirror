import hashlib
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from filelock import FileLock

from automl.cli import RuntimeConfig
from automl.core import configspacehelper
from automl.core.approaches.base_approach import Approach
from automl.core.datasets import get_dataset_class
from automl.core.registry import get_approach, register_all_approaches
from automl.core.trainers.base_trainer import Trainer
from automl.core.types import (
    DatasetSplit,
    EpochResult,
    TrialResult,
    ApproachName,
    TrainResult,
    EvaluationResult,
)
from automl.core.utils.misc import SavedIncumbent
from automl.core.utils.misc import (
    atomic_torch_save,
    get_device,
    numpy_and_config_encoder,
    save_incumbent,
)
from automl.logger import get_logger
from automl.trial_plots import save_all_plots


class Optimizer(ABC):

    def __init__(self, runtime_config: RuntimeConfig):
        self.runtime_config = runtime_config
        self.output_path = runtime_config["output_path"]
        self.logger = get_logger()

        self.logger.info(f"Selected seed: {runtime_config['seed']}")
        self.logger.info(f"Current runtime_id: {runtime_config['runtime_id']}")

        self.devices: list[torch.device] = self._resolve_devices(runtime_config)
        self.device = self.devices[0]

        self._state_lock = threading.Lock()
        self._checkpoint_locks: dict[str, threading.Lock] = {}

        self.dataset = get_dataset_class(runtime_config["dataset"])(
            runtime_config["data_path"]
        )

        self.space: ConfigurationSpace = configspacehelper.build_config_space(
            runtime_config["seed"],
            fixed_model_type=runtime_config["approach"],
        )
        self.min_budget = runtime_config["min_budget"]
        self.max_budget = runtime_config["max_budget"]
        self.n_trials = runtime_config["n_trials"]
        self.logger.info(f"Space: \n{self.space}")

        self.history_file_path = self.output_path / "history.log.json"
        self.history: list[TrialResult] = []
        # Live JSONLines logging
        self.enable_jsonl_history: bool = runtime_config["enable_jsonl_history"]
        self.history_jsonl_path = self.output_path / "history.log.jsonl"
        self.evaluate_incumbent_enabled: bool = runtime_config.get(
            "evaluate_incumbent", True
        )

        self.highest_budget_seen: float = 0.0
        self.best_val_error: float = float("inf")
        self.checkpoint_dir = self.output_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.trial_no: int = 0

        self.trainer_checkpoint_dir = self.checkpoint_dir / "trainers"
        self.trainer_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        register_all_approaches()

    @staticmethod
    def _resolve_devices(runtime_config: RuntimeConfig) -> list[torch.device]:
        """
        Enumerate the devices available for (potentially concurrent) trial
        execution. An explicit `device` in the runtime config is always
        honored as a single device; "auto" expands to every visible CUDA
        device so callers that support parallel trials (see
        IfboOptimizer) can dispatch one trial per GPU, falling back to the
        single best device (mps/cpu) otherwise.
        """
        if runtime_config["device"] != "auto":
            return [torch.device(runtime_config["device"])]

        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        return [get_device()]

    def _checkpoint_lock_for(self, config_id: str) -> threading.Lock:
        with self._state_lock:
            return self._checkpoint_locks.setdefault(config_id, threading.Lock())

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def _append_trial_to_jsonl(self, trial_result: TrialResult):
        if not self.enable_jsonl_history:
            return

        lock_path = str(self.history_jsonl_path) + ".lock"
        lock = FileLock(lock_path)

        try:
            with lock:
                with open(self.history_jsonl_path, "a") as f:
                    json.dump(trial_result, f, default=numpy_and_config_encoder)
                    f.write("\n")
        except Exception as err:
            self.logger.error(
                f"Failed to append trial result to JSONL history: {err}",
                exc_info=True,
            )

    def _finalize_optimization(
        self, incumbent: Optional[Union[Configuration, list[Configuration], dict]]
    ):
        """
        Common post-optimization logic for all optimizers:

        - Save incumbent to disk
        - Evaluate on held-out test data (unless disabled via
          `evaluate_incumbent=False` / `--no-evaluate-incumbent`)
        - Persist history + plots
        """

        if incumbent and not self.evaluate_incumbent_enabled:
            self.logger.info(
                f"[{self.__class__.__name__}] Skipping incumbent evaluation "
                "(--no-evaluate-incumbent set)."
            )
        elif incumbent:
            # Held-out Test Evaluation
            if isinstance(incumbent, Configuration):
                result: EvaluationResult = self.evaluate_incumbent(incumbent)

                save_incumbent(
                    incumbent=SavedIncumbent(
                        incumbent=incumbent, evaluation_result=result
                    ),
                    output_path=self.output_path,
                )
            elif isinstance(incumbent, list):
                saved_incumbents: list[SavedIncumbent] = []
                has_multiple_incumbents = len(incumbent) > 1
                incumbent_predictions: list[np.ndarray] = []
                heldout_labels: np.ndarray | None = None
                ensemble_evaluation_result: TrainResult | None = None
                for incumbent_idx, incumbent_ in enumerate(incumbent):
                    predictions_filename = (
                        f"predictions_incumbent_{incumbent_idx}.npy"
                        if has_multiple_incumbents
                        else "predictions.npy"
                    )
                    state_dict_filename = (
                        f"model_state_dict_incumbent_{incumbent_idx}.pt"
                        if has_multiple_incumbents
                        else "model_state_dict.pt"
                    )
                    evaluation_result = self.evaluate_incumbent(
                        incumbent_,
                        predictions_filename=predictions_filename,
                        state_dict_filename=state_dict_filename,
                    )
                    result, prediction_result = (
                        evaluation_result["train_result"],
                        evaluation_result["prediction_result"],
                    )
                    if has_multiple_incumbents:
                        incumbent_predictions.append(prediction_result["y_pred"])
                        incumbent_labels = prediction_result["y_true"]
                        if heldout_labels is None:
                            heldout_labels = incumbent_labels
                        elif not np.array_equal(heldout_labels, incumbent_labels):
                            raise ValueError(
                                "Cannot compute ensemble accuracy because incumbent "
                                "evaluations used different held-out labels."
                            )
                    saved_incumbents.append(
                        {
                            "incumbent": incumbent_,
                            "evaluation_result": evaluation_result,
                        }
                    )
                if has_multiple_incumbents:
                    if heldout_labels is None:
                        raise ValueError(
                            "Cannot compute ensemble accuracy without held-out labels."
                        )
                    ensemble_evaluation_result = self._save_ensemble_predictions(
                        incumbent_predictions,
                        heldout_labels,
                    )
                save_incumbent(
                    incumbent=saved_incumbents,
                    output_path=self.output_path,
                    ensemble_evaluation_result=ensemble_evaluation_result,
                )
            else:
                raise ValueError(f"Unknown data type for incumbent ({type(incumbent)})")

        self.save_plot_images()
        self.logger.info(
            f"[{self.__class__.__name__}] "
            "Optimization complete. Best configuration saved and evaluated."
        )

    def save_plot_images(self):
        if not self.history:
            self.logger.debug("No history to plot.")
            return
        self.logger.debug("Saving plot images...")
        save_all_plots(self.history, outdir=self.output_path)

        self.logger.info("Plot images saved.")

    def _save_test_predictions(
        self,
        predictions: np.ndarray,
        filename: str = "predictions.npy",
    ) -> Path:
        """Persist final test predictions in the expected NumPy format."""
        predictions_path = self.output_path / filename
        self.output_path.mkdir(parents=True, exist_ok=True)
        np.save(predictions_path, predictions)
        self.logger.info(
            f"[{self.__class__.__name__}] Saved test predictions to {predictions_path}"
        )
        return predictions_path

    def _save_model_state_dict(
        self,
        approach: Approach,
        filename: str = "model_state_dict.pt",
    ) -> Optional[Path]:
        """Persist the trained incumbent model's state dict."""
        if approach.trainer is None:
            self.logger.warning(
                f"[{self.__class__.__name__}] No trainer available; skipping "
                "model state dict save."
            )
            return None

        state_dict_path = self.output_path / filename
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_state_dict = {
            k: v.cpu() for k, v in approach.trainer.model.state_dict().items()
        }
        atomic_torch_save(model_state_dict, state_dict_path)
        self.logger.info(
            f"[{self.__class__.__name__}] Saved incumbent model state dict to "
            f"{state_dict_path}"
        )
        return state_dict_path

    @staticmethod
    def _majority_vote(labels: np.ndarray):
        classes, counts = np.unique(labels, return_counts=True)
        return classes[int(np.argmax(counts))]

    def _save_ensemble_predictions(
        self,
        incumbent_predictions: list[np.ndarray],
        heldout_labels: np.ndarray,
        filename: str = "predictions.npy",
    ) -> TrainResult:
        """
        Persist a deterministic majority-vote ensemble and compute held-out accuracy.

        Ties are broken by the natural sort order of class labels via np.unique.
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
            self._majority_vote,
            axis=0,
            arr=stacked_predictions,
        )
        self._save_test_predictions(
            ensemble_predictions,
            filename=filename,
        )
        if ensemble_predictions.shape != heldout_labels.shape:
            raise ValueError(
                "Cannot compute ensemble accuracy because predictions and labels "
                f"have different shapes: {ensemble_predictions.shape} vs "
                f"{heldout_labels.shape}."
            )
        ensemble_accuracy = float(np.mean(ensemble_predictions == heldout_labels))
        ensemble_result = TrainResult(val_accuracy=ensemble_accuracy, history=[])
        self.logger.info(
            f"[{self.__class__.__name__}] Saved majority-vote ensemble from "
            f"{len(incumbent_predictions)} incumbents with held-out accuracy "
            f"{ensemble_accuracy:.4f}."
        )
        return ensemble_result

    @staticmethod
    def _config_to_hash_id(config: Configuration) -> str:
        config_payload = json.dumps(dict(config), sort_keys=True, default=str)
        return hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]

    def _trainer_checkpoint_path(self, config_id: str) -> Path:
        return self.trainer_checkpoint_dir / config_id / "trainer.pth"

    def _trainer_load_kwargs(self, checkpoint_path: Path) -> dict[str, Path]:
        """
        Return load kwargs understood by the concrete approaches.

        Some approaches call the trainer checkpoint argument `load_path`, while
        others call it `trainer_load_path`. Concrete approaches accept
        additional kwargs, so passing both keeps this optimizer independent of
        those implementation details.
        """
        if not checkpoint_path.exists():
            return {}

        self.logger.info(
            f"[{self.__class__.__name__}] Loading trainer checkpoint from {checkpoint_path}"
        )
        return {
            "load_path": checkpoint_path,
            "trainer_load_path": checkpoint_path,
        }

    def _save_trainer_checkpoint(
        self, config_id: str, trainer: Optional[Trainer]
    ) -> None:
        if trainer is None:
            self.logger.warning(
                f"[{self.__class__.__name__}] No trainer available to checkpoint "
                f"for config {config_id}."
            )
            return

        checkpoint_path = self._trainer_checkpoint_path(config_id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(checkpoint_path)
        self.logger.debug(
            f"[{self.__class__.__name__}] Saved trainer checkpoint to {checkpoint_path}"
        )

    def train_single_configuration(
        self,
        config: Configuration,
        seed: int,
        budget: float,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
        data_seed: Optional[int] = None,
    ) -> float:
        val_error, _ = self._train_single_configuration_with_history(
            config=config,
            seed=seed,
            budget=budget,
            device=device,
            num_workers=num_workers,
            data_seed=data_seed,
        )
        return val_error

    def _train_single_configuration_with_history(
        self,
        config: Configuration,
        seed: int,
        budget: float,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
        data_seed: Optional[int] = None,
    ) -> tuple[float, list[EpochResult]]:
        from automl.core.utils import timer

        device = device or self.device
        num_workers = (
            self.runtime_config["num_workers"] if num_workers is None else num_workers
        )
        # Defaults to `seed` for callers that evaluate each config once
        # (SMAC/Hyperband, RandomSearch). Callers that resume the same
        # config across multiple calls (e.g. ifBO's freeze-thaw steps) must
        # pass a `data_seed` that stays fixed for that config's lifetime,
        # since the model checkpoint carries over between calls but a
        # changing split would silently move rows between train and val.
        data_seed = seed if data_seed is None else data_seed

        model_type = config["model_type"]
        config_id = self._config_to_hash_id(config)
        config_dict = dict(config)  # avoid mutating original
        config_dict["epochs"] = int(budget)

        train_fraction = 1.0
        max_num_rows = int(self.runtime_config["max_num_rows"])

        val_error = float("nan")
        epoch_history: list[EpochResult] = []
        try:
            with self._state_lock:
                self.trial_no += 1
                trial_no = self.trial_no

            self.logger.info(
                f"[{self.__class__.__name__}] Trial #{trial_no}/"
                f"{self.runtime_config['n_trials']}, "
                f"Config ID: {config_id}, "
                f"Budget: {budget}, "
                f"Train Fraction: {train_fraction:.2f}, "
                f"Seed: {seed}, "
                f"Data Seed: {data_seed}, "
                f"Approach: {model_type}, "
                f"Device: {device}"
            )

            val_size = self.runtime_config["val_size"]
            data_info = self.dataset.create_dataloaders(
                val_size=val_size,
                random_state=data_seed,
                train_fraction=train_fraction,
                max_num_rows=max_num_rows,
            )
            train_df, val_df = data_info["train_df"], data_info["val_df"]

            train_split = DatasetSplit(
                texts=train_df["text"].tolist(),
                labels=train_df["label"].tolist(),
            )
            val_split = DatasetSplit(
                texts=val_df["text"].tolist(),
                labels=val_df["label"].tolist(),
            )

            self.logger.info(
                f"[{self.__class__.__name__}] Data sizes: "
                f"Train={train_split.size}, Val={val_split.size}"
            )

            approach = get_approach(model_type)(
                config_dict,
                data_info["num_classes"],
                device,
                num_workers=num_workers,
                stochastic_epochs=self.runtime_config["stochastic_epochs"],
                stochastic_epoch_fraction=self.runtime_config[
                    "stochastic_epoch_fraction"
                ],
                approach_params=self.runtime_config.get("approach_params"),
            )

            with approach.with_mode("train") as _approach:
                prepared_result = _approach.prepare(train_split, val_split)

                with timer.Timer() as t:
                    checkpoint_lock = self._checkpoint_lock_for(config_id)
                    with checkpoint_lock:
                        trainer_checkpoint_path = self._trainer_checkpoint_path(
                            config_id
                        )
                        load_kwargs = self._trainer_load_kwargs(trainer_checkpoint_path)
                    result = _approach.train(
                        prepared_result,
                        epochs=int(budget),
                        max_time_seconds=self.runtime_config.get(
                            "max_trial_time_seconds"
                        ),
                        **load_kwargs,
                    )
                    with checkpoint_lock:
                        self._save_trainer_checkpoint(config_id, _approach.trainer)

            execution_time = t.execution_time
            val_error = 1.0 - result["val_accuracy"]
            epoch_history = result["history"]

            with self._state_lock:
                is_best_yet = (
                    budget >= self.highest_budget_seen
                    and val_error < self.best_val_error
                )

                # Save best model checkpoint
                if is_best_yet:
                    self.best_val_error = val_error
                    approach.save(self.checkpoint_dir, replace_best=True)
                    self.logger.info(
                        f"[{self.__class__.__name__}] New best model with "
                        f"val acc={result['val_accuracy']:.4f}"
                    )

                self.highest_budget_seen = max(self.highest_budget_seen, budget)

                trial_result: TrialResult = TrialResult(
                    config=dict(config),
                    seed=seed,
                    budget=budget,
                    trialNo=trial_no,
                    execution_time=execution_time or float("nan"),
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S,%f"),
                    val_error=val_error,
                    best_so_far=is_best_yet,
                    epoch_history=result["history"],
                )
                self.history.append(trial_result)

            self._append_trial_to_jsonl(trial_result)

            self.logger.info(
                f"[{self.__class__.__name__}] Trial #{trial_no} "
                f"completed (val_error={val_error:.4f})."
            )

        except KeyboardInterrupt:
            self.logger.warning(
                f"[{self.__class__.__name__}] Optimization interrupted by user."
            )
            raise
        except Exception as err:
            self.logger.error(err, exc_info=True)
            val_error = float("nan")
            epoch_history = []

        return val_error, epoch_history

    def evaluate_incumbent(
        self,
        incumbent: Configuration,
        predictions_filename: str = "predictions.npy",
        state_dict_filename: str = "model_state_dict.pt",
    ) -> EvaluationResult:
        """Same evaluation protocol as SmacOptimizer."""
        epochs = self.runtime_config["evaluation_budget"]

        self.logger.info(
            f"[{self.__class__.__name__}] Retraining incumbent on full train data (epochs={epochs})..."
        )
        model_type: ApproachName = incumbent.get("model_type")  # type: ignore

        data_info = self.dataset.create_dataloaders(
            val_size=0.0,
            random_state=self.runtime_config["seed"],
        )
        train_df, test_df = data_info["train_df"], data_info["test_df"]

        # log size of train and test
        self.logger.info(
            f"[{self.__class__.__name__}] Train size: {len(train_df)}, Test size: {len(test_df)}"
        )

        train_split = DatasetSplit(
            texts=train_df["text"].tolist(),
            labels=train_df["label"].tolist(),
        )
        test_split = DatasetSplit(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
        )

        # should_evaluate = not any(pd.isna(label) for label in test_split.labels)
        should_evaluate = not pd.isna(test_split.labels[0])

        approach: Approach = get_approach(model_type)(
            incumbent,
            data_info["num_classes"],
            self.device,
            num_workers=self.runtime_config["num_workers"],
            stochastic_epochs=self.runtime_config["stochastic_epochs"],
            stochastic_epoch_fraction=self.runtime_config["stochastic_epoch_fraction"],
            approach_params=self.runtime_config.get("approach_params"),
        )

        best_ckpt_path = (
            self.checkpoint_dir / f"{Path(state_dict_filename).stem}_eval_best.pt"
        )

        with approach.with_mode("eval") as _approach:
            prepared = _approach.prepare(train_split, test_split)
            train_result = _approach.train(
                prepared,
                epochs=epochs,
                evaluate_validation=should_evaluate,
                save_path=best_ckpt_path if should_evaluate else None,
            )
            if should_evaluate and best_ckpt_path.exists():
                self.logger.info(
                    f"[{self.__class__.__name__}] Restoring best-epoch checkpoint "
                    f"(val_accuracy={train_result['val_accuracy']:.4f}) before "
                    f"final prediction."
                )
                _approach.trainer.load(best_ckpt_path)
            self.logger.info("Predicting for test set")
            prediction_result = _approach.predict(test_df)

        self.logger.info(
            f"[{self.__class__.__name__}] Final Held-Out Test Accuracy: "
            f"{train_result['val_accuracy']:.4f}"
        )
        self._save_test_predictions(
            prediction_result["y_pred"],
            filename=predictions_filename,
        )
        self._save_model_state_dict(approach, filename=state_dict_filename)

        return EvaluationResult(
            train_result=train_result, prediction_result=prediction_result
        )
