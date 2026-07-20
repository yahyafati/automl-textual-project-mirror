import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

from ConfigSpace import Configuration, ConfigurationSpace
from filelock import FileLock

from automl.core.trainers.base_trainer import Trainer
from automl.core import configspacehelper
from automl.core.approaches.base_approach import Approach
from automl.core.datasets import get_dataset_class
from automl.core.plot_history import (
    plot_learning_curves,
    plot_optimization_history,
    plot_budget_vs_performance,
    plot_epoch_heatmap,
)
from automl.core.registry import get_approach, register_all_approaches
from automl.core.types import DatasetSplit, TrialResult, ApproachName
from automl.core.utils.misc import SavedIncumbent
from automl.core.utils.misc import (
    get_device,
    numpy_and_config_encoder,
    set_seed,
    save_incumbent,
)
from automl.cli import RuntimeConfig
from automl.logger import get_logger
from automl.trial_plots import save_all_plots


class Optimizer(ABC):

    def __init__(self, runtime_config: RuntimeConfig):
        self.runtime_config = runtime_config
        self.output_path = runtime_config["output_path"]
        self.logger = get_logger()

        self.logger.info(f"Selected seed: {runtime_config['seed']}")
        self.logger.info(f"Current runtime_id: {runtime_config['runtime_id']}")

        self.device = get_device()
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

        self.highest_budget_seen: float = 0.0
        self.best_val_error: float = float("inf")
        self.checkpoint_dir = self.output_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.trial_no: int = 0

        self.trainer_checkpoint_dir = self.checkpoint_dir / "trainers"
        self.trainer_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        register_all_approaches()
        set_seed(runtime_config["seed"])

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
        - Evaluate on held-out test data
        - Persist history + plots
        """

        if incumbent:
            # Held-out Test Evaluation
            if isinstance(incumbent, Configuration):
                result = self.evaluate_incumbent(incumbent)

                save_incumbent(
                    incumbent=SavedIncumbent(
                        incumbent=incumbent, evaluation_result=result
                    ),
                    output_path=self.output_path,
                )
            elif isinstance(incumbent, list):
                saved_incumbents: list[SavedIncumbent] = []
                for incumbent_ in incumbent:
                    result = self.evaluate_incumbent(incumbent_)
                    saved_incumbents.append(
                        {
                            "incumbent": incumbent_,
                            "evaluation_result": result,
                        }
                    )
                save_incumbent(
                    incumbent=saved_incumbents,
                    output_path=self.output_path,
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
    ) -> float:
        """
        Train a single configuration for a given budget.

        Returns
        -------
        val_error : float
            1 - validation accuracy.
        """
        from automl.core.utils import timer

        model_type = config["model_type"]
        config_id = self._config_to_hash_id(config)
        config_dict = dict(config)  # avoid mutating original
        config_dict["epochs"] = int(budget)
        self.trial_no += 1

        train_fraction = 1.0
        max_num_rows = int(self.runtime_config["max_num_rows"])

        try:
            set_seed(seed)
            self.logger.info(
                f"[{self.__class__.__name__}] Trial #{self.trial_no}/"
                f"{self.runtime_config['n_trials']}, "
                f"Config ID: {config_id}, "
                f"Budget: {budget}, "
                f"Train Fraction: {train_fraction:.2f}, "
                f"Seed: {seed}, "
                f"Approach: {model_type}"
            )

            val_size = 0.2
            data_info = self.dataset.create_dataloaders(
                val_size=val_size,
                random_state=seed,
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
                self.device,
                num_workers=self.runtime_config["num_workers"],
            )

            with timer.Timer() as t:
                with approach.with_mode("train") as _approach:
                    prepared_result = _approach.prepare(train_split, val_split)
                    trainer_checkpoint_path = self._trainer_checkpoint_path(config_id)
                    result = _approach.train(
                        prepared_result,
                        epochs=int(budget),
                        **self._trainer_load_kwargs(trainer_checkpoint_path),
                    )
                    self._save_trainer_checkpoint(config_id, _approach.trainer)
            execution_time = t.execution_time
            val_error = 1.0 - result["val_accuracy"]

            # TODO: Maybe we don't need this
            # if budget > self.highest_budget_seen:
            #     self.highest_budget_seen = budget
            #     self.best_val_error = float("inf")

            is_best_yet = (
                budget >= self.highest_budget_seen and val_error < self.best_val_error
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
                trialNo=self.trial_no,
                execution_time=execution_time or float("nan"),
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S,%f"),
                val_error=val_error,
                best_so_far=is_best_yet,
                epoch_history=result["history"],
            )
            self.history.append(trial_result)
            self._append_trial_to_jsonl(trial_result)

            self.logger.info(
                f"[{self.__class__.__name__}] Trial #{self.trial_no} "
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

        return val_error

    def evaluate_incumbent(self, incumbent: Configuration):
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

        approach: Approach = get_approach(model_type)(
            incumbent,
            data_info["num_classes"],
            self.device,
            num_workers=self.runtime_config["num_workers"],
        )

        with approach.with_mode("eval") as _approach:
            prepared = _approach.prepare(train_split, test_split)
            train_result = _approach.train(prepared, epochs=epochs)

        self.logger.info(
            f"[{self.__class__.__name__}] Final Held-Out Test Accuracy: "
            f"{train_result['val_accuracy']:.4f}"
        )

        return train_result
