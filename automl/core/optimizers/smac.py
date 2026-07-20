from __future__ import annotations

from pathlib import Path
from typing import Optional

from ConfigSpace import Configuration

from automl.core.optimizers.base_optimizer import Optimizer
from automl.cli import RuntimeConfigDict


class SmacOptimizer(Optimizer):
    """Encapsulates SMAC-based HPO and evaluation logic."""

    def __init__(self, runtime_config: RuntimeConfigDict):
        super().__init__(runtime_config)

        # Optional: load base data once, if supported by the dataset
        if hasattr(self.dataset, "load_base_data"):
            self.dataset.load_base_data()

    # -------------------------
    # Core public API
    # -------------------------

    def run(self):
        """Run SMAC optimization and evaluate the incumbent(s)."""
        self._perform_smac()

    # -------------------------
    # SMAC optimization
    # -------------------------

    def _perform_smac(self):
        from smac import MultiFidelityFacade, Scenario
        from smac.intensifier.hyperband import Hyperband

        scenario_name = "smac3_" + self.runtime_config["runtime_id"]
        scenario = Scenario(
            self.space,
            seed=self.runtime_config["seed"],
            deterministic=False,
            n_trials=self.n_trials,
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            name=scenario_name,
            output_directory=self.output_path,
        )
        self.logger.info(
            f"Scenario ({scenario_name}) initialized with: "
            f"n_trials: {self.runtime_config['n_trials']}, "
            f"min_budget: {self.runtime_config['min_budget']}, "
            f"max_budget: {self.runtime_config['max_budget']}",
        )

        intensifier = Hyperband(scenario, eta=3)
        # intensifier = IfboIntensifier(scenario=scenario)
        smac = MultiFidelityFacade(
            scenario,
            self._train_fn,
            intensifier=intensifier,
            overwrite=True,
            logging_level=Path("logging.yaml"),
        )

        incumbent: Optional[Configuration | list[Configuration]] = None
        try:
            incumbent = smac.optimize()
        except KeyboardInterrupt:
            self.logger.error("Optimization Interrupted by user. Saving existing data")
        finally:
            self._finalize_optimization(incumbent)

    # -------------------------
    # Train function callback
    # -------------------------

    def _train_fn(self, config: Configuration, seed: int, budget: float) -> float:
        """Training function used by SMAC."""
        return self.train_single_configuration(config=config, seed=seed, budget=budget)
