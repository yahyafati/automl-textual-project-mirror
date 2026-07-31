from __future__ import annotations

from ConfigSpace import (
    Configuration,
)

from automl.core.optimizers.base_optimizer import Optimizer
from automl.cli import RuntimeConfig


class RandomSearch(Optimizer):
    """
    Simple baseline optimizer that:
      * samples configurations uniformly at random from the ConfigSpace
      * uses fixed budgets between min_budget and max_budget
      * keeps track of the best (incumbent) and evaluates it at the end
    """

    def __init__(self, runtime_config: RuntimeConfig):
        super().__init__(runtime_config)

        if hasattr(self.dataset, "load_base_data"):
            # FIXME: Remove this!
            self.dataset.load_base_data()

        # State
        self._best_config: Configuration | None = None

    # -------------------------
    # Core public API
    # -------------------------

    def run(self):
        """Run random search and evaluate the incumbent."""
        self._perform_random_search()

    # -------------------------
    # Random search loop
    # -------------------------

    def _perform_random_search(self):
        n_trials: int = self.runtime_config["n_trials"]
        min_budget: float = self.runtime_config["min_budget"]
        max_budget: float = self.runtime_config["max_budget"]

        self.logger.info(
            f"[RandomOptimizer] Starting random search with "
            f"n_trials={n_trials}, min_budget={min_budget}, max_budget={max_budget}"
        )

        try:
            for i in range(n_trials):
                # Uniform random configuration from ConfigSpace
                config: Configuration = self.space.sample_configuration()
                self.logger.info(
                    f"[RandomOptimizer] Trial {i + 1}: Sampled configuration: {config}"
                )

                # Here we simply use max_budget every time.
                # If you want varying budgets, you could sample uniformly between min and max.
                budget = max_budget

                # Seed per trial (optional: use global seed + i)
                seed = self.runtime_config["seed"] + i

                val_error = self._train_fn(config=config, seed=seed, budget=budget)

                if val_error < self.best_val_error:
                    self.best_val_error = val_error
                    self._best_config = config

            # After the loop, save incumbent and evaluate
            if self._best_config is not None:
                self.evaluate_incumbent(self._best_config)

        except KeyboardInterrupt:
            self.logger.error(
                "[RandomOptimizer] Interrupted by user. Saving what we have."
            )
        finally:
            self._finalize_optimization(self._best_config)

    # -------------------------
    # Train function
    # -------------------------

    def _train_fn(self, config: Configuration, seed: int, budget: float) -> float:
        """Training function for a single random configuration."""
        return self.train_single_configuration(config=config, seed=seed, budget=budget)
