"""
In-Context Freeze-Thaw Bayesian Optimization (ifBO) optimizer.

- Samples a fixed set of random candidate configurations from ConfigSpace.
- Uses the FT-PFN surrogate (ifbo.FTPFN) to schedule which configuration
  to "thaw" next and at which (future) horizon, following MFPI-random.
- Each step corresponds to one call to `train_single_configuration`.
- Hyperparameters are encoded into [0,1]^d for FT-PFN using a simple
  type-aware scheme (float/int/log/categorical), similar to ifbo_impl.py.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from ifbo import Curve
from ifbo.surrogate import FTPFN

from automl.core.optimizers.base_optimizer import Optimizer
from automl.cli import RuntimeConfig
from .candidate import IfBOCandidate as _IfBOCandidate
from .hp_space import Categorical, Float, HPSpec, HyperparameterSpace, Integer


class IfboOptimizer(Optimizer):
    """
    In-Context Freeze-Thaw Bayesian Optimization (ifBO) optimizer.

    - Samples a fixed set of random candidate configurations from ConfigSpace.
    - Uses the FT-PFN surrogate (ifbo.FTPFN) to schedule which configuration
      to "thaw" next and at which (future) horizon, following MFPI-random.
    - Each step corresponds to one call to `train_single_configuration`.
    - Hyperparameters are encoded into [0,1]^d for FT-PFN using a simple
      type-aware scheme (float/int/log/categorical), similar to ifbo_impl.py.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        super().__init__(runtime_config)

        # Optional: load base data once, if supported by the dataset
        if hasattr(self.dataset, "load_base_data"):
            self.dataset.load_base_data()

        self._rng = random.Random(runtime_config["seed"])

        self.min_budget: int = int(runtime_config["min_budget"])
        self.max_budget: int = int(runtime_config["max_budget"])
        if self.max_budget <= self.min_budget:
            raise ValueError(
                f"[IfboOptimizer] max_budget ({self.max_budget}) must be > "
                f"min_budget ({self.min_budget}) for freeze-thaw."
            )

        # Number of discrete freeze-thaw steps per configuration
        # step 1 -> budget = min_budget
        # step b_max -> budget = max_budget
        self.b_max: int = self.max_budget - self.min_budget + 1

        # Build encoder for ConfigSpace -> [0,1]^d
        self.hp_space: HyperparameterSpace = self._build_hp_space(self.space)

        # How many random candidates to sample initially
        self.n_candidates: int = int(runtime_config.get("ifbo_n_candidates", 40))

        # Total iFBO "steps" (each is one call to train_single_configuration)
        requested_steps: int = int(runtime_config["n_trials"])
        max_possible_steps: int = self.n_candidates * self.b_max
        if requested_steps > max_possible_steps:
            self.logger.warning(
                "[IfboOptimizer] Requested n_trials=%d exceeds the maximum number "
                "of distinct freeze-thaw steps (%d candidates x %d steps=%d). "
                "Clipping to %d.",
                requested_steps,
                self.n_candidates,
                self.b_max,
                max_possible_steps,
                max_possible_steps,
            )
            requested_steps = max_possible_steps
        self.total_steps: int = max(1, requested_steps)

        # Sample the fixed set of candidate configurations
        self.candidates: list[_IfBOCandidate] = []
        for _ in range(self.n_candidates):
            cfg: Configuration = self.space.sample_configuration()
            z = self.hp_space.encode(dict(cfg))
            self.candidates.append(_IfBOCandidate(config=cfg, z=z))

        self.logger.info(
            "[IfboOptimizer] Initialized with %d candidates, budgets in [%d, %d], "
            "b_max=%d, total_steps=%d, hp_dim=%d",
            self.n_candidates,
            self.min_budget,
            self.max_budget,
            self.b_max,
            self.total_steps,
            self.hp_space.dim,
        )

        # Load FT-PFN surrogate model
        self.logger.info("[IfboOptimizer] Loading pretrained FT-PFN surrogate...")
        self.model = FTPFN(version="0.0.1")

    # -------------------------
    # Public API
    # -------------------------

    def run(self):
        """
        Run ifBO optimization and evaluate the final incumbent.
        """
        if self.total_steps <= 0:
            self.logger.warning(
                "[IfboOptimizer] total_steps <= 0, nothing to optimize."
            )
            self._finalize_optimization(None)
            return

        try:
            incumbent = self._perform_ifbo()
        except KeyboardInterrupt:
            self.logger.error(
                "[IfboOptimizer] Optimization interrupted by user. "
                "Saving existing data."
            )
            incumbent = self._select_incumbent()  # best seen so far
        self._finalize_optimization(incumbent)

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _build_hp_space(cs: ConfigurationSpace) -> HyperparameterSpace:
        """
        Build a HyperparameterSpace encoder from a ConfigSpace.
        Constant hyperparameters are ignored (do not contribute a dimension).
        """
        specs: dict[str, HPSpec] = {}

        for hp in cs.get_hyperparameters():
            name = hp.name

            # Skip constants; they don't add information to FT-PFN
            if isinstance(hp, Constant):
                continue

            if isinstance(hp, UniformFloatHyperparameter):
                specs[name] = Float(
                    low=float(hp.lower),
                    high=float(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            elif isinstance(hp, UniformIntegerHyperparameter):
                specs[name] = Integer(
                    low=int(hp.lower),
                    high=int(hp.upper),
                    log=bool(getattr(hp, "log", False)),
                )
            elif isinstance(hp, CategoricalHyperparameter):
                specs[name] = Categorical(tuple(hp.choices))
            else:
                raise ValueError(
                    f"[IfboOptimizer] Unsupported hyperparameter type for ifBO "
                    f"encoding: {hp} (type={type(hp)})"
                )

        return HyperparameterSpace(**specs)

    def _observed_candidates(self) -> list[_IfBOCandidate]:
        return [c for c in self.candidates if c.steps_done > 0]

    def _build_context(self) -> list[Curve]:
        """
        Build the context curves for FT-PFN: one Curve per candidate that has
        already been evaluated at least once.
        """
        ctx: list[Curve] = []
        for c in self._observed_candidates():
            t_tensor = torch.tensor(c.ts, dtype=torch.float32)
            y_tensor = torch.tensor(c.ys, dtype=torch.float32)
            ctx.append(
                Curve(
                    hyperparameters=c.z,
                    t=t_tensor,
                    y=y_tensor,
                )
            )
        return ctx

    def _step_to_budget(self, step: int) -> int:
        """
        Map freeze-thaw step index (1..b_max) to actual training budget (epochs).
        step = 1 -> min_budget
        step = b_max -> max_budget
        """
        if not (1 <= step <= self.b_max):
            raise ValueError(
                f"[IfboOptimizer] step_to_budget called with invalid step={step}, "
                f"b_max={self.b_max}"
            )
        return self.min_budget + step - 1

    def _step(self, cand: _IfBOCandidate, step: int = 1) -> None:
        """
        Thaw `cand` for exactly one more freeze-thaw step:
        - Increase its step counter
        - Train for the corresponding epoch budget
        - Record normalized time t and performance y (accuracy) for FT-PFN
        """
        cand.steps_done = min(cand.steps_done + step, self.b_max)
        budget = self._step_to_budget(cand.steps_done)

        # Derive a seed for this evaluation (for reproducibility yet variability)
        seed = self._rng.randint(1, 2**31 - 1)

        # train_single_configuration returns val_error = 1 - val_accuracy
        val_error = self.train_single_configuration(
            config=cand.config, seed=seed, budget=float(budget)
        )
        # In case of failure val_error may be NaN
        if math.isnan(val_error):
            y = float("nan")
        else:
            y = 1.0 - float(val_error)  # convert to accuracy in [0,1]

        # Normalized time t in [0,1], as in the synthetic ifbo_impl: step / b_max
        t = cand.steps_done / self.b_max

        cand.ts.append(t)
        cand.ys.append(y)

    def _best_so_far_accuracy(self) -> float:
        ys: list[float] = []
        for c in self.candidates:
            for y in c.ys:
                if math.isfinite(y):
                    ys.append(y)
        return max(ys) if ys else 0.0

    def _select_next_candidate(
        self, context: list[Curve]
    ) -> tuple[_IfBOCandidate, int]:
        """
        MFPI-random acquisition, adapted from ifbo_impl.py:

        - Sample a random future horizon h_rand in {1, ..., b_max}
        - Sample a random target T_rand above current best accuracy
        - For each candidate, query FT-PFN at time t' = (steps_done + h_rand)/b_max
        - Pick the candidate with largest PI(T_rand).
        - Then actually advance that candidate by *one* step (not h_rand).
        """
        f_best = self._best_so_far_accuracy()

        h_rand = self._rng.randint(1, self.b_max)
        tau_rand = 10 ** self._rng.uniform(-4, -1)  # same scale as in ifbo_impl
        T_rand = f_best + tau_rand * (1.0 - f_best)

        pending: list[_IfBOCandidate] = [
            c for c in self.candidates if c.steps_done < self.b_max
        ]
        if not pending:
            # All candidates fully trained; nothing left to do.
            # Fallback: return the current best candidate (won't be stepped further).
            self.logger.warning(
                "[IfboOptimizer] No pending candidates (all reached max steps)."
            )
            return self._select_incumbent_candidate(), 1

        query: list[Curve] = []
        for c in pending:
            t_query = min(c.steps_done + h_rand, self.b_max) / self.b_max
            query.append(
                Curve(
                    hyperparameters=c.z,
                    t=torch.tensor([t_query], dtype=torch.float32),
                )
            )

        predictions = self.model.predict(context=context, query=query)
        # Each PredictionResult has .pi(threshold) -> probability of improvement
        T_tensor = torch.tensor(T_rand, dtype=torch.float32)
        pi_scores = torch.stack([pred.pi(T_tensor).squeeze() for pred in predictions])

        # TODO: We can sample here instead of doing it greedily
        best_idx = int(torch.argmax(pi_scores))
        return pending[best_idx], h_rand

    def _select_incumbent_candidate(self) -> _IfBOCandidate:
        """
        Select the best candidate observed so far, based on maximum accuracy
        across all evaluated budgets.
        """

        def cand_best_y(c: _IfBOCandidate) -> float:
            vals = [y for y in c.ys if math.isfinite(y)]
            return max(vals) if vals else float("-inf")

        return max(self.candidates, key=cand_best_y)

    def _select_incumbent(self) -> Optional[Configuration]:
        if not any(c.ys for c in self.candidates):
            return None
        return self._select_incumbent_candidate().config

    # -------------------------
    # Main ifBO loop
    # -------------------------

    def _perform_ifbo(self) -> Optional[Configuration]:
        """
        Core freeze-thaw loop (Algorithm 1 + MFPI-random acquisition),
        adapted to call `train_single_configuration`.
        """
        self.logger.info(
            "[IfboOptimizer] Starting ifBO optimization with total_steps=%d.",
            self.total_steps,
        )

        # Initial random sample: pick one candidate, evaluate for one step.
        first = self._rng.choice(self.candidates)
        self._step(first)
        used_steps = 1

        while used_steps < self.total_steps:
            context = self._build_context()
            next_cand, h_rand = self._select_next_candidate(context)
            # If all have reached max steps, _select_next_candidate returns
            # the current best; don't advance further.
            if next_cand.steps_done >= self.b_max:
                self.logger.info(
                    "[IfboOptimizer] All candidates reached max steps. "
                    "Stopping early at used_steps=%d.",
                    used_steps,
                )
                break

            self._step(next_cand, h_rand)
            used_steps += 1

            if used_steps % 25 == 0 or used_steps == self.total_steps:
                best_acc = self._best_so_far_accuracy()
                self.logger.info(
                    "[IfboOptimizer] step %4d/%4d | best-so-far accuracy = %.4f",
                    used_steps,
                    self.total_steps,
                    best_acc,
                )

        best_candidate = self._select_incumbent_candidate()
        best_acc = max(y for y in best_candidate.ys if math.isfinite(y))
        self.logger.info(
            "[IfboOptimizer] Best configuration found with max val accuracy = %.4f",
            best_acc,
        )
        return best_candidate.config
