"""
In-Context Freeze-Thaw Bayesian Optimization (ifBO) optimizer.

- Dynamically samples candidate configurations from ConfigSpace.
- Uses the FT-PFN surrogate (ifbo.FTPFN) to schedule which configuration
  to "thaw" next and at which (future) horizon, following MFPI-random.
- Each step corresponds to one call to `train_single_configuration`.
- Hyperparameters are encoded into [0,1]^d for FT-PFN using a simple
  type-aware scheme (float/int/log/categorical), similar to ifbo_impl.py.
"""

from __future__ import annotations

import gc
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

    - Dynamically samples candidate configurations from ConfigSpace.
    - Uses the FT-PFN surrogate (ifbo.FTPFN) to schedule which configuration
      to "thaw" next and at which (future) horizon, following MFPI-random.
    - Each step corresponds to one call to `train_single_configuration`.
    - Hyperparameters are encoded into [0,1]^d for FT-PFN using a simple
      type-aware scheme (float/int/log/categorical), similar to ifbo_impl.py.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        super().__init__(runtime_config)
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

        # Probability of exploring by sampling a new candidate.
        # The actual epsilon decays with the number of completed trials.
        self.initial_epsilon: float = float(
            runtime_config.get("ifbo_initial_epsilon", 1.0)
        )
        if not 0.0 <= self.initial_epsilon <= 1.0:
            raise ValueError(
                "[IfboOptimizer] ifbo_initial_epsilon must be in [0, 1], "
                f"got {self.initial_epsilon}."
            )

        self.use_random_selection = runtime_config.get("use_random_selection")
        self.greedy_selection = runtime_config.get("ifbo_greedy_candidate_selection")
        self.incumbent_ensemble_top_k: int = runtime_config.get(
            "ifbo_incumbent_ensemble_top_k"
        )

        if self.incumbent_ensemble_top_k < 1:
            raise ValueError(
                "[IfboOptimizer] ifbo_incumbent_ensemble_top_k must be >= 1, "
                f"got {self.incumbent_ensemble_top_k}."
            )

        self.incumbent_ensemble_accuracy_threshold: float = float(
            runtime_config.get("ifbo_incumbent_ensemble_accuracy_threshold", 0.01)
        )
        if not 0.0 <= self.incumbent_ensemble_accuracy_threshold <= 1.0:
            raise ValueError(
                "[IfboOptimizer] ifbo_incumbent_ensemble_accuracy_threshold must be "
                f"in [0, 1], got {self.incumbent_ensemble_accuracy_threshold}."
            )

        # Total iFBO "steps" (each is one call to train_single_configuration)
        requested_steps: int = int(runtime_config["n_trials"])
        self.total_steps: int = max(1, requested_steps)

        # Candidate pool starts empty and grows dynamically via epsilon exploration.
        self.candidates: list[_IfBOCandidate] = []

        self.logger.info(
            "[IfboOptimizer] Initialized with dynamic candidates, budgets in [%d, %d], "
            "b_max=%d, total_steps=%d, hp_dim=%d, initial_epsilon=%.4f, "
            "use_random_selection=%s, greedy_selection=%s, "
            "incumbent_ensemble_top_k=%d, incumbent_ensemble_accuracy_threshold=%.4f",
            self.min_budget,
            self.max_budget,
            self.b_max,
            self.total_steps,
            self.hp_space.dim,
            self.initial_epsilon,
            self.use_random_selection,
            self.greedy_selection,
            self.incumbent_ensemble_top_k,
            self.incumbent_ensemble_accuracy_threshold,
        )

        # Load FT-PFN surrogate model
        self.logger.info("[IfboOptimizer] Loading pretrained FT-PFN surrogate...")
        self.model = FTPFN(version="0.0.1", target_path=".model")

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

    def _sample_new_candidate(self) -> _IfBOCandidate:
        cfg: Configuration = self.space.sample_configuration()
        z = self.hp_space.encode(dict(cfg))
        cand = _IfBOCandidate(config=cfg, z=z)
        return cand

    def _epsilon(self, completed_trials: int) -> float:
        r"""
        Decaying exploration probability.

        completed_trials is the number of already executed calls to
        train_single_configuration. With the default initial_epsilon=1.0 this
        yields 1.0, 0.5, 0.333..., ... for completed_trials 0, 1, 2, ...

        Polynomial Decay:
        e_t = e_min + (e_0 - e_min) * (1 - t/T)^p
        """
        eps_min = 0.1
        p = 2.0
        frac = completed_trials / self.total_steps
        return eps_min + (self.initial_epsilon - eps_min) * (1 - frac) ** p

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
        Thaw `cand` for one or more freeze-thaw steps:
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
        self, context: list[Curve], completed_trials: int
    ) -> tuple[_IfBOCandidate, int]:
        """
        Dynamic epsilon-greedy MFPI-random acquisition:

        - If the candidate list is empty, sample a new candidate
        - Otherwise sample a new candidate with probability epsilon
        - With probability 1 - epsilon, sample among pending existing candidates
          using PI(T_rand) as the sampling weights
        - Sample a random future horizon h_rand in {1, ..., b_max}
        - Sample a random target T_rand above current best accuracy
        - For each pending existing candidate, query FT-PFN at time
          t' = (steps_done + h_rand)/b_max
        - Then advance the selected candidate by h_rand freeze-thaw steps.
        """
        if not self.candidates:
            self.logger.debug("No Candidates, sampling a new one.")
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate, 1

        epsilon = self._epsilon(completed_trials)
        self.logger.debug(f"Selected epsilon: {epsilon}")
        if self._rng.random() < epsilon:
            self.logger.debug("Exploration: Sampling a new candidate.")
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate, 1

        pending: list[_IfBOCandidate] = [
            c for c in self.candidates if c.steps_done < self.b_max
        ]
        if not pending:
            self.logger.debug("No pending candidates, sampling a new one.")
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate, 1

        # For baselines
        if self.use_random_selection:
            candidate = self._rng.choice(pending)
            return candidate, 1

        MAX_LOOKAHEAD = 3  # To prevent it from running to max budget
        f_best = self._best_so_far_accuracy()
        h_rand = self._rng.randint(1, MAX_LOOKAHEAD)
        tau_rand = 10 ** self._rng.uniform(-4, -1)
        T_rand = f_best + tau_rand * (1.0 - f_best)

        query: list[Curve] = []
        for c in pending:
            t_query = min(c.steps_done + h_rand, self.b_max) / self.b_max
            query.append(
                Curve(
                    hyperparameters=c.z,
                    t=torch.tensor([t_query], dtype=torch.float32),
                )
            )

        # Wrap predictions in no_grad to drastically reduce memory usage
        with torch.no_grad():
            predictions = self.model.predict(context=context, query=query)
            T_tensor = torch.tensor(T_rand, dtype=torch.float32)
            pi_scores = torch.stack(
                [pred.pi(T_tensor).squeeze() for pred in predictions]
            )
            pi_scores = torch.nan_to_num(
                pi_scores.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            pi_scores = torch.clamp(pi_scores, min=0.0)

        if self.greedy_selection:
            idx = torch.argmax(pi_scores)
            selected = pending[idx]
        else:
            weights = torch.softmax(pi_scores, dim=0).tolist()
            selected = self._rng.choices(pending, weights=weights, k=1)[0]

        # Free inference-related variables right away
        del query, predictions, T_tensor

        return selected, h_rand

    @staticmethod
    def _candidate_best_accuracy(c: _IfBOCandidate) -> float:
        vals = [y for y in c.ys if math.isfinite(y)]
        return max(vals) if vals else float("-inf")

    def _select_incumbent_candidate(self) -> _IfBOCandidate:
        """
        Select the best candidate observed so far, based on maximum accuracy
        across all evaluated budgets.
        """
        return max(self.candidates, key=self._candidate_best_accuracy)

    def _select_incumbent_candidates(self) -> list[_IfBOCandidate]:
        """
        Select up to top-k evaluated incumbents whose best observed validation
        accuracy is within the configured tolerance of the best incumbent.
        """
        evaluated = [
            c
            for c in self.candidates
            if math.isfinite(self._candidate_best_accuracy(c))
        ]
        if not evaluated:
            return []

        ranked = sorted(
            evaluated,
            key=self._candidate_best_accuracy,
            reverse=True,
        )
        best_accuracy = self._candidate_best_accuracy(ranked[0])
        threshold = self.incumbent_ensemble_accuracy_threshold
        return [
            c
            for c in ranked
            if best_accuracy - self._candidate_best_accuracy(c) <= threshold
        ][: self.incumbent_ensemble_top_k]

    def _select_incumbent(self) -> Optional[Configuration | list[Configuration]]:
        if not any(c.ys for c in self.candidates):
            return None
        incumbent_candidates = self._select_incumbent_candidates()
        if not incumbent_candidates:
            return None

        best_accuracy = self._candidate_best_accuracy(incumbent_candidates[0])
        self.logger.info(
            "[IfboOptimizer] Selected %d incumbent(s) for final evaluation "
            "(best val accuracy=%.4f, top_k=%d, accuracy_threshold=%.4f).",
            len(incumbent_candidates),
            best_accuracy,
            self.incumbent_ensemble_top_k,
            self.incumbent_ensemble_accuracy_threshold,
        )
        if len(incumbent_candidates) == 1:
            return incumbent_candidates[0].config
        return [c.config for c in incumbent_candidates]

    # -------------------------
    # Main ifBO loop
    # -------------------------

    def _perform_ifbo(self) -> Optional[Configuration | list[Configuration]]:
        """
        Core freeze-thaw loop (Algorithm 1 + MFPI-random acquisition),
        adapted to call `train_single_configuration`.
        """
        self.logger.info(
            "[IfboOptimizer] Starting ifBO optimization with total_steps=%d.",
            self.total_steps,
        )

        used_steps = 0

        while used_steps < self.total_steps:
            context = self._build_context()
            next_cand, steps = self._select_next_candidate(context, used_steps + 1)

            # Delete context early since we no longer need it for this step
            del context

            if next_cand.steps_done >= self.b_max:
                self.logger.info(
                    "[IfboOptimizer] All candidates reached max steps. "
                    "Stopping early at used_steps=%d.",
                    used_steps,
                )
                break
            self.logger.debug(
                "[IfboOptimizer] Selected candidate: %s for steps: %d",
                next_cand.config,
                steps,
            )
            self._step(next_cand, steps)
            used_steps += 1

            # --- MEMORY CLEANUP: Clear resources tied up by the training step ---
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.mps.is_available():
                torch.mps.empty_cache()
            # --------------------------------------------------------------------

            if used_steps % 25 == 0 or used_steps == self.total_steps:
                best_acc = self._best_so_far_accuracy()
                self.logger.info(
                    "[IfboOptimizer] step %4d/%4d | best-so-far accuracy = %.4f",
                    used_steps,
                    self.total_steps,
                    best_acc,
                )

        return self._select_incumbent()
