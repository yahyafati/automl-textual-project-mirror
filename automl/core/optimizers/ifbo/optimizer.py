from __future__ import annotations

import gc
import itertools
import math
import os.path
import random
from concurrent.futures import Future, ThreadPoolExecutor
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
      type-aware scheme (float/int/log/categorical).
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

        self.b_max: int = self.max_budget - self.min_budget + 1

        # Build encoder for ConfigSpace -> [0,1]^d
        self.hp_space: HyperparameterSpace = self._build_hp_space(self.space)

        # Probability of exploring by sampling a new candidate.
        self.initial_epsilon: float = 1.0
        if not 0.0 <= self.initial_epsilon <= 1.0:
            raise ValueError(
                "[IfboOptimizer] ifbo_initial_epsilon must be in [0, 1], "
                f"got {self.initial_epsilon}."
            )

        self.ifbo_use_random_selection = runtime_config.get("ifbo_use_random_selection")
        self.greedy_selection = runtime_config.get("ifbo_greedy_candidate_selection")
        self.incumbent_ensemble_top_k: int = runtime_config.get(
            "ifbo_incumbent_ensemble_top_k"
        )
        self._thaw_step: int = runtime_config.get("ifbo_thaw_step", 1)

        if self.incumbent_ensemble_top_k < 1:
            raise ValueError(
                "[IfboOptimizer] ifbo_incumbent_ensemble_top_k must be >= 1, "
                f"got {self.incumbent_ensemble_top_k}."
            )

        self.incumbent_ensemble_accuracy_threshold: float = float(
            runtime_config.get("ifbo_incumbent_ensemble_accuracy_threshold")
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
            "ifbo_use_random_selection=%s, greedy_selection=%s, "
            "incumbent_ensemble_top_k=%d, incumbent_ensemble_accuracy_threshold=%.4f",
            self.min_budget,
            self.max_budget,
            self.b_max,
            self.total_steps,
            self.hp_space.dim,
            self.initial_epsilon,
            self.ifbo_use_random_selection,
            self.greedy_selection,
            self.incumbent_ensemble_top_k,
            self.incumbent_ensemble_accuracy_threshold,
        )

        # Load FT-PFN surrogate model
        self.logger.info("[IfboOptimizer] Loading pretrained FT-PFN surrogate...")
        self.model = FTPFN(version="0.0.1", target_path=".model")

        self._uid_counter = itertools.count()

        self._parallelism: int = max(
            1, int(runtime_config.get("num_parallel_trials", 1))
        )
        self._effective_num_workers: int = (
            0 if self._parallelism > 1 else int(runtime_config["num_workers"])
        )

        if self._parallelism > 1:
            self.logger.info(
                "[IfboOptimizer] Parallel trial execution enabled: "
                "num_parallel_trials=%d, devices=%s, effective_num_workers=%d",
                self._parallelism,
                self.devices,
                self._effective_num_workers,
            )
            self._prewarm_shared_resources(runtime_config)

    def _prewarm_shared_resources(self, runtime_config: RuntimeConfig) -> None:
        """
        Load resources that are cached (but not lock-protected) behind a
        check-then-set the first time they're touched, so the very first
        parallel batch of trials doesn't race to populate that cache and
        redundantly repeat expensive work (parsing the full dataset,
        downloading/loading a pretrained transformer). Note: this does
        NOT cover either approach's tokenizer - that cache is thread-local by
        design (see `text_encoding.load_tokenizer`'s docstring), so warming it
        here would only populate this (main) thread's copy and not help any
        worker thread; each worker loads its own on first use instead.
        """
        self.logger.info("[IfboOptimizer] Prewarming shared caches...")
        self.dataset.load_data()

        if runtime_config["approach"] == "sequence-dl":
            from automl.core.approaches.sequence_dl import (
                SequenceDLApproach,
                _load_pretrained_word_embeddings,
            )

            for model_name in SequenceDLApproach.MODEL_NAME_CHOICES:
                _load_pretrained_word_embeddings(
                    os.path.join(SequenceDLApproach.MODELS_DIR, model_name)
                )

        elif runtime_config["approach"] == "transformer":
            from transformers import AutoModel

            from automl.core.approaches.transformer import TransformerApproach

            for model_name in TransformerApproach.MODEL_NAME_CHOICES:
                AutoModel.from_pretrained(model_name)

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
        data_seed = self._rng.randint(1, 2**31 - 1)
        cand = _IfBOCandidate(
            config=cfg, z=z, uid=next(self._uid_counter), data_seed=data_seed
        )
        return cand

    def _epsilon(self, completed_trials: int) -> float:
        r"""
        Polynomial decay of exploration rate ε from initial_epsilon to eps_min.

        ε(t) = ε_min + (ε_0 - ε_min) * (1 - t / T)^p

        where t = completed_trials and T = total_steps.
        """
        eps_min = 0.05
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

    def _step(
        self,
        cand: _IfBOCandidate,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        prev_budget = (
            self._step_to_budget(cand.steps_done) if cand.steps_done > 0 else 0
        )
        cand.steps_done = min(cand.steps_done + self._thaw_step, self.b_max)
        budget = self._step_to_budget(cand.steps_done)

        with self._state_lock:
            seed = self._rng.randint(1, 2**31 - 1)

        val_error, epoch_history = self._train_single_configuration_with_history(
            config=cand.config,
            seed=seed,
            budget=float(budget),
            device=device,
            num_workers=num_workers,
            data_seed=cand.data_seed,
        )

        new_epochs = [
            e
            for e in epoch_history
            if e["epoch"] > prev_budget and e["epoch"] >= self.min_budget
        ]

        if not new_epochs:
            y = float("nan") if math.isnan(val_error) else 1.0 - float(val_error)
            cand.ts.append(cand.steps_done / self.b_max)
            cand.ys.append(y)
            return

        for e in new_epochs:
            step = e["epoch"] - self.min_budget + 1
            t = step / self.b_max  # normalized time in [0, 1]
            y = e["val_accuracy"]
            cand.ts.append(t)
            cand.ys.append(float(y) if y is not None else float("nan"))

    def _step_on_device(
        self,
        cand: _IfBOCandidate,
        device: torch.device,
        num_workers: int,
    ) -> None:
        """
        Entry point submitted to the parallel-trial thread pool. PyTorch's
        "current CUDA device" is thread-local, not inherited from the
        thread that created the pool, so any implicit current-device op
        inside training (e.g. `torch.cuda.empty_cache()` in
        TorchTrainer.train's cleanup) would silently target device 0
        unless this thread's current device is set explicitly first.
        """
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self._step(cand, device=device, num_workers=num_workers)

    def _best_so_far_accuracy(self) -> float:
        ys: list[float] = []
        for c in self.candidates:
            for y in c.ys:
                if math.isfinite(y):
                    ys.append(y)
        return max(ys) if ys else 0.0

    def _select_next_candidate(
        self,
        context: list[Curve],
        completed_trials: int,
        exclude: Optional[set[int]] = None,
    ) -> _IfBOCandidate:
        if not self.candidates:
            self.logger.debug("No Candidates, sampling a new one.")
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate

        epsilon = self._epsilon(completed_trials)
        self.logger.debug(f"Selected epsilon: {epsilon}")
        if self._rng.random() < epsilon:
            self.logger.debug("Exploration: Sampling a new candidate.")
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate

        excluded_uids = exclude or set()
        pending: list[_IfBOCandidate] = [
            c
            for c in self.candidates
            if c.steps_done < self.b_max and c.uid not in excluded_uids
        ]

        if not pending:
            self.logger.debug(
                "No pending candidates left this round (all maxed out or "
                "excluded) - sampling a new one."
            )
            candidate = self._sample_new_candidate()
            self.candidates.append(candidate)
            return candidate

        # For baselines
        if self.ifbo_use_random_selection:
            return self._rng.choice(pending)

        MAX_LOOKAHEAD = 5
        f_best = self._best_so_far_accuracy()
        h_rand = min(self._rng.randint(1, self.b_max), MAX_LOOKAHEAD)
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

        return selected

    @staticmethod
    def _candidate_best_accuracy(c: _IfBOCandidate) -> float:
        vals = [y for y in c.ys if math.isfinite(y)]
        return max(vals) if vals else float("-inf")

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

        if self._parallelism > 1:
            return self._perform_ifbo_parallel()
        return self._perform_ifbo_sequential()

    def _perform_ifbo_sequential(
        self,
    ) -> Optional[Configuration | list[Configuration]]:
        used_steps = 0

        while used_steps < self.total_steps:
            context = self._build_context()
            next_cand = self._select_next_candidate(context, used_steps + 1)

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
                self._thaw_step,
            )
            self._step(next_cand)
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

    def _perform_ifbo_parallel(
        self,
    ) -> Optional[Configuration | list[Configuration]]:
        used_steps = 0

        with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
            while used_steps < self.total_steps:
                context = self._build_context()
                round_size = min(self._parallelism, self.total_steps - used_steps)

                batch: list[tuple[_IfBOCandidate, int]] = []
                selected_uids: set[int] = set()
                for _ in range(round_size):
                    cand = self._select_next_candidate(
                        context, used_steps + 1, exclude=selected_uids
                    )
                    if cand.steps_done >= self.b_max:
                        break
                    selected_uids.add(cand.uid)
                    batch.append((cand, self._thaw_step))

                del context

                if not batch:
                    self.logger.info(
                        "[IfboOptimizer] All candidates reached max steps. "
                        "Stopping early at used_steps=%d.",
                        used_steps,
                    )
                    break

                self.logger.debug(
                    "[IfboOptimizer] Dispatching round of %d candidate(s): %s",
                    len(batch),
                    [(c.config, s) for c, s in batch],
                )

                futures: list[Future] = [
                    executor.submit(
                        self._step_on_device,
                        cand,
                        self.devices[i % len(self.devices)],
                        self._effective_num_workers,
                    )
                    for i, (cand, steps) in enumerate(batch)
                ]

                try:
                    for fut in futures:
                        try:
                            fut.result()
                        except Exception:
                            self.logger.error(
                                "[IfboOptimizer] Unexpected error in parallel "
                                "ifBO worker.",
                                exc_info=True,
                            )
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

                used_steps += len(batch)

                gc.collect()
                if torch.cuda.is_available():
                    for d in self.devices:
                        if d.type == "cuda":
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
                if torch.mps.is_available():
                    torch.mps.empty_cache()
                # ------------------------------------------------------------

                if used_steps % 25 == 0 or used_steps == self.total_steps:
                    best_acc = self._best_so_far_accuracy()
                    self.logger.info(
                        "[IfboOptimizer] step %4d/%4d | best-so-far accuracy = %.4f",
                        used_steps,
                        self.total_steps,
                        best_acc,
                    )

        return self._select_incumbent()
