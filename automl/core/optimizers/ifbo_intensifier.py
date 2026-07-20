import math
import random
from typing import Iterator

import torch
from ConfigSpace import Configuration
from ifbo import Curve
from ifbo.surrogate import FTPFN
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo
from smac.scenario import Scenario

from .ifbo_opt import (
    HyperparameterSpace,
    _IfBOCandidate,
    HPSpec,
    Float,
    Integer,
    Categorical,
)  # reuse your classes


class IfboIntensifier(AbstractIntensifier):
    """
    SMAC intensifier implementing the ifBO freeze-thaw scheduling policy.

    - Uses budgets (epochs) from Scenario.min_budget / max_budget.
    - Maintains a fixed pool of random candidates.
    - Uses FT-PFN to decide which candidate to thaw next.
    """

    def __init__(
        self,
        scenario: Scenario,
        n_candidates: int = 40,
        total_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            scenario=scenario,
            n_seeds=None,  # we don't use seeds
            max_config_calls=None,
            max_incumbents=10,
            seed=seed,
        )

        if scenario.min_budget is None or scenario.max_budget is None:
            raise ValueError(
                "IfboIntensifier requires min_budget and max_budget in the Scenario."
            )

        self._min_budget = int(scenario.min_budget)
        self._max_budget = int(scenario.max_budget)
        if self._max_budget <= self._min_budget:
            raise ValueError(
                f"[IfboIntensifier] max_budget ({self._max_budget}) must be > "
                f"min_budget ({self._min_budget})."
            )

        self._b_max = self._max_budget - self._min_budget + 1
        self._n_candidates = int(n_candidates)
        self._rng_py = random.Random(self._seed)

        # sample pool of candidates from the configspace
        cs = self._scenario.configspace
        # build encoder for FT-PFN
        self.hp_space: HyperparameterSpace = self._build_hp_space(cs)

        self._candidates: list[_IfBOCandidate] = []
        for _ in range(self._n_candidates):
            cfg: Configuration = cs.sample_configuration()
            z = self.hp_space.encode(dict(cfg))
            self._candidates.append(_IfBOCandidate(config=cfg, z=z))

        # total freeze-thaw steps (each step = one TrialInfo)
        max_possible = self._n_candidates * self._b_max
        if total_steps is None:
            total_steps = max_possible
        self._total_steps = min(int(total_steps), max_possible)

        # FT-PFN surrogate
        self.model = FTPFN(version="0.0.1")

        # internal counter of how many steps we have *scheduled*
        self._used_steps = 0

    # ---- required AbstractIntensifier flags ----

    @property
    def uses_seeds(self) -> bool:
        return False

    @property
    def uses_budgets(self) -> bool:
        return True

    @property
    def uses_instances(self) -> bool:
        return False

    @property
    def uses_cutoffs(self) -> bool:
        return False

    # ---- helper: encode ConfigSpace -> HyperparameterSpace ----

    def _build_hp_space(self, cs) -> HyperparameterSpace:
        # literally your IfboOptimizer._build_hp_space, just using scenario.configspace
        from ConfigSpace.hyperparameters import (
            CategoricalHyperparameter,
            UniformFloatHyperparameter,
            UniformIntegerHyperparameter,
            Constant,
        )

        specs: dict[str, HPSpec] = {}
        for hp in cs.get_hyperparameters():
            name = hp.name
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
                    f"[IfboIntensifier] Unsupported hyperparameter type: {hp}"
                )
        return HyperparameterSpace(**specs)

    # ---- mapping RunHistory -> candidate curves ----

    def _update_candidate_from_runhistory(self, cand: _IfBOCandidate) -> None:
        """Recompute steps_done, ts, ys for a candidate from the runhistory."""
        rh = self.runhistory

        keys = rh.get_instance_seed_budget_keys(
            cand.config,
            highest_observed_budget_only=False,
        )

        # keep only budgets in [min_budget, max_budget]
        keys = [
            k
            for k in keys
            if k.budget is not None and self._min_budget <= k.budget <= self._max_budget
        ]
        if not keys:
            cand.steps_done = 0
            cand.ts = []
            cand.ys = []
            return

        # sort by budget
        keys.sort(key=lambda k: k.budget)

        ts: list[float] = []
        ys: list[float] = []
        for key in keys:
            # generic way to get the cost for this single key
            cost = float(rh.average_cost(cand.config, [key]))
            if not math.isfinite(cost):
                y = float("nan")
            else:
                y = 1.0 - cost  # accuracy
            step = int(key.budget) - self._min_budget + 1
            t = step / self._b_max
            ts.append(t)
            ys.append(y)

        cand.steps_done = len(ts)
        cand.ts = ts
        cand.ys = ys

    def _build_context(self) -> list[Curve]:
        ctx: list[Curve] = []
        for c in self._candidates:
            self._update_candidate_from_runhistory(c)
            if c.steps_done <= 0:
                continue
            t_tensor = torch.tensor(c.ts, dtype=torch.float32)
            y_tensor = torch.tensor(c.ys, dtype=torch.float32)
            ctx.append(Curve(hyperparameters=c.z, t=t_tensor, y=y_tensor))
        return ctx

    def _best_so_far_accuracy(self) -> float:
        best = 0.0
        for c in self._candidates:
            self._update_candidate_from_runhistory(c)
            for y in c.ys:
                if math.isfinite(y) and y > best:
                    best = y
        return best

    def _select_incumbent_candidate(self) -> _IfBOCandidate:
        def cand_best_y(c: _IfBOCandidate) -> float:
            self._update_candidate_from_runhistory(c)
            vals = [y for y in c.ys if math.isfinite(y)]
            return max(vals) if vals else float("-inf")

        return max(self._candidates, key=cand_best_y)

    def _select_next_candidate(self, context: list[Curve]) -> _IfBOCandidate:
        """MFPI-random acquisition, same logic as in IfboOptimizer._select_next_candidate."""
        f_best = self._best_so_far_accuracy()

        h_rand = self._rng_py.randint(1, self._b_max)
        tau_rand = 10 ** self._rng_py.uniform(-4, -1)
        T_rand = f_best + tau_rand * (1.0 - f_best)

        pending: list[_IfBOCandidate] = []
        for c in self._candidates:
            self._update_candidate_from_runhistory(c)
            if c.steps_done < self._b_max:
                pending.append(c)

        if not pending:
            return self._select_incumbent_candidate()

        query: list[Curve] = []
        for c in pending:
            t_query = min(c.steps_done + h_rand, self._b_max) / self._b_max
            query.append(
                Curve(
                    hyperparameters=c.z,
                    t=torch.tensor([t_query], dtype=torch.float32),
                )
            )

        predictions = self.model.predict(context=context, query=query)
        T_tensor = torch.tensor(T_rand, dtype=torch.float32)
        pi_scores = torch.stack([pred.pi(T_tensor).squeeze() for pred in predictions])
        best_idx = int(torch.argmax(pi_scores))
        return pending[best_idx]

    def _step_to_budget(self, step: int) -> int:
        if not (1 <= step <= self._b_max):
            raise ValueError(f"Invalid step {step} for b_max={self._b_max}")
        return self._min_budget + step - 1

    # ---- main intensifier loop ----

    def __iter__(self) -> Iterator[TrialInfo]:
        """
        Main loop: repeatedly yield TrialInfo objects that correspond
        to one more freeze-thaw step for some candidate.
        """
        # Initialize base class bookkeeping and incumbents
        self.__post_init__()

        # Warm start: sync candidates with existing runhistory
        for c in self._candidates:
            self._update_candidate_from_runhistory(c)
        self._used_steps = sum(c.steps_done for c in self._candidates)

        rh = self.runhistory

        # If no candidate has been evaluated yet, do one purely random step
        if self._used_steps < self._total_steps and not any(
            c.steps_done > 0 for c in self._candidates
        ):
            first = self._rng_py.choice(self._candidates)
            step = first.steps_done + 1  # should be 1
            budget = float(self._step_to_budget(step))
            trial = TrialInfo(
                config=first.config, instance=None, seed=None, budget=budget
            )
            self._used_steps += 1
            yield trial

        # Main loop: delegate scheduling to FT-PFN based policy
        while self._used_steps < self._total_steps:
            context = self._build_context()
            next_cand = self._select_next_candidate(context)
            self._update_candidate_from_runhistory(next_cand)

            if next_cand.steps_done >= self._b_max:
                # nothing left to thaw
                break

            step = next_cand.steps_done + 1
            budget = float(self._step_to_budget(step))

            # Don't reschedule an already evaluated (or running) trial
            trial = TrialInfo(
                config=next_cand.config, instance=None, seed=0, budget=budget
            )
            evaluated = rh.get_trials(
                next_cand.config, highest_observed_budget_only=False
            )
            running = rh.get_running_trials(next_cand.config)
            if trial in evaluated or trial in running:
                # Shouldn't really happen in a sequential run, but be safe
                break

            self._used_steps += 1
            yield trial
