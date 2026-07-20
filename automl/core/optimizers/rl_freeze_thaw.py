"""RL controlled Freeze--Thaw hyperparameter optimization.

The environment deliberately has no hard dependency on Gym or Stable-Baselines.
Its observation is a vector of the best validation accuracies of the fixed
candidate pool (zero means not started).  Actions ``[0, len(start_budgets))``
start an unused candidate for the associated number of epochs; subsequent
actions thaw candidate ``action - len(start_budgets)`` for one epoch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from automl.core.optimizers.base_optimizer import Optimizer
from automl.cli import RuntimeConfigDict
from automl.logger import get_logger

logger = get_logger()


class LearningCurveSurrogate(Protocol):
    """Pluggable next-point predictor for an observed partial learning curve."""

    def predict_next(self, history: list[float]) -> float: ...


class LastValueSurrogate:
    """Conservative default surrogate: predict that the last accuracy persists."""

    def predict_next(self, history: list[float]) -> float:
        return history[-1] if history else 0.0


@dataclass
class FreezeThawCandidate:
    config: Configuration
    epochs: int = 0
    accuracies: list[float] = field(default_factory=list)
    surrogate_epochs: list[bool] = field(default_factory=list)
    metric_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def started(self) -> bool:
        return self.epochs > 0

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracies, default=0.0)


class FreezeThawEnvironment:
    """Small Gym-like Freeze--Thaw scheduler environment.

    ``step`` returns ``(observation, reward, done, info)``.  The reward is the
    improvement in the global best validation accuracy.  A surrogate step still
    consumes one *scheduler* epoch, so an episode has a bounded virtual budget;
    ``info['real_epochs']`` distinguishes the training cost actually incurred.
    """

    def __init__(
        self,
        space: ConfigurationSpace,
        train_fn: Any,
        *,
        n_initial_configs: int,
        total_epoch_budget: int,
        max_epochs_per_config: int,
        start_budgets: tuple[int, ...] = (1, 2, 4, 8),
        seed: int = 0,
        surrogate: LearningCurveSurrogate | None = None,
        surrogate_probability: float = 0.0,
        event_logger: Any | None = None,
    ):
        if n_initial_configs < 1 or total_epoch_budget < 1 or max_epochs_per_config < 1:
            raise ValueError(
                "Freeze--Thaw budgets and candidate count must be positive."
            )
        self.space, self.train_fn = space, train_fn
        self.n_initial_configs = n_initial_configs
        self.total_epoch_budget = total_epoch_budget
        self.max_epochs_per_config = max_epochs_per_config
        self.start_budgets = tuple(
            sorted({int(x) for x in start_budgets if int(x) > 0})
        )
        if not self.start_budgets:
            raise ValueError("start_budgets must contain a positive epoch count.")
        self.rng = random.Random(seed)
        self.seed = seed
        self.surrogate = surrogate
        self.surrogate_probability = float(surrogate_probability)
        self.logger = event_logger or logger
        self.candidates: list[FreezeThawCandidate] = []
        self.epochs_used = 0
        self.global_best = 0.0
        self.logger.info(
            "[FreezeThawEnvironment] Initialized: candidates=%d, total_epoch_budget=%d, "
            "max_epochs_per_config=%d, start_budgets=%s, surrogate=%s, "
            "surrogate_probability=%.3f, seed=%d",
            self.n_initial_configs,
            self.total_epoch_budget,
            self.max_epochs_per_config,
            self.start_budgets,
            type(self.surrogate).__name__ if self.surrogate else "disabled",
            self.surrogate_probability,
            self.seed,
        )

    @property
    def observation_size(self) -> int:
        return self.n_initial_configs

    @property
    def action_size(self) -> int:
        return len(self.start_budgets) + self.n_initial_configs

    def reset(self) -> np.ndarray:
        self.candidates = [
            FreezeThawCandidate(self.space.sample_configuration())
            for _ in range(self.n_initial_configs)
        ]
        self.epochs_used, self.global_best = 0, 0.0
        observation = self.observation()
        self.logger.info(
            "[FreezeThawEnvironment] Reset: sampled %d fixed configurations; "
            "observation=%s",
            len(self.candidates),
            observation.tolist(),
        )
        for index, candidate in enumerate(self.candidates):
            self.logger.debug(
                "[FreezeThawEnvironment] Candidate %d config=%s",
                index,
                dict(candidate.config),
            )
        return observation

    def observation(self) -> np.ndarray:
        return np.asarray([c.best_accuracy for c in self.candidates], dtype=np.float32)

    def valid_action_mask(self) -> np.ndarray:
        remaining = self.total_epoch_budget - self.epochs_used
        mask = np.zeros(self.action_size, dtype=bool)
        has_unused = any(not c.started for c in self.candidates)
        for action in range(len(self.start_budgets)):
            mask[action] = has_unused and remaining > 0
        for i, candidate in enumerate(self.candidates):
            mask[len(self.start_budgets) + i] = (
                candidate.started
                and candidate.epochs < self.max_epochs_per_config
                and remaining > 0
            )
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if not self.candidates:
            raise RuntimeError("Call reset() before step().")
        mask = self.valid_action_mask()
        if action < 0 or action >= self.action_size or not mask[action]:
            done = not mask.any()
            self.logger.warning(
                "[FreezeThawEnvironment] Invalid action=%s; valid_actions=%s; "
                "epochs_used=%d/%d; done=%s",
                action,
                np.flatnonzero(mask).tolist(),
                self.epochs_used,
                self.total_epoch_budget,
                done,
            )
            return (
                self.observation(),
                -0.01,
                done,
                {"invalid_action": True, "real_epochs": 0},
            )

        is_start = action < len(self.start_budgets)
        if is_start:
            candidate = self.rng.choice([c for c in self.candidates if not c.started])
            requested = self.start_budgets[action]
        else:
            candidate = self.candidates[action - len(self.start_budgets)]
            requested = 1
        epochs = min(
            requested,
            self.total_epoch_budget - self.epochs_used,
            self.max_epochs_per_config - candidate.epochs,
        )
        previous_best = self.global_best
        real_epochs = 0
        surrogate_used = False
        candidate_index = self.candidates.index(candidate)
        self.logger.info(
            "[FreezeThawEnvironment] Step: action=%d (%s), candidate=%d, "
            "requested_epochs=%d, scheduled_epochs=%d, candidate_epochs=%d, "
            "budget=%d/%d",
            action,
            "start" if is_start else "thaw",
            candidate_index,
            requested,
            epochs,
            candidate.epochs,
            self.epochs_used,
            self.total_epoch_budget,
        )
        self.logger.debug(
            "[FreezeThawEnvironment] Step action mask=%s; candidate config=%s",
            mask.astype(int).tolist(),
            dict(candidate.config),
        )
        for _ in range(epochs):
            use_surrogate = bool(
                candidate.accuracies
                and self.surrogate
                and self.rng.random() < self.surrogate_probability
            )
            if use_surrogate:
                accuracy = float(
                    np.clip(self.surrogate.predict_next(candidate.accuracies), 0.0, 1.0)
                )
                surrogate_used = True
                self.logger.info(
                    "[FreezeThawEnvironment] Candidate %d epoch %d: surrogate predicted "
                    "val_accuracy=%.4f from history=%s",
                    candidate_index,
                    candidate.epochs + 1,
                    accuracy,
                    [round(value, 4) for value in candidate.accuracies],
                )
            else:
                # The base hook is stateless, hence one call represents one logical epoch.
                self.logger.info(
                    "[FreezeThawEnvironment] Candidate %d epoch %d: running real training "
                    "with seed=%d",
                    candidate_index,
                    candidate.epochs + 1,
                    self.seed + self.epochs_used,
                )
                error = float(
                    self.train_fn(candidate.config, self.seed + self.epochs_used, 1)
                )
                accuracy = (
                    0.0
                    if not np.isfinite(error)
                    else float(np.clip(1.0 - error, 0.0, 1.0))
                )
                real_epochs += 1
            candidate.epochs += 1
            candidate.accuracies.append(accuracy)
            candidate.surrogate_epochs.append(use_surrogate)
            metrics: dict[str, Any] = {
                "val_accuracy": accuracy,
                "val_error": 1.0 - accuracy,
                "surrogate": use_surrogate,
            }
            # Preserve richer trainer-provided epoch metrics (notably train loss)
            # whenever this was an actual base-optimizer evaluation.
            self_history = getattr(self.train_fn, "__self__", None)
            if not use_surrogate and self_history:
                history = getattr(self_history, "history", [])
                if history and history[-1].get("epoch_history"):
                    metrics.update(history[-1]["epoch_history"][-1])
            candidate.metric_history.append(metrics)
            self.epochs_used += 1
            self.global_best = max(self.global_best, accuracy)
            self.logger.info(
                "[FreezeThawEnvironment] Candidate %d epoch %d complete: "
                "val_accuracy=%.4f, candidate_best=%.4f, global_best=%.4f, "
                "source=%s, budget=%d/%d",
                candidate_index,
                candidate.epochs,
                accuracy,
                candidate.best_accuracy,
                self.global_best,
                "surrogate" if use_surrogate else "real",
                self.epochs_used,
                self.total_epoch_budget,
            )
        done = (
            self.epochs_used >= self.total_epoch_budget
            or not self.valid_action_mask().any()
        )
        reward = self.global_best - previous_best
        observation = self.observation()
        self.logger.info(
            "[FreezeThawEnvironment] Step complete: candidate=%d, reward=%.6f, "
            "real_epochs=%d, surrogate_used=%s, done=%s, observation=%s",
            candidate_index,
            reward,
            real_epochs,
            surrogate_used,
            done,
            observation.tolist(),
        )
        return (
            observation,
            reward,
            done,
            {
                "candidate_index": candidate_index,
                "started": is_start,
                "epochs": epochs,
                "real_epochs": real_epochs,
                "surrogate_used": surrogate_used,
                "global_best_accuracy": self.global_best,
            },
        )


class RandomFreezeThawController:
    """Uniform random valid-action baseline for the identical environment."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: FreezeThawEnvironment, _observation: np.ndarray) -> int:
        valid_actions = np.flatnonzero(env.valid_action_mask())
        action = int(self.rng.choice(valid_actions))
        env.logger.info(
            "[RandomFreezeThawController] Selected action=%d uniformly from %s",
            action,
            valid_actions.tolist(),
        )
        return action


class RLFreezeThawOptimizer(Optimizer):
    """PPO scheduler with a random-controller fallback/baseline.

    Set ``optimizer: rl_freeze_thaw`` in the runtime configuration.  Optional
    keys are ``rl_controller`` (``ppo`` or ``random``), ``rl_ppo_episodes``,
    ``rl_n_initial_configs``, ``rl_total_epoch_budget`` and
    ``rl_surrogate_probability``.
    """

    def __init__(self, runtime_config: RuntimeConfigDict):
        super().__init__(runtime_config)
        if hasattr(self.dataset, "load_base_data"):
            self.dataset.load_base_data()
        self.rng = random.Random(runtime_config["seed"])
        self.n_initial_configs = int(
            runtime_config.get("rl_n_initial_configs", runtime_config["n_trials"])
        )
        self.total_epoch_budget = int(
            runtime_config.get(
                "rl_total_epoch_budget",
                runtime_config["n_trials"] * runtime_config["min_budget"],
            )
        )
        self.start_budgets = tuple(runtime_config.get("rl_start_budgets", (1, 2, 4, 8)))
        self.surrogate_probability = float(
            runtime_config.get("rl_surrogate_probability", 0.0)
        )
        self.controller_kind = str(runtime_config.get("rl_controller", "ppo")).lower()
        self.ppo_episodes = int(runtime_config.get("rl_ppo_episodes", 2))
        self.logger.info(
            "[RLFreezeThawOptimizer] Initialized: controller=%s, ppo_episodes=%d, "
            "n_initial_configs=%d, total_epoch_budget=%d, start_budgets=%s, "
            "surrogate_probability=%.3f",
            self.controller_kind,
            self.ppo_episodes,
            self.n_initial_configs,
            self.total_epoch_budget,
            self.start_budgets,
            self.surrogate_probability,
        )

    def _make_env(self, seed_offset: int = 0) -> FreezeThawEnvironment:
        return FreezeThawEnvironment(
            self.space,
            self.train_single_configuration,
            n_initial_configs=self.n_initial_configs,
            total_epoch_budget=self.total_epoch_budget,
            max_epochs_per_config=int(self.runtime_config["max_budget"]),
            start_budgets=self.start_budgets,
            seed=self.runtime_config["seed"] + seed_offset,
            surrogate=LastValueSurrogate(),
            surrogate_probability=self.surrogate_probability,
            event_logger=self.logger,
        )

    def _train_ppo(self):
        """Train a compact masked categorical PPO policy without external RL packages."""
        import torch
        from torch import nn

        torch.manual_seed(self.runtime_config["seed"])
        model = nn.Sequential(
            nn.Linear(self.n_initial_configs, 64),
            nn.Tanh(),
            nn.Linear(64, self._make_env().action_size),
        )
        value_net = nn.Sequential(
            nn.Linear(self.n_initial_configs, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(value_net.parameters()), lr=3e-3
        )
        self.logger.info(
            "[RLFreezeThawOptimizer] Starting PPO training: episodes=%d, "
            "policy_parameters=%d, value_parameters=%d",
            max(1, self.ppo_episodes),
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in value_net.parameters()),
        )
        for episode in range(max(1, self.ppo_episodes)):
            env, obs, done = self._make_env(episode + 1), None, False
            obs = env.reset()
            records = []
            episode_reward = 0.0
            self.logger.info(
                "[RLFreezeThawOptimizer] PPO episode %d/%d started.",
                episode + 1,
                max(1, self.ppo_episodes),
            )
            while not done:
                x = torch.tensor(obs, dtype=torch.float32)
                action_mask = torch.tensor(env.valid_action_mask())
                logits = model(x).masked_fill(~action_mask, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                next_obs, reward, done, _ = env.step(int(action))
                records.append(
                    (
                        x,
                        action,
                        dist.log_prob(action).detach(),
                        value_net(x).squeeze().detach(),
                        reward,
                        action_mask,
                    )
                )
                obs = next_obs
                episode_reward += reward
            returns, running = [], 0.0
            for record in reversed(records):
                reward = record[4]
                running = reward + 0.99 * running
                returns.append(running)
            returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            for _ in range(4):
                policy_losses, value_losses = [], []
                for (x, action, old_logp, old_value, _, mask), target in zip(
                    records, returns
                ):
                    # Reuse the mask from the sampled state so PPO optimizes the
                    # same constrained action distribution it interacted with.
                    dist = torch.distributions.Categorical(
                        logits=model(x).masked_fill(~mask, -1e9)
                    )
                    ratio = (dist.log_prob(action) - old_logp).exp()
                    advantage = target - old_value
                    policy_losses.append(
                        -torch.minimum(
                            ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage
                        )
                        - 0.01 * dist.entropy()
                    )
                    value_losses.append((value_net(x).squeeze() - target).pow(2))
                optimizer.zero_grad()
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                (policy_loss + 0.5 * value_loss).backward()
                optimizer.step()
                self.logger.debug(
                    "[RLFreezeThawOptimizer] PPO episode %d update: "
                    "policy_loss=%.6f, value_loss=%.6f",
                    episode + 1,
                    policy_loss.item(),
                    value_loss.item(),
                )
            self.logger.info(
                "[RLFreezeThawOptimizer] PPO episode %d complete: steps=%d, "
                "return=%.6f, global_best_accuracy=%.4f, epochs_used=%d/%d",
                episode + 1,
                len(records),
                episode_reward,
                env.global_best,
                env.epochs_used,
                env.total_epoch_budget,
            )
        self.logger.info("[RLFreezeThawOptimizer] PPO training complete.")
        return model

    def _ppo_action(
        self, model, env: FreezeThawEnvironment, observation: np.ndarray
    ) -> int:
        import torch

        with torch.no_grad():
            logits = model(torch.tensor(observation, dtype=torch.float32))
            logits = logits.masked_fill(~torch.tensor(env.valid_action_mask()), -1e9)
            action = int(torch.argmax(logits).item())
            env.logger.info(
                "[RLFreezeThawOptimizer] PPO selected action=%d; valid_actions=%s; "
                "logits=%s",
                action,
                np.flatnonzero(env.valid_action_mask()).tolist(),
                [round(float(value), 4) for value in logits.tolist()],
            )
            return action

    def run(self):
        incumbent = None
        try:
            self.logger.info(
                "[RLFreezeThawOptimizer] Starting final Freeze--Thaw run with controller=%s.",
                self.controller_kind,
            )
            if self.controller_kind == "random":
                controller, policy = (
                    RandomFreezeThawController(self.runtime_config["seed"]),
                    None,
                )
            elif self.controller_kind == "ppo":
                policy, controller = self._train_ppo(), None
            else:
                raise ValueError("rl_controller must be 'ppo' or 'random'.")
            env, observation, done = self._make_env(10_000), None, False
            observation = env.reset()
            while not done:
                action = (
                    controller.act(env, observation)
                    if controller
                    else self._ppo_action(policy, env, observation)
                )
                observation, _, done, _ = env.step(action)
            started = [c for c in env.candidates if c.started]
            incumbent = (
                max(started, key=lambda c: c.best_accuracy).config if started else None
            )
            self.logger.info(
                "[RLFreezeThawOptimizer] Final run complete: started_configs=%d/%d, "
                "epochs_used=%d/%d, best_accuracy=%.4f, incumbent=%s",
                len(started),
                len(env.candidates),
                env.epochs_used,
                env.total_epoch_budget,
                env.global_best,
                dict(incumbent) if incumbent is not None else None,
            )
            for index, candidate in enumerate(env.candidates):
                self.logger.info(
                    "[RLFreezeThawOptimizer] Final candidate %d: started=%s, epochs=%d, "
                    "best_accuracy=%.4f, surrogate_epochs=%d",
                    index,
                    candidate.started,
                    candidate.epochs,
                    candidate.best_accuracy,
                    sum(candidate.surrogate_epochs),
                )
        except KeyboardInterrupt:
            self.logger.warning(
                "[RLFreezeThawOptimizer] Interrupted; saving available history."
            )
        except Exception:
            self.logger.exception(
                "[RLFreezeThawOptimizer] Freeze--Thaw optimization failed; "
                "finalizing any available history before re-raising."
            )
            raise
        finally:
            self.logger.info(
                "[RLFreezeThawOptimizer] Finalizing optimization; incumbent_available=%s.",
                incumbent is not None,
            )
            self._finalize_optimization(incumbent)
