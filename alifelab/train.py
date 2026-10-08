"""High-level training harness: wire a LifeEnv + Agent and run episodes.

The :class:`Trainer` owns training orchestration, per-episode testing, and an
optional experiment directory under ``runs/<experiment>/<run_id>`` containing
``config.json``, ``metrics.jsonl``, ``evaluation.json``, ``metadata.json`` and
``checkpoint_latest.pt`` / ``checkpoint_best.pt``.

All checkpoint/experiment logic is algorithm-agnostic: it talks to the agent
only through the generic ``state_dict()``/``load_state_dict()`` protocol and to
metrics through the dict returned by ``agent.train()``.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .actions import DiscreteMoveActions
from .checkpoint import (
    apply_agent_state,
    as_python,
    build_checkpoint,
    capture_rng_states,
    json_dumps_pretty,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from .core.interfaces import Agent, Transition
from .env import CompositeReward, LifeEnv
from .obs import MultiChannelGridObservation
from .world import WorldSpec, build_world, rules_from_config


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    return_: float
    terminated: bool
    truncated: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    training_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def mean_loss(self) -> float:
        if not self.losses:
            return float("nan")
        return float(np.mean(self.losses))


def _mean_numeric(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _merge_metrics(accumulated: dict[str, list[float]]) -> dict[str, float]:
    return {name: _mean_numeric(values) for name, values in accumulated.items()}


class Trainer:
    """Loop an agent against a LifeEnv, collecting per-episode results.

    Optional experiment tracking: set ``experiment_name`` (and optionally a run
    id) to create ``runs/<experiment_name>/<run_id>/`` with config, metrics,
    evaluation and checkpoint files, including a best checkpoint tracked by
    ``best_metric`` (e.g. ``evaluation_return``).
    """

    def __init__(
        self,
        env: LifeEnv,
        agent: Agent,
        *,
        episodes: int = 100,
        seed: int = 0,
        train_every: int = 1,
        max_steps: int | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        runs_dir: str | os.PathLike = "runs",
        config: Mapping[str, Any] | None = None,
        eval_every: int = 0,
        eval_episodes: int = 5,
        checkpoint_every: int = 1,
        best_metric: str = "evaluation_return",
        rng_tracking: bool = True,
    ):
        self.env = env
        self.agent = agent
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.train_every = max(1, int(train_every))
        self.max_steps = int(max_steps) if max_steps else env.max_steps
        self.experiment_name = experiment_name
        self.run_id = run_id or _default_run_id()
        self.runs_dir = str(runs_dir)
        self.eval_every = max(0, int(eval_every))
        self.eval_episodes = max(1, int(eval_episodes))
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.best_metric = str(best_metric)
        self.rng_tracking = bool(rng_tracking)
        self.config = self._resolve_config(config)

        self.history: list[EpisodeResult] = []
        self.global_steps = 0
        self.completed_episodes = 0
        self._start_time: float | None = None
        self._best_value: float | None = None
        self._best_episode: int = 0
        self._best_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ paths

    @property
    def run_dir(self) -> str:
        return os.path.join(self.runs_dir, self.experiment_name or "run", self.run_id)

    @property
    def tracking(self) -> bool:
        return self.experiment_name is not None

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "checkpoint_latest.pt")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "checkpoint_best.pt")

    def _resolve_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return {**self._introspect_config(), **as_python(dict(config))}
        return self._introspect_config()

    def _introspect_config(self) -> dict[str, Any]:
        obs = self.env.observation_builder
        return {
            "agent": {
                "name": type(self.agent).__name__,
                "architecture": getattr(self.agent, "architecture_name", None),
                "feature_dim": getattr(self.agent, "feature_dim", None),
                "recurrent": getattr(self.agent, "recurrent_name", None),
                "lr": getattr(self.agent, "lr", None),
                "cfg": as_python(getattr(self.agent, "cfg", {})),
            },
            "env": {
                "grid": (self.env.grid_size or None),
                "controller": self.env.controller_name,
                "control_switch": self.env.control_switch,
                "max_steps": self.max_steps,
                "wrap": bool(self.env.world.wrap),
                "seed": self.env.world.seed,
            },
            "observation": {
                "type": type(obs).__name__,
                "kind": getattr(obs, "kind", None),
                "shape": list(getattr(obs, "shape", ())),
                "cfg": as_python(getattr(obs, "cfg", {})),
            },
            "action": {"type": type(self.env.action_interface).__name__},
            "reward": {"type": type(self.env.reward_function).__name__},
            "training": {
                "episodes": self.episodes,
                "seed": self.seed,
                "train_every": self.train_every,
                "eval_every": self.eval_every,
                "eval_episodes": self.eval_episodes,
                "checkpoint_every": self.checkpoint_every,
                "best_metric": self.best_metric,
            },
        }

    # ------------------------------------------------------------- run setup

    def _setup_run_dir(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write(json_dumps_pretty(self.config))
            handle.write("\n")
        self._write_metadata(partial=True)
        metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        if not os.path.exists(metrics_path):
            open(metrics_path, "a", encoding="utf-8").close()

    def _write_metadata(self, partial: bool = False) -> None:
        meta: dict[str, Any] = {
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "seed": self.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(getattr(self.agent, "device", "cpu")),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if not partial:
            meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if self._start_time is not None:
                meta["duration_seconds"] = round(time.monotonic() - self._start_time, 3)
            meta["episodes_completed"] = self.completed_episodes
            meta["global_steps"] = self.global_steps
            meta["best_metric"] = self.best_metric
            meta["best_value"] = self._best_value
        with open(os.path.join(self.run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            handle.write(json_dumps_pretty(meta))
            handle.write("\n")

    def _record_metric_row(self, row: Mapping[str, Any]) -> None:
        if not self.tracking:
            return
        row = as_python(dict(row))
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    # ------------------------------------------------------------------ train

    def run(self, resume_from: str | os.PathLike | None = None) -> list[EpisodeResult]:
        """Alias for :meth:`train` (kept for backwards compatibility)."""
        return self.train(resume_from=resume_from)

    def train(self, resume_from: str | os.PathLike | None = None) -> list[EpisodeResult]:
        """Run all pending episodes, resuming from an optional checkpoint path."""
        if self._start_time is None:
            self._start_time = time.monotonic()
        if self.tracking:
            self._setup_run_dir()
        if resume_from is not None:
            self._resume(resume_from)

        while self.completed_episodes < self.episodes:
            episode = self.completed_episodes + 1
            result = self._train_episode(episode)
            self.global_steps += result.steps
            self.completed_episodes = episode
            self.history.append(result)
            self._record_training(result)

            if self.eval_every and (episode % self.eval_every == 0):
                self._evaluate_and_track(episode)
            elif episode == self.episodes and self.eval_every:
                self._evaluate_and_track(episode)

            if self.tracking and self.checkpoint_every and episode % self.checkpoint_every == 0:
                self.save_checkpoint(self.latest_checkpoint_path)

        if self.tracking:
            self._write_metadata(partial=False)
        return self.history

    def _train_episode(self, episode: int) -> EpisodeResult:
        obs, _ = self.env.reset(seed=self.seed)
        self.agent.reset()
        total = 0.0
        losses: list[float] = []
        accumulated: dict[str, list[float]] = {}
        terminated = truncated = False
        steps = 0
        for step in range(self.max_steps):
            action = self.agent.act(obs, explore=True)
            next_obs, reward, term, trunc, info = self.env.step(action)
            self.agent.observe(Transition(obs, action, reward, next_obs, term, trunc, info))
            if (step + 1) % self.train_every == 0:
                metrics = self.agent.train()
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and value == value:
                        accumulated.setdefault(name, []).append(float(value))
                loss = metrics.get("loss")
                if loss is not None and loss == loss:
                    losses.append(float(loss))
            obs = next_obs
            total += reward
            steps += 1
            if term or trunc:
                terminated, truncated = term, trunc
                break
        return EpisodeResult(
            episode=episode,
            steps=steps,
            return_=total,
            terminated=terminated,
            truncated=truncated,
            metrics=dict(info.get("metrics", {})),
            losses=losses,
            training_metrics=_merge_metrics(accumulated),
        )

    # ---------------------------------------------------------------- metrics

    def _current_lr(self) -> float | None:
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            return float(optimizer.param_groups[0].get("lr", float("nan")))
        return None

    def _record_training(self, result: EpisodeResult) -> None:
        row: dict[str, Any] = {
            "type": "training",
            "global_step": self.global_steps,
            "episode": result.episode,
            "episode_return": result.return_,
            "episode_length": result.steps,
            "training_loss": result.mean_loss,
            "episode_terminated": result.terminated,
            "episode_truncated": result.truncated,
            "elapsed_time": self._elapsed(),
        }
        lr = self._current_lr()
        if lr is not None:
            row["learning_rate"] = lr
        row.update(result.training_metrics)
        for key, value in result.metrics.items():
            row.setdefault(f"info_{key}", value)
        self._record_metric_row(row)

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._start_time, 3) if self._start_time is not None else 0.0

    def _evaluate_and_track(self, episode: int) -> dict[str, float]:
        result = self.evaluate(self.eval_episodes)
        track = dict(result)
        track["evaluation_return"] = result["mean_return"]
        track["evaluation_length"] = result["mean_length"]
        track["episode"] = episode
        track["global_step"] = self.global_steps
        row: dict[str, Any] = {
            "type": "evaluation",
            "global_step": self.global_steps,
            "episode": episode,
            "evaluation_return": track["evaluation_return"],
            "evaluation_length": track["evaluation_length"],
            "elapsed_time": self._elapsed(),
        }
        for key, value in result.items():
            if key not in {"mean_return", "mean_length", "episode", "global_step"}:
                row[f"evaluation_{key}"] = value
        self._record_metric_row(row)
        self._write_evaluation(episode, track)
        self._maybe_update_best(episode, track)
        return result

    def _maybe_update_best(self, episode: int, result: Mapping[str, Any]) -> None:
        if self.best_metric not in result:
            return
        value = float(result[self.best_metric])
        if self._best_value is None or value > self._best_value:
            self._best_value = value
            self._best_episode = episode
            self._best_result = dict(result)
            if self.tracking:
                self.save_checkpoint(self.best_checkpoint_path)

    def _write_evaluation(self, episode: int, result: Mapping[str, Any]) -> None:
        if not self.tracking:
            return
        payload = {
            "last": {"episode": episode, **as_python(dict(result))},
            "best": {
                "metric": self.best_metric,
                "value": self._best_value,
                "episode": self._best_episode,
                "result": self._best_result,
            },
        }
        with open(os.path.join(self.run_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
            handle.write(json_dumps_pretty(payload))
            handle.write("\n")

    def evaluate(self, episodes: int = 10, *, seed: int | None = None) -> dict[str, float]:
        """Roll out the agent with exploration off; returns mean statistics."""
        returns: list[float] = []
        lengths: list[int] = []
        foods: list[int] = []
        survivors = 0
        rng_seed = self.seed if seed is None else int(seed)
        for _ in range(int(episodes)):
            obs, _ = self.env.reset(seed=rng_seed)
            self.agent.reset()
            total = 0.0
            length = 0
            for _ in range(self.max_steps):
                action = self.agent.act(obs, explore=False)
                obs, reward, term, trunc, info = self.env.step(action)
                total += reward
                length += 1
                if term or trunc:
                    break
            returns.append(total)
            lengths.append(length)
            foods.append(int(info.get("metrics", {}).get("food", 0)))
            survivors += int(info.get("metrics", {}).get("survived", False))
        return {
            "episodes": float(len(returns)),
            "mean_return": float(np.mean(returns)),
            "mean_length": float(np.mean(lengths)),
            "mean_food": float(np.mean(foods)),
            "survival_rate": float(survivors / max(1, len(returns))),
        }

    def returns(self) -> list[float]:
        return [result.return_ for result in self.history]

    def mean_return(self, window: int | None = None) -> float:
        values = self.returns()
        if window:
            values = values[-int(window):]
        return float(np.mean(values)) if values else float("nan")

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str | os.PathLike) -> str:
        """Write a full, resumable checkpoint of the current trainer + agent."""
        payload = build_checkpoint(
            self.agent,
            config=self.config,
            env_state=self.env.get_state(),
            global_step=self.global_steps,
            episode=self.completed_episodes,
            rng=capture_rng_states() if self.rng_tracking else None,
            metrics={"mean_return": self.mean_return(window=self.eval_episodes or 10)},
            metadata={"experiment": self.experiment_name, "run_id": self.run_id, "best_metric": self.best_metric, "best_value": self._best_value},
            extra={"trainer": {"completed_episodes": self.completed_episodes, "episodes": self.episodes, "seed": self.seed, "history_length": len(self.history)}},
        )
        return save_checkpoint(str(path), payload)

    def load_checkpoint(self, path: str | os.PathLike) -> dict[str, Any]:
        """Load a checkpoint, restoring the agent and trainer state in-place."""
        payload = load_checkpoint(str(path), device=getattr(self.agent, "device", None) or "cpu")
        apply_agent_state(self.agent, payload)
        if "rng" in payload and self.rng_tracking:
            restore_rng_states(payload["rng"])
        self.global_steps = int(payload.get("global_step", 0))
        self.completed_episodes = int(payload.get("episode", 0))
        extra = payload.get("extra") or {}
        trainer_state = (extra.get("trainer") or {}) if isinstance(extra, dict) else {}
        if trainer_state.get("episodes"):
            self.episodes = int(trainer_state["episodes"])
        self.config = payload.get("config") or self.config
        return payload

    def _resume(self, resume_from: str | os.PathLike) -> None:
        resume_from = str(resume_from)
        if os.path.isdir(resume_from):
            candidate = os.path.join(resume_from, "checkpoint_latest.pt")
            if os.path.exists(candidate):
                resume_from = candidate
        payload = self.load_checkpoint(resume_from)
        self._log(f"resumed from {resume_from} at episode {payload.get('episode')} step {payload.get('global_step')}")

    def _log(self, message: str) -> None:
        print(f"[trainer] {message}")


def _default_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _resolve_spec(name_or_spec: Any, name: str):
    """Resolve an interface given as a registry name, a {"name","params"} dict, or an instance."""
    if name_or_spec is None:
        return None
    if isinstance(name_or_spec, str):
        from .core.registry import OBSERVATIONS, ACTIONS, REWARDS

        table = {"observation": OBSERVATIONS, "action": ACTIONS, "reward": REWARDS}[name]
        return table.create(name_or_spec)
    if isinstance(name_or_spec, Mapping) and "name" in name_or_spec:
        from .core.registry import OBSERVATIONS, ACTIONS, REWARDS

        table = {"observation": OBSERVATIONS, "action": ACTIONS, "reward": REWARDS}[name]
        params = dict(name_or_spec.get("params") or {})
        return table.create(name_or_spec["name"], **params)
    return name_or_spec


def resolve_observation(name_or_spec: Any, spec: WorldSpec):
    if name_or_spec is None:
        return MultiChannelGridObservation(grid_size=(spec.height, spec.width), channels=8)
    return _resolve_spec(name_or_spec, "observation")


def build_env(
    spec: WorldSpec,
    *,
    observation: Any = None,
    action: Any = None,
    reward: Any = None,
    controller: str = "agent",
    control_switch: str = "none",
    max_steps: int = 200,
    seed: int | None = None,
    **env_cfg: Any,
) -> LifeEnv:
    """Construct a LifeEnv from a WorldSpec plus optional name/instance interfaces."""
    if action is None:
        action = DiscreteMoveActions(n=5)
    else:
        action = _resolve_spec(action, "action")
    if reward is None:
        reward = CompositeReward()
    else:
        reward = _resolve_spec(reward, "reward")
    world = build_world(spec)
    observation = resolve_observation(observation, spec)
    return LifeEnv(
        world,
        action,
        observation,
        reward,
        max_steps=max_steps,
        controller=controller,
        control_switch=control_switch,
        seed=seed,
        **env_cfg,
    )