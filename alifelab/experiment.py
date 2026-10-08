"""Declarative experiment configuration and run management.

An :class:`Experiment` is created from a :class:`RunConfig` (a fully
JSON-serializable description of the world, interfaces, agent and training
settings). It builds the environment and agent, constructs a
:class:`~alifelab.train.Trainer` pointed at ``runs/<name>/<run_id>/``, and
offers resume from a saved checkpoint.

Example::

    config = RunConfig(
        name="snake_ai",
        seed=7,
        episodes=50,
        world={...},
        agent={"name": "dqn", "architecture": "cnn", "feature_dim": 64},
    )
    experiment = Experiment(config)
    experiment.train()
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .agents.agents import build_agent
from .checkpoint import json_dumps_pretty
from .core.registry import AGENTS
from .train import Trainer, build_env as build_env_from_spec
from .world import WorldSpec, rules_from_config


@dataclass
class RunConfig:
    """Declarative, JSON-serializable description of a training run."""

    name: str
    world: Mapping[str, Any]
    seed: int = 0
    episodes: int = 100
    observation: Any = None
    action: Any = "discrete_move"
    reward: Any = "composite"
    agent: Mapping[str, Any] = field(default_factory=lambda: {"name": "random"})
    train_every: int = 1
    max_steps: int | None = None
    eval_every: int = 10
    eval_episodes: int = 5
    checkpoint_every: int = 1
    best_metric: str = "evaluation_return"
    run_id: str | None = None
    runs_dir: str = "runs"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {"name": self.name, "run_id": self.run_id, "runs_dir": self.runs_dir},
            "seed": self.seed,
            "training": {
                "episodes": self.episodes,
                "train_every": self.train_every,
                "max_steps": self.max_steps,
                "eval_every": self.eval_every,
                "eval_episodes": self.eval_episodes,
                "checkpoint_every": self.checkpoint_every,
                "best_metric": self.best_metric,
            },
            "world": dict(self.world),
            "observation": dict(self.observation) if isinstance(self.observation, Mapping) else self.observation,
            "action": dict(self.action) if isinstance(self.action, Mapping) else self.action,
            "reward": dict(self.reward) if isinstance(self.reward, Mapping) else self.reward,
            "agent": dict(self.agent),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        experiment = data.get("experiment") or {}
        training = data.get("training") or {}
        return cls(
            name=str(experiment.get("name", "run")),
            run_id=experiment.get("run_id"),
            runs_dir=str(experiment.get("runs_dir", "runs")),
            seed=int(data.get("seed", 0)),
            episodes=int(training.get("episodes", 100)),
            train_every=int(training.get("train_every", 1)),
            max_steps=training.get("max_steps"),
            eval_every=int(training.get("eval_every", 10)),
            eval_episodes=int(training.get("eval_episodes", 5)),
            checkpoint_every=int(training.get("checkpoint_every", 1)),
            best_metric=str(training.get("best_metric", "evaluation_return")),
            world=dict(data.get("world") or {}),
            observation=data.get("observation"),
            action=data.get("action", "discrete_move"),
            reward=data.get("reward", "composite"),
            agent=dict(data.get("agent") or {"name": "random"}),
            extra=dict(data.get("extra") or {}),
        )

    def build_world_spec(self) -> WorldSpec:
        world = dict(self.world)
        rules_cfg = world.pop("rules", None)
        entities = world.pop("entities", [])
        spec = WorldSpec(**{**world, "rules": rules_from_config(rules_cfg), "entities": list(entities)})
        return spec  # type: ignore[arg-type]

    def build_env(self):
        return build_env_from_spec(
            self.build_world_spec(),
            observation=self.observation,
            action=self.action,
            reward=self.reward,
            max_steps=self.max_steps if self.max_steps else 200,
            seed=self.seed,
        )

    def build_agent_for(self, env):
        params = {k: v for k, v in dict(self.agent).items() if k != "name"}
        name = str(self.agent.get("name", "random"))
        return build_agent(name, env.observation_space, env.action_space, **params)

    def default_run_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")


class Experiment:
    """Directory-backed training run built from a :class:`RunConfig`."""

    def __init__(self, config: RunConfig, run_id: str | None = None, resume: str | os.PathLike | None = None):
        self.config = config
        if run_id is not None:
            self.config.run_id = run_id
        elif self.config.run_id is None:
            self.config.run_id = self.config.default_run_id()

        self.env = self.config.build_env()
        self.agent = self.config.build_agent_for(self.env)
        self.trainer = Trainer(
            self.env,
            self.agent,
            episodes=self.config.episodes,
            seed=self.config.seed,
            train_every=self.config.train_every,
            max_steps=self.config.max_steps,
            experiment_name=self.config.name,
            run_id=self.config.run_id,
            runs_dir=self.config.runs_dir,
            config=self.config.to_dict(),
            eval_every=self.config.eval_every,
            eval_episodes=self.config.eval_episodes,
            checkpoint_every=self.config.checkpoint_every,
            best_metric=self.config.best_metric,
        )
        if resume is not None:
            self.trainer._resume(resume)

    @property
    def run_dir(self) -> str:
        return self.trainer.run_dir

    def write_config(self) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write(json_dumps_pretty(self.config.to_dict()))
            handle.write("\n")
        return os.path.join(self.run_dir, "config.json")

    def train(self, resume_from: str | os.PathLike | None = None):
        self.write_config()
        return self.trainer.train(resume_from=resume_from)

    def evaluate(self, episodes: int | None = None):
        return self.trainer.evaluate(episodes or self.config.eval_episodes)

    def save_checkpoint(self, path: str | os.PathLike):
        self.write_config()
        return self.trainer.save_checkpoint(path)

    @classmethod
    def open(cls, run_dir: str | os.PathLike, *, resume: bool = True):
        """Rebuild an Experiment from an existing run directory."""
        run_dir = str(run_dir)
        config: RunConfig | None = None
        for candidate in ("config.json", os.path.join(run_dir, "config.json")):
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as handle:
                    config = RunConfig.from_dict(json.load(handle))
                break
        if config is None:
            raise FileNotFoundError(f"No config.json found in {run_dir!r}")
        checkpoint = None
        if resume:
            candidate = os.path.join(run_dir, "checkpoint_latest.pt")
            if os.path.exists(candidate):
                checkpoint = candidate
        return cls(config, resume=checkpoint)