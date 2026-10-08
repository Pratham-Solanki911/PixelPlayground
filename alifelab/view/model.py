"""Engine-independent view model consumed by the visualizer.

These dataclasses are the *only* thing the interactive visualizer reads.
They contain no simulation, RL, trainer, or agent objects -- just plain
numbers, tuples and NumPy arrays, so any source (a live simulation, a saved
checkpoint, a scripted demo) can produce them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ViewEntity:
    """Generic, renderable entity -- NOT snake-specific.

    Everything here is filled from the standardized world-state interface, so
    predators, prey, plants, rocks, evolving agents and arbitrary future
    entities render through the same code path.
    """

    id: int
    kind: str
    species: str
    x: int
    y: int
    body: tuple[tuple[int, int], ...] = ()
    alive: bool = True
    size: float = 1.0
    health: float | None = None
    max_health: float | None = None
    energy: float | None = None
    max_energy: float | None = None
    age: float = 0.0
    lifetime: float | None = None
    velocity: tuple[int, int] | None = None
    direction: tuple[int, int] | None = None
    genome: Mapping[str, float] = field(default_factory=dict)
    controller: str | None = None
    state: str | None = None
    action: str | None = None
    edible: bool = False
    obstacle: bool = False
    solid: bool = False
    movable: bool = False
    predator: bool = False

    @property
    def head(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),) + self.body

    @property
    def display_name(self) -> str:
        base = (self.species or self.kind or "entity").replace("_", " ").title()
        return f"{base} {self.id}"

    @property
    def is_agent(self) -> bool:
        return self.controller is not None


@dataclass(frozen=True)
class ViewEvent:
    """A single generic simulation event destined for the event log."""

    kind: str
    tick: int
    actor_id: int | None = None
    target_id: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewWorld:
    width: int
    height: int
    wrap: bool
    tick: int
    episode: int
    terrain: np.ndarray
    resources: np.ndarray
    entities: tuple[ViewEntity, ...]
    population: Mapping[str, int] = field(default_factory=dict)

    def entity_by_id(self, entity_id: int | None) -> ViewEntity | None:
        if entity_id is None:
            return None
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def entities_at(self, x: int, y: int) -> list[ViewEntity]:
        return [e for e in self.entities if e.x == x and e.y == y]

    @property
    def total_population(self) -> int:
        return len(self.entities)


@dataclass(frozen=True)
class AgentEphemera:
    """Optional, algorithm-agnostic live info about the driven agent.

    Every field is optional (``None`` when a particular algorithm does not
    expose it). The visualizer must render gracefully with partial data.
    """

    active: bool = False
    mode: str = "idle"
    episode: int = 0
    step: int = 0
    global_step: int = 0
    reward: float | None = None
    episode_return: float | None = None
    episode_length: int | None = None
    done: bool | None = None
    loss: float | None = None
    epsilon: float | None = None
    value: float | None = None
    policy: str | None = None
    architecture: str | None = None
    recurrent: bool | None = None
    recurrent_layer: str | None = None
    recurrent_active: bool | None = None
    current_action: Any = None
    current_action_name: str | None = None
    action_labels: tuple[str, ...] = ()
    action_probs: tuple[float, ...] | None = None
    q_values: tuple[float, ...] | None = None
    agent_id: int | None = None
    agent_ids: tuple[int, ...] = ()
    observation: Any = None
    observation_kind: str | None = None
    observation_shape: tuple[int, ...] | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def current_action_label(self) -> str:
        if self.current_action_name:
            return self.current_action_name
        if self.current_action is None:
            return "n/a"
        index = self.current_action
        if self.action_labels and 0 <= index < len(self.action_labels):
            return self.action_labels[index]
        return str(self.current_action)

    def metric(self, name: str, default: float | None = None) -> float | None:
        return self.metrics.get(name, default)

    def agent_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "episode": self.episode,
            "step": self.step,
            "reward": self.reward,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "loss": self.loss,
            "epsilon": self.epsilon,
            "value": self.value,
            "policy": self.policy,
            "architecture": self.architecture,
            "recurrent": self.recurrent,
            "recurrent_layer": self.recurrent_layer,
            "recurrent_active": self.recurrent_active,
            "current_action": self.current_action_label,
            "agent_id": self.agent_id,
            "agent_ids": self.agent_ids,
        }


@dataclass(frozen=True)
class ViewFrame:
    """One renderable instant of the world + the driving agent."""

    world: ViewWorld
    events: tuple[ViewEvent, ...] = ()
    population: Mapping[str, int] = field(default_factory=dict)
    episode: int = 0
    step: int = 0
    episode_return: float | None = None
    episode_length: int | None = None
    done: bool = False
    mode: str = "idle"
    agent: AgentEphemera = field(default_factory=AgentEphemera)


TERRAIN_EMPTY = 0
TERRAIN_WALL = 1
TERRAIN_WATER = 2