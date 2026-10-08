from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .spaces import BoxSpace, Space
from .snapshot import WorldSnapshot
from .types import Event, Intent

if TYPE_CHECKING:
    from ..world.engine import WorldEngine


class Rule(ABC):
    name: str = "rule"

    @abstractmethod
    def apply(self, world: "WorldEngine") -> None:
        ...

    def reset(self) -> None:
        pass


class ObservationBuilder(ABC):
    kind: str = "vector"
    preferred_by: tuple[str, ...] = ()

    def __init__(self, low: float = 0.0, high: float = 1.0, **cfg: Any):
        self.cfg = cfg
        self._low = float(low)
        self._high = float(high)
        self._space: BoxSpace | None = None

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def space(self) -> BoxSpace:
        if self._space is None:
            self._space = BoxSpace(self._low, self._high, self.shape, dtype=np.float32)
        return self._space

    @abstractmethod
    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ...

    def reset(self) -> None:
        pass

    def channel_index(self, name: str) -> int | None:
        return None


class ActionSpace(ABC):
    @property
    @abstractmethod
    def space(self) -> Space:
        ...

    @abstractmethod
    def decode(self, action: Any) -> Intent:
        ...

    def encode(self, intent: Intent) -> Any:
        raise NotImplementedError

    @property
    def deltas(self) -> list[tuple[int, int]]:
        return []


@dataclass
class RewardContext:
    tick: int
    agent_id: int | None
    events: tuple[Event, ...]
    energy: float | None = None
    prev_energy: float | None = None
    health: float | None = None
    prev_health: float | None = None
    size: float = 1.0
    prev_size: float = 1.0
    age: float = 0.0
    population: int = 0
    died: bool = False
    done: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class RewardFunction(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, ctx: RewardContext) -> float:
        ...


@dataclass
class Transition:
    observation: np.ndarray
    action: Any
    reward: float
    next_observation: np.ndarray | None
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Agent(ABC):
    architecture: str = "custom"
    update_on: str = "step"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        seed: int | None = None,
        **cfg: Any,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observation: np.ndarray, explore: bool = True) -> Any:
        ...

    def reset(self) -> None:
        pass

    def observe(self, transition: Transition) -> None:
        pass

    @abstractmethod
    def train(self) -> dict[str, float]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    def state_dict(self) -> dict[str, Any]:
        """Capture all trainable/resumable state (weights, buffers, RNGs).

        Defaults to delegating to a dedicated :meth:`load_state_dict`-compatible
        implementation; algorithms override this for full resume support.
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose a state_dict")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot restore from a state_dict")

    def parameter_count(self) -> int:
        return 0

    def diagnostics(self) -> dict[str, Any]:
        """Optional, algorithm-specific diagnostics for visualizers.

        Every key is optional. A visualizer should degrade gracefully when a
        key is missing. Known optional keys:

        - ``policy``: human-readable policy name
        - ``architecture``: network architecture name
        - ``recurrent``: whether the agent uses a recurrent cell
        - ``recurrent_layer``: ``'lstm'``, ``'gru'``, ...
        - ``recurrent_active``: whether a hidden state is being carried now
        - ``current_action``: the action produced by the last ``act()``
        - ``action_probs``: action probability vector (when available)
        - ``q_values``: per-action Q-values (when available)
        - ``value``: scalar value/critic estimate (when available)
        - ``epsilon``: exploration epsilon (when available)
        - ``steps``: number of agent update steps
        """
        return {}
