"""Replay and on-policy buffers for agent training."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..core.interfaces import Transition


class ReplayBuffer:
    """Store transitions; sample minibatches of stacked observation tensors."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._obs: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._next: deque[np.ndarray | None] = deque(maxlen=self.capacity)
        self._action: deque[Any] = deque(maxlen=self.capacity)
        self._reward: deque[float] = deque(maxlen=self.capacity)
        self._done: deque[bool] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._action)

    def clear(self) -> None:
        self._obs.clear()
        self._next.clear()
        self._action.clear()
        self._reward.clear()
        self._done.clear()

    def add(self, transition: Transition) -> None:
        self._obs.append(np.asarray(transition.observation, dtype=np.float32))
        self._next.append(
            None if transition.next_observation is None else np.asarray(transition.next_observation, dtype=np.float32)
        )
        self._action.append(transition.action)
        self._reward.append(float(transition.reward))
        self._done.append(bool(transition.terminated or transition.truncated))

    def _stack(self, values: list[np.ndarray]) -> np.ndarray:
        obs = [v for v in values]
        sample = obs[0]
        if all(v.shape == sample.shape for v in obs):
            try:
                return np.stack(obs).astype(np.float32)
            except ValueError:
                pass
        raise ValueError("Transitions in the batch have inconsistent observation shapes")

    def sample(self, batch_size: int) -> dict[str, Any]:
        batch_size = min(int(batch_size), len(self))
        indices = self.rng.integers(0, len(self), size=batch_size)

        obs = [self._obs[i] for i in indices]
        action = [self._action[i] for i in indices]
        reward = np.asarray([self._reward[i] for i in indices], dtype=np.float32)
        done = np.asarray([self._done[i] for i in indices], dtype=np.float32)
        next_obs = [self._next[i] for i in indices]

        has_next = all(v is not None for v in next_obs)
        batch: dict[str, Any] = {
            "observation": self._stack(obs),
            "action": np.asarray(action),
            "reward": reward,
            "done": done,
        }
        if has_next:
            batch["next_observation"] = self._stack([v for v in next_obs])
        else:
            batch["next_observation"] = None
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Export buffer contents + rng as plain python (weights_only-safe)."""
        def to_list(container: deque) -> list[Any]:
            return [None if v is None else v.tolist() for v in container]

        return {
            "obs": to_list(self._obs),
            "next": to_list(self._next),
            "action": [a.tolist() if isinstance(a, np.ndarray) else a for a in self._action],
            "reward": [float(r) for r in self._reward],
            "done": [bool(d) for d in self._done],
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        def to_array(value: Any, dtype: np.dtype) -> np.ndarray:
            return np.asarray(value, dtype=dtype)

        obs = [to_array(v, np.float32) for v in state["obs"]]
        raw_next = state["next"]
        next_obs = [None if v is None else to_array(v, np.float32) for v in raw_next]
        actions = [np.asarray(a) for a in state["action"]]
        rewards = [float(r) for r in state["reward"]]
        dones = [bool(d) for d in state["done"]]
        if len(obs) != len(next_obs) != len(actions) != len(rewards) != len(dones):
            raise ValueError("ReplayBuffer checkpoint contents are inconsistent")
        self.clear()
        for o, n, a, r, d in zip(obs, next_obs, actions, rewards, dones):
            self._obs.append(o)
            self._next.append(n)
            self._action.append(a)
            self._reward.append(r)
            self._done.append(d)
        self.rng.bit_generator.state = state["rng"]


class RolloutBuffer:
    """Collect full transitions for on-policy updates; drain on train()."""

    def __init__(self):
        self._transitions: list[Transition] = []

    def __len__(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def drain(self) -> list[Transition]:
        transitions = self._transitions
        self._transitions = []
        return transitions