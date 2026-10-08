"""Learning agents: TorchAgent base plus DQN and actor-critic implementations.

Agents follow the core ``Agent`` protocol: ``act`` returns an env action,
``observe`` consumes a ``Transition``, ``train`` performs a gradient update and
returns metrics, and ``save``/``load`` persist network state.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.interfaces import Agent, Transition
from ..core.registry import AGENTS, register_agent
from ..core.spaces import BoxSpace, DiscreteSpace, Space
from .architectures import build_architecture
from .buffers import ReplayBuffer, RolloutBuffer
from .heads import GaussianHead, QValueHead, ValueHead, make_heads


def build_agent(name: str, observation_space: Space, action_space: Space, **cfg: Any) -> Agent:
    cls = AGENTS.get(name)
    return cls(observation_space, action_space, **cfg)


def _to_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=torch.float32)


class AgentNetwork(nn.Module):
    """Shared architecture backbone plus named output heads.

    When ``recurrent`` is ``'lstm'`` or ``'gru'`` an extra recurrent cell runs
    over the backbone features, giving the agent memory across environment steps.
    """

    def __init__(
        self,
        architecture_name: str,
        obs_shape: tuple[int, ...],
        feature_dim: int,
        head_specs: dict[str, nn.Module],
        recurrent: str | None = None,
    ):
        super().__init__()
        self.backbone = build_architecture(architecture_name, obs_shape, feature_dim=feature_dim)
        self.feature_dim = int(feature_dim)
        self.heads = nn.ModuleDict(head_specs)
        if recurrent in (None, "", "none", False):
            self.recurrent: nn.Module | None = None
        elif recurrent == "lstm":
            self.recurrent = nn.LSTM(self.feature_dim, self.feature_dim, batch_first=True)
        elif recurrent == "gru":
            self.recurrent = nn.GRU(self.feature_dim, self.feature_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown recurrent cell '{recurrent}' (choose 'lstm' or 'gru')")

    @property
    def is_recurrent(self) -> bool:
        return self.recurrent is not None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def sequence(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a (1, T, feature_dim) feature sequence with zero initial state."""
        if self.recurrent is None:
            return features[0]
        out, _ = self.recurrent(features)
        return out[0]

    def step(self, x: torch.Tensor, state=None):
        """Forward a single timestep, advancing the recurrent state if present."""
        feat = self.backbone(x)
        if self.recurrent is None:
            return feat, None
        out, new_state = self.recurrent(feat.unsqueeze(1), state)
        return out[:, -1], new_state

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {name: head(h) for name, head in self.heads.items()}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IncrementalTorchAgent(Agent):
    """Base for torch agents: builds an AgentNetwork, owns an optimizer and device."""

    name = "torch"
    update_on = "step"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        seed: int | None = None,
        architecture: str = "mlp",
        feature_dim: int = 128,
        lr: float = 3e-4,
        device: str = "auto",
        recurrent: str | None = None,
        **cfg: Any,
    ):
        super().__init__(observation_space, action_space, seed=seed, **cfg)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.architecture_name = architecture
        self.recurrent_name = None if recurrent in (None, "", "none") else str(recurrent)
        self.feature_dim = int(feature_dim)
        self.lr = float(lr)

        obs_shape = tuple(int(s) for s in observation_space.shape) if isinstance(observation_space, BoxSpace) else (int(observation_space.size),)
        head_specs = self._heads(action_space, self.feature_dim)
        self.net = AgentNetwork(architecture, obs_shape, self.feature_dim, head_specs, recurrent=self.recurrent_name).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.steps = 0
        self._last_action: Any = None
        self._init_extra()

    def _heads(self, action_space: Space, feature_dim: int) -> dict[str, nn.Module]:
        raise NotImplementedError

    def _init_extra(self) -> None:
        pass

    def _features(self, observation: np.ndarray) -> torch.Tensor:
        return self.net.features(_to_tensor(observation, self.device))

    def obs_to_tensor(self, observations: np.ndarray) -> torch.Tensor:
        return _to_tensor(observations, self.device)

    def train(self) -> dict[str, float]:
        raise NotImplementedError

    def parameter_count(self) -> int:
        return self.net.parameter_count()

    def state_dict(self) -> dict[str, Any]:
        """Capture weights, optimizer, step counter, RNG and algorithm extras."""
        return {
            "agent": type(self).__name__,
            "architecture": self.architecture_name,
            "feature_dim": self.feature_dim,
            "steps": self.steps,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "extra": self._extra_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state.get("steps", 0))
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
        self._load_extra_state(state.get("extra") or {})

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_state(self, extra: dict[str, Any]) -> None:
        pass

    @staticmethod
    def _tensor_to_plain(tensor: torch.Tensor) -> list[Any]:
        return tensor.detach().cpu().tolist()

    @staticmethod
    def _plain_to_tensor(plain: Any, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(plain, dtype=torch.float32, device=device)

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": "alifelab-agent-v1",
                "agent": type(self).__name__,
                "state": self.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(payload["state"])

    def _metrics(self, **values: float) -> dict[str, float]:
        out: dict[str, float] = {"steps": float(self.steps)}
        out.update(values)
        return out

    def _recurrent_active(self) -> bool:
        return bool(getattr(self, "_state", None) is not None)

    def diagnostics(self) -> dict[str, Any]:
        """Generic torch-agent diagnostics; subclasses enrich it."""
        return {
            "architecture": self.architecture_name,
            "recurrent": bool(self.net.is_recurrent),
            "recurrent_layer": self.recurrent_name,
            "recurrent_active": self._recurrent_active(),
            "steps": self.steps,
            "current_action": self._last_action,
        }


@register_agent("dqn")
class DQNAgent(IncrementalTorchAgent):
    """Deep Q-Network with target network, soft updates, and epsilon-greedy exploration."""

    name = "dqn"
    update_on = "step"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        train_every: int = 1,
        grad_steps: int = 1,
        **cfg: Any,
    ):
        super().__init__(observation_space, action_space, seed=seed, **cfg)
        if not isinstance(self.action_space, DiscreteSpace):
            raise ValueError("DQNAgent requires a DiscreteSpace action space")
        if self.net.is_recurrent:
            raise ValueError("DQNAgent does not support recurrent memory; use a2c_lstm / a2c_gru")
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.train_every = max(1, int(train_every))
        self.grad_steps = max(1, int(grad_steps))
        self.buffer = ReplayBuffer(int(buffer_capacity), seed=seed)
        self.target = copy.deepcopy(self.net)
        self.target.eval()
        for param in self.target.parameters():
            param.requires_grad_(False)
        self._last_obs: np.ndarray | None = None
        self._last_action: Any = None

    def _heads(self, action_space: Space, feature_dim: int) -> dict[str, nn.Module]:
        return {"q": QValueHead(feature_dim, action_space.n)}

    def act(self, observation: np.ndarray, explore: bool = True) -> Any:
        with torch.no_grad():
            q = self.net(self.obs_to_tensor(observation))["q"]
            if explore and self.rng.random() < self.epsilon:
                action = self.action_space.sample(self.rng)
            else:
                action = int(q.argmax(dim=-1).item())
            self._last_q = q.detach().cpu().numpy().reshape(-1).tolist()
        self._last_obs = np.asarray(observation, dtype=np.float32)
        self._last_action = action
        return action

    def observe(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def reset(self) -> None:
        self._last_obs = None
        self._last_action = None

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                target_param.mul_(1.0 - self.tau).add_(param, alpha=self.tau)

    def train(self) -> dict[str, float]:
        self._decay_epsilon()
        if len(self.buffer) < self.batch_size:
            return self._metrics(loss=float("nan"))
        total_loss = 0.0
        counts = 0
        for _ in range(self.grad_steps):
            batch = self.buffer.sample(self.batch_size)
            obs = self.obs_to_tensor(batch["observation"])
            actions = torch.as_tensor(batch["action"], device=self.device, dtype=torch.long)
            rewards = torch.as_tensor(batch["reward"], device=self.device, dtype=torch.float32)
            done = torch.as_tensor(batch["done"], device=self.device, dtype=torch.float32)

            q = self.net(obs)["q"].gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_q = self.target(self.obs_to_tensor(batch["next_observation"]))["q"].max(dim=-1).values
                targets = rewards + self.gamma * next_q * (1.0 - done)

            loss = F.smooth_l1_loss(q, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            counts += 1
            self.steps += 1
        self._soft_update()
        return self._metrics(loss=total_loss / max(1, counts), epsilon=self.epsilon)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _extra_state(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "target": self.target.state_dict(),
            "buffer": self.buffer.state_dict(),
        }

    def _load_extra_state(self, extra: dict[str, Any]) -> None:
        if "epsilon" in extra:
            self.epsilon = float(extra["epsilon"])
        if "target" in extra:
            self.target.load_state_dict(extra["target"])
            self.target.eval()
        if "buffer" in extra:
            self.buffer.load_state_dict(extra["buffer"])

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "policy": "epsilon-greedy (dueling Q)",
                "epsilon": self.epsilon,
                "q_values": getattr(self, "_last_q", None),
                "buffer_size": len(self.buffer),
            }
        )
        return info


@register_agent("a2c")
class A2CAgent(IncrementalTorchAgent):
    """Advantage actor-critic (A2C) with a shared backbone and discounted returns."""

    name = "a2c"
    update_on = "episode"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        seed: int | None = None,
        *,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        min_rollout: int = 8,
        **cfg: Any,
    ):
        super().__init__(observation_space, action_space, seed=seed, **cfg)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.min_rollout = max(2, int(min_rollout))
        self.buffer = RolloutBuffer()
        self._state = None

    def _heads(self, action_space: Space, feature_dim: int) -> dict[str, nn.Module]:
        heads = dict(make_heads(action_space, feature_dim))
        heads["value"] = ValueHead(feature_dim)
        return heads

    def reset(self) -> None:
        self.buffer.clear()
        self._state = None

    def observe(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def _extra_state(self) -> dict[str, Any]:
        """Serialise the carried recurrent hidden state (LSTM -> h+c, GRU -> h)."""
        if self._state is None:
            return {"recurrent": None}
        if isinstance(self._state, tuple):
            return {
                "recurrent": {
                    "h": self._tensor_to_plain(self._state[0]),
                    "c": self._tensor_to_plain(self._state[1]),
                }
            }
        if isinstance(self._state, torch.Tensor):
            return {"recurrent": {"h": self._tensor_to_plain(self._state)}}
        return {"recurrent": None}

    def _load_extra_state(self, extra: dict[str, Any]) -> None:
        recurrent = extra.get("recurrent")
        if recurrent is None:
            self._state = None
        elif isinstance(recurrent, dict) and "c" in recurrent:
            self._state = (
                self._plain_to_tensor(recurrent["h"], self.device),
                self._plain_to_tensor(recurrent["c"], self.device),
            )
        elif isinstance(recurrent, dict) and "h" in recurrent:
            self._state = self._plain_to_tensor(recurrent["h"], self.device)
        else:
            self._state = None

    def act(self, observation: np.ndarray, explore: bool = True) -> Any:
        with torch.no_grad():
            h, self._state = self.net.step(self.obs_to_tensor(observation), self._state)
            self._last_value = float(self.net.heads["value"](h).item())
            if isinstance(self.action_space, DiscreteSpace):
                logits = self.net.heads["policy"](h)
                probs = F.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, num_samples=1).item())
                self._last_probs = probs.detach().cpu().numpy().reshape(-1).tolist()
            elif isinstance(self.action_space, BoxSpace):
                action, _ = self.net.heads["policy"].sample(h)
                action = action.squeeze(0).cpu().numpy()
                self._last_probs = None
            else:
                raise ValueError(f"Unsupported action space {self.action_space!r}")
        self._last_action = action
        return action

    def train(self) -> dict[str, float]:
        if len(self.buffer) < self.min_rollout:
            return self._metrics(loss=float("nan"))
        transitions = self.buffer.drain()

        obs = np.stack([t.observation for t in transitions]).astype(np.float32)
        rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
        dones = np.asarray([1.0 if t.done else 0.0 for t in transitions], dtype=np.float32)
        obs_t = self.obs_to_tensor(obs)

        returns = np.zeros_like(rewards)
        acc = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            acc = rewards[i] + self.gamma * acc * (1.0 - dones[i])
            returns[i] = acc
        returns_t = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        features = self.net.features(obs_t)
        if self.net.is_recurrent:
            features = self.net.sequence(features.unsqueeze(0))
        values = self.net.heads["value"](features)
        advantages = returns_t - values.detach()

        if isinstance(self.action_space, DiscreteSpace):
            actions = torch.as_tensor([t.action for t in transitions], device=self.device, dtype=torch.long)
            logits = self.net.heads["policy"](features)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            policy_loss = -(log_probs * advantages).mean()
        else:
            actions = torch.as_tensor(np.stack([np.asarray(t.action) for t in transitions]).astype(np.float32), device=self.device)
            log_probs = self.net.heads["policy"].log_prob(features, actions)
            policy_loss = -(log_probs * advantages).mean()
            entropy = -log_probs.mean()

        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        return self._metrics(
            loss=float(loss.item()),
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.mean().item()),
            mean_return=float(np.mean(returns)),
        )

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "policy": "actor-critic (categorical)" if isinstance(self.action_space, DiscreteSpace) else "actor-critic (gaussian)",
                "action_probs": getattr(self, "_last_probs", None),
                "value": getattr(self, "_last_value", None),
            }
        )
        return info


@register_agent("a2c_lstm")
class A2CLSTMAgent(A2CAgent):
    """A2C with an LSTM hidden state carried across environment steps."""

    name = "a2c_lstm"

    def __init__(self, *args: Any, recurrent: str = "lstm", **kwargs: Any):
        super().__init__(*args, recurrent=recurrent, **kwargs)


@register_agent("a2c_gru")
class A2CGRUAgent(A2CAgent):
    """A2C with a GRU hidden state carried across environment steps."""

    name = "a2c_gru"

    def __init__(self, *args: Any, recurrent: str = "gru", **kwargs: Any):
        super().__init__(*args, recurrent=recurrent, **kwargs)


@register_agent("random")
class RandomAgent(Agent):
    """Uniformly random baseline agent."""

    name = "random"
    update_on = "step"

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        seed: int | None = None,
        **cfg: Any,
    ):
        super().__init__(observation_space, action_space, seed=seed, **cfg)

    def act(self, observation: np.ndarray, explore: bool = True) -> Any:
        action = self.action_space.sample(self.rng)
        self._last_action = action
        return action

    def train(self) -> dict[str, float]:
        return {"steps": 0.0}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": "uniform",
            "recurrent": False,
            "recurrent_layer": None,
            "recurrent_active": False,
            "current_action": getattr(self, "_last_action", None),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"agent": "random", "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]

    def save(self, path: str) -> None:
        torch.save({"format": "alifelab-agent-v1", "agent": "random", "state": self.state_dict()}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(payload["state"])


AGENT_NAMES = ("random", "dqn", "a2c", "a2c_lstm", "a2c_gru")