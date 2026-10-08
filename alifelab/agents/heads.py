"""Policy/value output heads that convert architecture features into action distributions."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.spaces import BoxSpace, DiscreteSpace, Space


class Head(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CategoricalHead(Head):
    """Discrete action logits over n actions."""

    def __init__(self, feature_dim: int, n: int):
        super().__init__()
        self.n = int(n)
        self.linear = nn.Linear(int(feature_dim), self.n)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class GaussianHead(Head):
    """Diagonal-Gaussian policy for continuous Box actions (mean + log std)."""

    def __init__(self, feature_dim: int, dim: int, log_std_init: float = -0.5):
        super().__init__()
        self.dim = int(dim)
        self.mean = nn.Linear(int(feature_dim), self.dim)
        self.log_std = nn.Parameter(torch.full((self.dim,), float(log_std_init)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mean(features)

    def log_prob(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean = F.tanh(self.mean(features))
        log_std = self.log_std.clamp(max=math.log(2.0))
        variance = torch.exp(2.0 * log_std)
        return -0.5 * (((action - mean) ** 2) / variance + 2.0 * log_std + math.log(2.0 * math.pi)).sum(-1)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = F.tanh(self.mean(features))
        std = torch.exp(self.log_std.clamp(max=math.log(2.0)))
        z = torch.randn_like(mean)
        action = mean + std * z
        return action, self.log_prob(features, action)


class ValueHead(Head):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(int(feature_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class QValueHead(Head):
    """Dueling-style Q head: separate value and advantage streams."""

    def __init__(self, feature_dim: int, n: int):
        super().__init__()
        self.n = int(n)
        self.value = nn.Linear(int(feature_dim), 1)
        self.advantage = nn.Linear(int(feature_dim), self.n)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        v = self.value(features)
        a = self.advantage(features)
        return v + a - a.mean(dim=-1, keepdim=True)


def make_heads(action_space: Space, feature_dim: int, q_head: bool = False):
    """Build output heads for a Space. Returns a dict of named submodules."""
    if isinstance(action_space, DiscreteSpace):
        if q_head:
            return {"q": QValueHead(feature_dim, action_space.n)}
        return {"policy": CategoricalHead(feature_dim, action_space.n)}
    if isinstance(action_space, BoxSpace) and len(action_space.shape) == 1:
        return {"policy": GaussianHead(feature_dim, action_space.shape[0])}
    raise ValueError(f"Unsupported action space {action_space!r}")