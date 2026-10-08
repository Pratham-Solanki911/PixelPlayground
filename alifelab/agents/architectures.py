from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import ARCHITECTURES, register_architecture


def build_architecture(name: str, obs_shape: tuple[int, ...], **cfg) -> "Architecture":
    cls = ARCHITECTURES.get(name)
    return cls(obs_shape, **cfg)


def token_feature_dim(obs_shape: tuple[int, ...]) -> int:
    n = len(obs_shape)
    if n == 1:
        return int(obs_shape[0])
    if n == 2:
        return int(obs_shape[1])
    if n == 3:
        return int(obs_shape[1] * obs_shape[2])
    if n == 4:
        return int(obs_shape[1] * obs_shape[2] * obs_shape[3])
    raise ValueError(f"Unsupported observation rank {n}")


def tokens_from(x: torch.Tensor, obs_shape: tuple[int, ...]) -> torch.Tensor:
    n = len(obs_shape)
    if n == 1:
        return x.unsqueeze(1)
    if n == 2:
        return x
    if n == 3:
        batch, height, width, channels = x.shape
        return x.reshape(batch, height, width * channels)
    if n == 4:
        batch, time, height, width, channels = x.shape
        return x.reshape(batch, time, height * width * channels)
    raise ValueError(f"Unsupported observation rank {n}")


class Architecture(nn.Module):
    name = "architecture"

    def __init__(self, obs_shape: tuple[int, ...], feature_dim: int = 128, **cfg):
        super().__init__()
        self.obs_shape = tuple(int(s) for s in obs_shape)
        self.feature_dim = int(feature_dim)
        self.arch_cfg = dict(cfg)

    def batch(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == len(self.obs_shape):
            return obs.unsqueeze(0)
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register_architecture("mlp")
class MLPArchitecture(Architecture):
    name = "mlp"

    def __init__(self, obs_shape, feature_dim: int = 128, hidden=(128, 128), **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        in_dim = int(np.prod(self.obs_shape)) if self.obs_shape else 1
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers.append(nn.Linear(dim, int(width)))
            layers.append(nn.ReLU())
            dim = int(width)
        layers.append(nn.Linear(dim, feature_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.batch(obs)
        return self.net(x.reshape(x.shape[0], -1))


class _ConvGridStack(nn.Module):
    def __init__(self, channels: int, pool: int = 4):
        super().__init__()
        self.pool = pool
        self.stack = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool, pool)),
        )
        self.out_dim = 64 * pool * pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


@register_architecture("cnn")
class CNNArchitecture(Architecture):
    name = "cnn"

    def __init__(self, obs_shape, feature_dim: int = 128, pool: int = 4, **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.rank = len(obs_shape)
        if self.rank == 3:
            self.grid = _ConvGridStack(int(obs_shape[2]), pool)
            self.head = nn.Linear(self.grid.out_dim, feature_dim)
        elif self.rank == 4:
            self.grid = _ConvGridStack(int(obs_shape[3]), pool)
            self.head = nn.Linear(self.grid.out_dim, feature_dim)
        elif self.rank == 2:
            channels = int(obs_shape[1])
            self.conv = nn.Sequential(
                nn.Conv1d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(64, feature_dim)
        elif self.rank == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(64, feature_dim)
        else:
            raise ValueError(f" cnn expects rank 1-4 observation, got rank {self.rank}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.batch(obs)
        if self.rank == 3:
            grid = x.permute(0, 3, 1, 2).contiguous()
            features = self.grid(grid).reshape(x.shape[0], -1)
        elif self.rank == 4:
            batch, time = x.shape[0], x.shape[1]
            grid = x.reshape(batch * time, *x.shape[2:]).permute(0, 3, 1, 2).contiguous()
            features = self.grid(grid).reshape(batch * time, -1)
            features = self.head(features).reshape(batch, time, -1).mean(dim=1)
            return F.relu(features)
        elif self.rank == 1:
            seq = x.unsqueeze(1)
            features = self.conv(seq).mean(dim=-1)
        else:
            conv_in = x.transpose(1, 2)
            features = self.conv(conv_in).mean(dim=-1)
        return F.relu(self.head(features))


class DiagonalSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.log_alpha = nn.Parameter(torch.randn(d_state) * 0.5 - 1.5)
        self.omega = nn.Parameter(torch.rand(d_state) * math.pi)
        self.log_dt = nn.Parameter(torch.zeros(d_model))
        self.b = nn.Parameter(torch.randn(d_state) * 0.1)
        self.c = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.d = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        a = -torch.exp(self.log_alpha) + 1j * self.omega
        dt = F.softplus(self.log_dt)
        dt_a = a[None, :] * dt[:, None]
        a_bar = torch.exp(dt_a)
        b_bar = (torch.exp(dt_a) - 1.0) / (a[None, :] + 1e-8) * self.b[None, :]
        c = self.c.to(torch.complex64)
        h = torch.zeros(batch, dim, self.d_state, dtype=torch.complex64, device=x.device)
        xc = x.to(torch.complex64)
        outputs = []
        for t in range(steps):
            h = a_bar[None] * h + b_bar[None] * xc[:, t].unsqueeze(-1)
            y = (h * c[None]).sum(-1).real + self.d * x[:, t]
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class _TokenProjection(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], d_model: int):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.proj = nn.Linear(token_feature_dim(obs_shape), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens_from(x, self.obs_shape))


@register_architecture("lstm")
class LSTMArchitecture(Architecture):
    name = "lstm"

    def __init__(self, obs_shape, feature_dim: int = 128, layers: int = 1, **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.proj = _TokenProjection(obs_shape, feature_dim)
        self.encoder = nn.LSTM(feature_dim, feature_dim, num_layers=layers, batch_first=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(self.batch(obs))
        out, _ = self.encoder(tokens)
        return F.relu(out[:, -1, :])


@register_architecture("gru")
class GRUArchitecture(Architecture):
    name = "gru"

    def __init__(self, obs_shape, feature_dim: int = 128, layers: int = 1, **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.proj = _TokenProjection(obs_shape, feature_dim)
        self.encoder = nn.GRU(feature_dim, feature_dim, num_layers=layers, batch_first=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(self.batch(obs))
        out, _ = self.encoder(tokens)
        return F.relu(out[:, -1, :])


@register_architecture("transformer")
class TransformerArchitecture(Architecture):
    name = "transformer"

    def __init__(
        self,
        obs_shape,
        feature_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        **cfg,
    ):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.proj = _TokenProjection(obs_shape, feature_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, feature_dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(self.batch(obs))
        length = tokens.shape[1]
        tokens = tokens + self.pos[:, :length, :]
        encoded = self.encoder(tokens)
        return F.relu(encoded.mean(dim=1))


@register_architecture("ssm")
class SSMArchitecture(Architecture):
    name = "ssm"

    def __init__(self, obs_shape, feature_dim: int = 128, d_state: int = 16, **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.proj = _TokenProjection(obs_shape, feature_dim)
        self.ssm = DiagonalSSM(feature_dim, d_state=d_state)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(self.batch(obs))
        out = self.ssm(tokens)
        return F.relu(out[:, -1, :])


class CNNTokenBackbone(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], d_model: int, pool: int = 4):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.rank = len(obs_shape)
        self.pool = pool
        if self.rank == 3:
            channels = int(obs_shape[2])
        elif self.rank == 4:
            channels = int(obs_shape[3])
        else:
            channels = 0
        if self.rank in (3, 4):
            self.conv = _ConvGridStack(channels, pool)
            self.proj = nn.Linear(64, d_model)
        elif self.rank in (1, 2):
            self.proj = nn.Linear(token_feature_dim(obs_shape), d_model)
        else:
            raise ValueError(f"cnn_* architecture expects rank 1-4 observation, got rank {self.rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 3:
            grid = x.permute(0, 3, 1, 2).contiguous()
            features = self.conv(grid)
            tokens = features.reshape(x.shape[0], 64, -1).transpose(1, 2)
            return self.proj(tokens)
        if self.rank == 4:
            batch, time = x.shape[0], x.shape[1]
            frames = x.reshape(batch * time, *x.shape[2:]).permute(0, 3, 1, 2).contiguous()
            features = self.conv(frames)
            tokens = features.reshape(batch * time, 64, -1).transpose(1, 2)
            tokens = self.proj(tokens)
            tokens = tokens.reshape(batch, time * tokens.shape[1], -1)
            return tokens
        return self.proj(tokens_from(x, self.obs_shape))


class _CNNSequenceBase(Architecture):
    def __init__(self, obs_shape, feature_dim: int = 128, pool: int = 4, **cfg):
        super().__init__(obs_shape, feature_dim, **cfg)
        self.backbone = CNNTokenBackbone(obs_shape, feature_dim, pool=pool)


@register_architecture("cnn_lstm")
class CNNLSTMArchitecture(_CNNSequenceBase):
    name = "cnn_lstm"

    def __init__(self, obs_shape, feature_dim: int = 128, pool: int = 4, layers: int = 1, **cfg):
        super().__init__(obs_shape, feature_dim, pool=pool, **cfg)
        self.encoder = nn.LSTM(feature_dim, feature_dim, num_layers=layers, batch_first=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(self.batch(obs))
        out, _ = self.encoder(tokens)
        return F.relu(out[:, -1, :])


@register_architecture("cnn_gru")
class CNNGRUArchitecture(_CNNSequenceBase):
    name = "cnn_gru"

    def __init__(self, obs_shape, feature_dim: int = 128, pool: int = 4, layers: int = 1, **cfg):
        super().__init__(obs_shape, feature_dim, pool=pool, **cfg)
        self.encoder = nn.GRU(feature_dim, feature_dim, num_layers=layers, batch_first=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(self.batch(obs))
        out, _ = self.encoder(tokens)
        return F.relu(out[:, -1, :])


@register_architecture("cnn_transformer")
class CNNTransformerArchitecture(_CNNSequenceBase):
    name = "cnn_transformer"

    def __init__(
        self,
        obs_shape,
        feature_dim: int = 128,
        pool: int = 4,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        **cfg,
    ):
        super().__init__(obs_shape, feature_dim, pool=pool, **cfg)
        self.pos = nn.Parameter(torch.zeros(1, 4096, feature_dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(self.batch(obs))
        length = tokens.shape[1]
        tokens = tokens + self.pos[:, :length, :]
        encoded = self.encoder(tokens)
        return F.relu(encoded.mean(dim=1))


@register_architecture("cnn_ssm")
class CNNSMArchitecture(_CNNSequenceBase):
    name = "cnn_ssm"

    def __init__(self, obs_shape, feature_dim: int = 128, pool: int = 4, d_state: int = 16, **cfg):
        super().__init__(obs_shape, feature_dim, pool=pool, **cfg)
        self.ssm = DiagonalSSM(feature_dim, d_state=d_state)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(self.batch(obs))
        out = self.ssm(tokens)
        return F.relu(out[:, -1, :])


ARCHITECTURE_NAMES = (
    "mlp",
    "cnn",
    "lstm",
    "gru",
    "transformer",
    "ssm",
    "cnn_lstm",
    "cnn_gru",
    "cnn_transformer",
    "cnn_ssm",
)
