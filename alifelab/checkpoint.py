"""Generic checkpointing for training runs.

Checkpoints are single torch files with a stable top-level shape::

    {
        "format": "alifelab-checkpoint-v1",
        "created_at": <iso timestamp>,
        "global_step": int,
        "episode": int,
        "agent": {"agent": <class name>, "state": <agent.state_dict()>},
        "env_state": {...},             # informational, JSON-safe
        "rng": {...},                   # python/numpy/torch/cuda states
        "config": {...},                # the run configuration (JSON-safe)
        "metrics": {...},               # latest performance values
        "metadata": {...},
    }

The module is algorithm-agnostic: everything the :class:`~alifelab.core.interfaces.Agent`
protocol exposes via ``state_dict()``/``load_state_dict()`` is captured the same way,
so DQN, A2C, recurrent agents and future PPO/SSM/Transformer agents all work alike.
"""
from __future__ import annotations

import json
import pickle
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from .core.interfaces import Agent

FORMAT = "alifelab-checkpoint-v1"


def as_python(value: Any) -> Any:
    """Recursively convert numpy/torch values into plain JSON-safe python."""
    if isinstance(value, Mapping):
        return {str(k): as_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not checkpoint-safe")


def capture_rng_states() -> dict[str, Any]:
    """Snapshot Python, NumPy, Torch and CUDA global RNG states."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        try:
            states["cuda"] = torch.cuda.get_rng_state_all()
            states["cuda"] = [s.tolist() for s in states["cuda"]]
        except (RuntimeError, ValueError):
            pass
    return states


def _coerce_python_state(state: Any) -> tuple[Any, ...]:
    version, internal, gauss = state
    gauss = float(gauss) if gauss is not None else None
    return (int(version), tuple(internal), gauss)


def _coerce_numpy_state(state: Any) -> tuple[Any, ...]:
    kind, keys, pos, has_gauss, cached = state
    return (str(kind), np.asarray(keys, dtype=np.uint32), int(pos), bool(has_gauss), float(cached))


def restore_rng_states(states: Mapping[str, Any]) -> None:
    """Restore global RNG states captured by :func:`capture_rng_states`."""
    if "python" in states:
        random.setstate(_coerce_python_state(states["python"]))
    if "numpy" in states:
        np.random.set_state(_coerce_numpy_state(states["numpy"]))
    if "torch" in states:
        torch.random.set_rng_state(torch.as_tensor(states["torch"], dtype=torch.uint8))
    if "cuda" in states and torch.cuda.is_available():
        try:
            devices = [torch.as_tensor(s, dtype=torch.uint8) for s in states["cuda"]]
            torch.cuda.set_rng_state_all(devices)
        except (RuntimeError, ValueError):
            pass


def build_checkpoint(
    agent: Agent,
    *,
    config: Mapping[str, Any] | None = None,
    env_state: Mapping[str, Any] | None = None,
    global_step: int = 0,
    episode: int = 0,
    rng: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint payload from generic, algorithm-independent pieces."""
    payload: dict[str, Any] = {
        "format": FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "global_step": int(global_step),
        "episode": int(episode),
        "agent": {"agent": type(agent).__name__, "state": agent.state_dict()},
    }
    if config is not None:
        payload["config"] = as_python(config)
    if env_state is not None:
        payload["env_state"] = as_python(env_state)
    if rng is not None:
        payload["rng"] = as_python(rng)
    if metrics is not None:
        payload["metrics"] = as_python(metrics)
    if metadata is not None:
        payload["metadata"] = as_python(metadata)
    if extra is not None:
        payload["extra"] = as_python(extra)
    return payload


def save_checkpoint(path: str, payload: Mapping[str, Any]) -> str:
    """Atomically write a checkpoint payload to ``path``."""
    tmp = f"{path}.tmp-{time.time_ns()}"
    torch.save(dict(payload), tmp)
    import os

    os.replace(tmp, path)
    return path


def _torch_load(path: str, map_location: str | torch.device | None = None) -> Any:
    map_location = "cpu" if map_location is None else map_location
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError):
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(path: str, device: str | torch.device | None = None) -> dict[str, Any]:
    """Load a checkpoint payload, stripping any client payload object wrappers."""
    map_location = torch.device(device) if isinstance(device, str) else device
    return _unwrap(_torch_load(path, map_location))


def _unwrap(payload: Any) -> dict[str, Any]:
    while hasattr(payload, "keys") and "state_dict" in payload:
        payload = payload["state_dict"]
    if not hasattr(payload, "keys"):
        raise ValueError("Not an alifelab checkpoint (expected a mapping payload)")
    return dict(payload)


def apply_agent_state(agent: Agent, payload: Mapping[str, Any]) -> None:
    """Restore an agent from checkpoint data via the generic state protocol."""
    agent_state = dict(payload["agent"]["state"])
    agent.load_state_dict(agent_state)


def agent_from_checkpoint(cls: type[Agent], payload: Mapping[str, Any], **kwargs: Any) -> Agent:
    """Build an agent from a checkpoint payload (if the class offers ``from_state_dict``)."""
    if hasattr(cls, "from_state_dict"):
        return cls.from_state_dict(payload["agent"]["state"], **kwargs)  # type: ignore[no-any-return]
    raise TypeError(f"{cls.__name__} does not provide from_state_dict()")


def json_dumps_pretty(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)