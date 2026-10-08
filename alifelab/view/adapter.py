"""Engine-free frame adapter: standardized world-state object -> ViewFrame.

The adapter accepts any object with a ``WorldSnapshot``-like structure (the
standardized interface the existing renderers already consume). It never
imports the simulation engine, RL code, Trainer, or Agent implementation --
it reads plain attributes and copies plain data.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .model import AgentEphemera, ViewEntity, ViewEvent, ViewFrame, ViewWorld


class SnapshotLike(Protocol):
    """The structural contract the visualizer expects from a world state."""

    tick: int
    episode: int
    width: int
    height: int
    wrap: bool
    entities: Any
    by_id: Any
    layers: Any
    events: Any
    population: Any


def entity_to_view(entity) -> ViewEntity:
    body = tuple((int(px), int(py)) for px, py in (getattr(entity, "body", None) or ()))
    kind = str(getattr(entity, "kind", "entity") or "entity")
    species = str(getattr(entity, "species", None) or kind)
    velocity = _int_pair(getattr(entity, "velocity", None))
    direction = _int_pair(getattr(entity, "direction", None))
    genome = _as_dict(getattr(entity, "genome", None))
    return ViewEntity(
        id=int(entity.id),
        kind=kind,
        species=species,
        x=int(entity.x),
        y=int(entity.y),
        body=body,
        alive=bool(getattr(entity, "alive", True)),
        size=_float(getattr(entity, "size", 1.0), 1.0),
        health=_opt_float(getattr(entity, "health", None)),
        max_health=_opt_float(getattr(entity, "max_health", None)),
        energy=_opt_float(getattr(entity, "energy", None)),
        max_energy=_opt_float(getattr(entity, "max_energy", None)),
        age=_float(getattr(entity, "age", 0.0), 0.0),
        lifetime=_opt_float(getattr(entity, "lifetime", None)),
        velocity=velocity,
        direction=direction,
        genome=genome,
        controller=str(entity.controller) if getattr(entity, "controller", None) else None,
        state=str(entity.state) if getattr(entity, "state", None) else None,
        action=str(entity.action) if getattr(entity, "action", None) else None,
        edible=bool(getattr(entity, "edible", False)),
        obstacle=bool(getattr(entity, "obstacle", False)),
        solid=bool(getattr(entity, "solid", False)),
        movable=bool(getattr(entity, "movable", False)),
        predator=bool(getattr(entity, "predator", False)),
    )


def event_to_view(event) -> ViewEvent:
    return ViewEvent(
        kind=str(getattr(event, "kind", "event")),
        tick=int(getattr(event, "tick", 0)),
        actor_id=_opt_int(getattr(event, "actor_id", None)),
        target_id=_opt_int(getattr(event, "target_id", None)),
        data=_as_dict(getattr(event, "data", None)),
    )


def snapshot_to_frame(
    snapshot: SnapshotLike,
    *,
    agent: AgentEphemera | None = None,
    mode: str | None = None,
    population: Mapping[str, int] | None = None,
) -> ViewFrame:
    """Convert a standardized world-state object into a :class:`ViewFrame`.

    ``agent`` is optional :class:`AgentEphemera` produced by the ambient
    driver (or ``None`` for pure world viewing / recording). ``mode`` overrides
    the agent's own mode when provided.
    """
    terrain = np.asarray(snapshot.layers.terrain)
    resources = np.asarray(snapshot.layers.resources)
    entities = tuple(entity_to_view(e) for e in snapshot.entities)
    pop = dict(getattr(snapshot, "population", None) or {})
    if population:
        pop.update({str(k): int(v) for k, v in population.items()})
    world = ViewWorld(
        width=int(snapshot.width),
        height=int(snapshot.height),
        wrap=bool(snapshot.wrap),
        tick=int(snapshot.tick),
        episode=int(snapshot.episode),
        terrain=terrain,
        resources=resources,
        entities=entities,
        population=pop,
    )
    events = tuple(event_to_view(e) for e in (getattr(snapshot, "events", None) or ()))
    ephemera = agent or AgentEphemera()
    return ViewFrame(
        world=world,
        events=events,
        population=pop,
        episode=int(snapshot.episode),
        step=int(snapshot.tick),
        episode_return=ephemera.episode_return,
        episode_length=ephemera.episode_length,
        done=bool(ephemera.done),
        mode=mode or ephemera.mode or "idle",
        agent=ephemera,
    )


def ephemera_from_diagnostics(diagnostics: Mapping[str, Any], **overrides: Any) -> AgentEphemera:
    """Map an ``Agent.diagnostics()`` dict into :class:`AgentEphemera`.

    Unknown keys are ignored; everything is optional. The driver may override
    any field via ``overrides`` (e.g. the current frame's computed reward).
    """
    fields = {
        "mode": diagnostics.get("mode"),
        "episode": diagnostics.get("episode"),
        "step": diagnostics.get("step"),
        "global_step": diagnostics.get("global_step"),
        "reward": diagnostics.get("reward"),
        "episode_return": diagnostics.get("episode_return"),
        "episode_length": diagnostics.get("episode_length"),
        "done": diagnostics.get("done"),
        "loss": diagnostics.get("loss"),
        "epsilon": diagnostics.get("epsilon"),
        "value": diagnostics.get("value"),
        "policy": diagnostics.get("policy"),
        "architecture": diagnostics.get("architecture"),
        "recurrent": diagnostics.get("recurrent"),
        "recurrent_layer": diagnostics.get("recurrent_layer"),
        "recurrent_active": diagnostics.get("recurrent_active"),
        "current_action": diagnostics.get("current_action"),
        "current_action_name": diagnostics.get("current_action_name"),
        "action_labels": _tuple(diagnostics.get("action_labels")),
        "action_probs": _float_tuple(diagnostics.get("action_probs")),
        "q_values": _float_tuple(diagnostics.get("q_values")),
        "agent_id": diagnostics.get("agent_id"),
        "agent_ids": _int_tuple(diagnostics.get("agent_ids")),
        "observation": diagnostics.get("observation"),
        "observation_kind": diagnostics.get("observation_kind"),
        "observation_shape": _int_tuple(diagnostics.get("observation_shape")),
        "metrics": _as_dict(diagnostics.get("metrics")),
    }
    fields.update({k: v for k, v in overrides.items() if v is not None})
    active = any(
        fields[key] is not None and fields[key] != ()
        for key in ("agent_id", "agent_ids", "episode", "step", "reward", "metrics")
    )
    return AgentEphemera(active=bool(active), **fields)


def _opt_float(value) -> float | None:
    return None if value is None else float(value)


def _float(value, default: float) -> float:
    return default if value is None else float(value)


def _opt_int(value) -> int | None:
    return None if value is None else int(value)


def _int_pair(value) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        a, b = value
        return (int(a), int(b))
    except (TypeError, ValueError):
        return None


def _as_dict(value) -> dict[str, Any]:
    if not value:
        return {}
    return {str(k): v for k, v in dict(value).items()}


def _tuple(value) -> tuple:
    return tuple(value) if value else ()


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(v) for v in value) if value else ()


def _float_tuple(value) -> tuple[float, ...]:
    return tuple(float(v) for v in value) if value else ()