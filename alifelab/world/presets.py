from __future__ import annotations

from typing import Any, Mapping

from ..core.registry import PRESETS
from ..core.types import Vec2
from .components import (
    Body,
    ComponentContext,
    Position,
    build_components,
)
from .engine import WorldEngine, WorldFullError
from .entity import Entity

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "snake": {
        "species": "snake",
        "components": {
            "direction": {"dx": 1, "dy": 0},
            "body": {"length": 3},
            "growth": {"enabled": True, "amount": 1.0},
            "energy": {"value": 100.0, "max": 100.0},
            "health": {"value": 100.0, "max": 100.0},
            "age": {"lifetime": None},
            "size": {"value": 1.0},
            "sensory": {"radius": 4},
            "solid": {},
            "action": {},
            "action_capabilities": {},
            "genome": {"traits": {"efficiency": 1.0, "vigor": 1.0}},
            "reproduction": {"enabled": False},
        },
    },
    "prey": {
        "species": "prey",
        "components": {
            "direction": {"dx": 1, "dy": 0},
            "energy": {"value": 80.0, "max": 100.0},
            "health": {"value": 60.0, "max": 60.0},
            "age": {"lifetime": 400.0},
            "size": {"value": 0.8},
            "sensory": {"radius": 3},
            "solid": {},
            "action": {},
            "action_capabilities": {},
            "genome": {"traits": {"efficiency": 1.0, "vigor": 1.0}},
            "reproduction": {"enabled": False},
        },
    },
    "predator": {
        "species": "predator",
        "components": {
            "direction": {"dx": 1, "dy": 0},
            "energy": {"value": 90.0, "max": 120.0},
            "health": {"value": 90.0, "max": 90.0},
            "age": {"lifetime": 600.0},
            "size": {"value": 2.0},
            "sensory": {"radius": 4},
            "solid": {},
            "action": {},
            "action_capabilities": {},
            "genome": {"traits": {"efficiency": 1.0, "vigor": 1.0}},
            "reproduction": {"enabled": False},
            "predator": {
                "min_size_ratio": 1.5,
                "attack_range": 1,
                "attack_prob": 1.0,
                "energy_mult": 1.0,
                "transfer_size": 0.25,
                "grow": True,
                "partial": 1.0,
            },
        },
    },
    "food": {
        "species": "food",
        "components": {
            "edible": {"nutrition": 12.0, "category": "food"},
        },
    },
    "plant": {
        "species": "plant",
        "components": {
            "edible": {"nutrition": 6.0, "category": "resource"},
        },
    },
    "rock": {
        "species": "rock",
        "components": {
            "obstacle": {},
            "solid": {},
        },
    },
}

for _name, _spec in DEFAULT_PRESETS.items():
    PRESETS.register(_name, _spec)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(existing, value)
        else:
            out[key] = value
    return out


def apply_overrides(spec: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return dict(spec)
    out: dict[str, Any] = {"species": spec.get("species"), "components": dict(spec.get("components", {}))}
    comp_over = dict(overrides)
    if "species" in comp_over:
        out["species"] = comp_over.pop("species")
    if "components" in comp_over:
        comp_over = dict(comp_over.pop("components"))
    out["components"] = deep_merge(out["components"], comp_over)
    return out


def get_preset(name: str) -> dict[str, Any]:
    spec = PRESETS.get(name)
    return {
        "species": spec.get("species", name),
        "components": dict(spec.get("components", {})),
    }


def preset_with_overrides(name: str, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    return apply_overrides(get_preset(name), overrides)


def init_body(entity: Entity, length: int, dx: int, dy: int, normalize) -> None:
    if length <= 0:
        entity.components.pop(Body, None)
        return
    pos = entity.require(Position)
    segments: list[Vec2] = []
    x, y = pos.x, pos.y
    for _ in range(length):
        x -= dx
        y -= dy
        x, y = normalize(x, y)
        segments.append(Vec2(x, y))
    body = entity.get(Body)
    if body is None:
        entity.add(Body(segments=tuple(segments)))
    else:
        body.segments = tuple(segments)


def spawn_preset(
    world: WorldEngine,
    name: str,
    params: Mapping[str, Any] | None = None,
    controller: str | None = None,
    position: tuple[int, int] | None = None,
) -> Entity:
    spec = apply_overrides(get_preset(name), params)
    components_spec: dict[str, Any] = dict(spec.get("components", {}))
    if controller:
        components_spec["controller"] = {"name": controller}

    if position is None:
        cell = world.free_cell()
        if cell is None:
            raise WorldFullError(f"No free cell to spawn preset '{name}'")
    else:
        cell = world.grid.normalize(int(position[0]), int(position[1]))

    direction_spec = components_spec.get("direction") or {}
    dx = int(direction_spec.get("dx", 1)) if isinstance(direction_spec, Mapping) else 1
    dy = int(direction_spec.get("dy", 0)) if isinstance(direction_spec, Mapping) else 0
    if dx == 0 and dy == 0:
        dx = 1

    ctx = ComponentContext(
        x=cell[0],
        y=cell[1],
        dx=dx,
        dy=dy,
        kind=name,
        species=str(spec.get("species", name)),
        normalize=world.grid.normalize,
    )
    components = build_components(components_spec, ctx)
    if not any(isinstance(c, Position) for c in components):
        components.insert(0, Position(cell[0], cell[1]))
    return world.spawn_entity(name, components, cell[0], cell[1])
