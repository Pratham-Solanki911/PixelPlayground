from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .. import rules as _rules  # noqa: F401
from ..core.registry import RULES
from .engine import WorldEngine, WorldFullError
from .presets import spawn_preset
from .terrain import build_terrain

DEFAULT_RULE_ORDER: tuple[str, ...] = (
    "movement",
    "collision",
    "food",
    "growth",
    "predation",
    "reproduction",
    "mutation",
    "energy",
    "aging",
    "spawn",
    "death",
)


@dataclass
class WorldSpec:
    width: int = 16
    height: int = 16
    wrap: bool = True
    seed: int | None = None
    terrain: dict[str, Any] = field(default_factory=dict)
    rules: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)


def rules_from_config(rules_cfg: Any, preserve_order: bool = False) -> list[tuple[str, dict[str, Any]]]:
    collected: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    if rules_cfg is None:
        rules_cfg = {name: True for name in DEFAULT_RULE_ORDER}

    if isinstance(rules_cfg, Mapping):
        for name, cfg in rules_cfg.items():
            if cfg is False:
                continue
            if cfg is True:
                params: dict[str, Any] = {}
            elif isinstance(cfg, Mapping):
                params = dict(cfg)
                if not params.pop("enabled", True):
                    continue
            else:
                raise TypeError(f"Rule config for '{name}' must be bool or mapping, got {type(cfg)}")
            collected[name] = params
            ordered_keys.append(name)
        preserve_order = preserve_order or False
    elif isinstance(rules_cfg, (list, tuple)):
        for entry in rules_cfg:
            if isinstance(entry, str):
                collected[entry] = {}
                ordered_keys.append(entry)
            elif isinstance(entry, Mapping):
                name = str(entry.get("name"))
                params = dict(entry.get("params", {}))
                if not params.pop("enabled", True):
                    continue
                collected[name] = params
                ordered_keys.append(name)
            else:
                raise TypeError(f"Invalid rule entry: {entry!r}")
        preserve_order = True
    else:
        raise TypeError(f"rules config must be mapping or list, got {type(rules_cfg)}")

    if preserve_order:
        return [(name, collected[name]) for name in ordered_keys]

    result: list[tuple[str, dict[str, Any]]] = []
    for name in DEFAULT_RULE_ORDER:
        if name in collected:
            result.append((name, collected.pop(name)))
    for name in ordered_keys:
        if name in collected:
            result.append((name, collected.pop(name)))
    for name in sorted(collected):
        result.append((name, collected[name]))
    return result


def build_world(spec: WorldSpec) -> WorldEngine:
    seed = spec.seed if spec.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    terrain_rng = np.random.default_rng([seed, 17])
    terrain = build_terrain(spec.width, spec.height, spec.terrain, terrain_rng)
    rules = [RULES.create(name, **params) for name, params in spec.rules]
    entities = list(spec.entities)

    def initializer(world: WorldEngine) -> None:
        for entry in entities:
            preset = str(entry.get("preset") or entry.get("name"))
            count = int(entry.get("count", 1))
            params = entry.get("params") or {
                k: v for k, v in entry.items() if k not in {"preset", "name", "count", "controller", "params"}
            }
            controller = entry.get("controller")
            for _ in range(count):
                try:
                    spawn_preset(world, preset, params, controller)
                except WorldFullError:
                    return

    return WorldEngine(
        width=spec.width,
        height=spec.height,
        wrap=spec.wrap,
        seed=seed,
        rules=rules,
        initializer=initializer,
        terrain=terrain,
    )
