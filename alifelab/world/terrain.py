from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DEFAULT_LEGEND: dict[str, str] = {".": "empty", "#": "wall", "~": "water", " ": "empty"}


@dataclass(frozen=True)
class TerrainType:
    name: str
    code: int
    walkable: bool = True
    damage: float = 0.0
    energy_cost: float = 0.0
    color: tuple[int, int, int] = (24, 24, 32)


def default_types() -> dict[str, TerrainType]:
    return {
        "empty": TerrainType("empty", 0, True, 0.0, 0.0, (24, 24, 32)),
        "wall": TerrainType("wall", 1, False, 0.0, 0.0, (90, 90, 100)),
        "water": TerrainType("water", 2, False, 0.0, 0.0, (30, 70, 160)),
    }


class Terrain:
    def __init__(self, width: int, height: int, types: Mapping[str, TerrainType] | None = None):
        self.width = int(width)
        self.height = int(height)
        self.types: dict[str, TerrainType] = dict(types) if types else default_types()
        self.by_code: dict[int, TerrainType] = {t.code: t for t in self.types.values()}
        self.codes = np.zeros((self.height, self.width), dtype=np.uint8)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def code_of(self, name: str) -> int:
        if name not in self.types:
            raise KeyError(f"Unknown terrain type '{name}'. Available: {sorted(self.types)}")
        return self.types[name].code

    def set(self, x: int, y: int, name: str) -> None:
        self.codes[y, x] = self.code_of(name)

    def get(self, x: int, y: int) -> TerrainType:
        return self.by_code[int(self.codes[y, x])]

    def name_at(self, x: int, y: int) -> str:
        return self.get(x, y).name

    def walkable(self, x: int, y: int) -> bool:
        return self.get(x, y).walkable

    def to_ascii(self) -> list[str]:
        name_to_char = _inverse_legend(self.types)
        rev = {t.code: name_to_char.get(t.name, "?") for t in self.types.values()}
        rows = []
        for y in range(self.height):
            rows.append("".join(rev.get(int(self.codes[y, x]), "?") for x in range(self.width)))
        return rows

    @classmethod
    def from_ascii(
        cls,
        rows: list[str],
        legend: Mapping[str, str] | None = None,
        types: Mapping[str, TerrainType] | None = None,
    ) -> "Terrain":
        legend_map = dict(DEFAULT_LEGEND)
        if legend:
            legend_map.update({str(k): str(v) for k, v in legend.items()})
        height = len(rows)
        width = max(len(r) for r in rows) if rows else 0
        terrain = cls(width, height, types)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row[:width]):
                name = legend_map.get(ch)
                if name is None:
                    raise KeyError(f"Terrain character '{ch}' not in legend")
                terrain.set(x, y, name)
        return terrain


def _inverse_legend(types: Mapping[str, TerrainType]) -> dict[str, str]:
    inv: dict[str, str] = {}
    for ch, name in DEFAULT_LEGEND.items():
        if name in types:
            inv.setdefault(name, ch)
    for name in types:
        if name not in inv:
            inv[name] = "?" if name == "empty" else name[:1]
    return inv


def make_terrain_types(spec: Mapping[str, Any] | None) -> dict[str, TerrainType]:
    types = default_types()
    for name, cfg in (spec or {}).items():
        if not isinstance(cfg, Mapping):
            continue
        base = types.get(name, TerrainType(name, len(types)))
        types[name] = TerrainType(
            name=name,
            code=int(cfg.get("code", base.code)),
            walkable=bool(cfg.get("walkable", base.walkable)),
            damage=float(cfg.get("damage", base.damage)),
            energy_cost=float(cfg.get("energy_cost", base.energy_cost)),
            color=tuple(cfg.get("color", base.color)),  # type: ignore[arg-type]
        )
    return types


def build_terrain(
    width: int,
    height: int,
    cfg: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> Terrain:
    cfg = dict(cfg or {})
    types = make_terrain_types(cfg.pop("types", None))
    terrain = Terrain(width, height, types)

    terrain_map = cfg.get("map")
    if terrain_map:
        legend = cfg.get("legend")
        return Terrain.from_ascii([str(r) for r in terrain_map], legend, types)

    layout = str(cfg.get("layout", "empty"))
    if layout == "random":
        wall_p = float(cfg.get("wall", 0.1))
        water_p = float(cfg.get("water", 0.0))
        noise = rng.random((height, width))
        terrain.codes[noise < wall_p] = types["wall"].code
        terrain.codes[(noise >= wall_p) & (noise < wall_p + water_p)] = types["water"].code
    elif layout == "rooms":
        _layout_rooms(terrain, rng, types, cfg)
    elif layout == "empty":
        pass
    else:
        raise ValueError(f"Unknown terrain layout '{layout}'")

    if cfg.get("border"):
        terrain.codes[0, :] = types["wall"].code
        terrain.codes[-1, :] = types["wall"].code
        terrain.codes[:, 0] = types["wall"].code
        terrain.codes[:, -1] = types["wall"].code
    return terrain


def _layout_rooms(
    terrain: Terrain,
    rng: np.random.Generator,
    types: dict[str, TerrainType],
    cfg: Mapping[str, Any],
) -> None:
    rooms = int(cfg.get("rooms", 4))
    wall_code = types["wall"].code
    terrain.codes[:, :] = wall_code
    for _ in range(rooms):
        w = int(rng.integers(3, max(4, terrain.width // 2)))
        h = int(rng.integers(3, max(4, terrain.height // 2)))
        x = int(rng.integers(1, max(2, terrain.width - w - 1)))
        y = int(rng.integers(1, max(2, terrain.height - h - 1)))
        terrain.codes[y : y + h, x : x + w] = types["empty"].code
    for _ in range(max(1, rooms // 2)):
        y = int(rng.integers(1, terrain.height - 1))
        terrain.codes[y, 1:-1] = types["empty"].code
