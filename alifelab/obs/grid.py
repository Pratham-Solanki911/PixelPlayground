from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import register_observation
from ..core.snapshot import EntityView, WorldSnapshot
from .common import (
    ego_view,
    entity_cells,
    normalized_age,
    normalized_energy,
    normalized_health,
    normalized_size,
)

if TYPE_CHECKING:
    pass

ChannelFn = Callable[[WorldSnapshot, EntityView | None], np.ndarray]


def _zeros(snapshot: WorldSnapshot) -> np.ndarray:
    return np.zeros((snapshot.height, snapshot.width), dtype=np.float32)


def channel_empty(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    return (snapshot.layers.terrain == 0).astype(np.float32)


def channel_wall(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    return (snapshot.layers.terrain == 1).astype(np.float32)


def channel_water(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    return (snapshot.layers.terrain == 2).astype(np.float32)


def channel_terrain(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    return snapshot.layers.terrain.astype(np.float32) / 255.0


def channel_resource(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    layer = snapshot.layers.resources
    if layer.max() <= 0:
        return _zeros(snapshot)
    return np.clip(layer / float(layer.max()), 0.0, 1.0).astype(np.float32)


def channel_food(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.edible:
            out[view.y, view.x] = 1.0
    return out


def _paint_cells(out: np.ndarray, view: EntityView) -> None:
    for x, y in entity_cells(view):
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            out[y, x] = 1.0


def channel_ego(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    if ego is not None:
        _paint_cells(out, ego)
    return out


def channel_others(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if ego is not None and view.id == ego.id:
            continue
        _paint_cells(out, view)
    return out


def channel_agents(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.controller is not None:
            _paint_cells(out, view)
    return out


def channel_obstacle(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.obstacle or (view.solid and not view.movable):
            _paint_cells(out, view)
    return out


def channel_energy(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.energy is None:
            continue
        value = normalized_energy(view)
        for x, y in entity_cells(view):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = value
    if ego is not None:
        for x, y in entity_cells(ego):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = normalized_energy(ego)
    return out


def channel_health(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.health is None:
            continue
        value = normalized_health(view)
        for x, y in entity_cells(view):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = value
    return out


def channel_size(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.kind in {"food", "plant", "rock"}:
            continue
        value = normalized_size(view)
        for x, y in entity_cells(view):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = value
    return out


def channel_age(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    for view in snapshot.entities:
        if view.age <= 0:
            continue
        value = normalized_age(view)
        for x, y in entity_cells(view):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = value
    return out


def channel_species(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    seen: dict[str, float] = {}
    counter = 1
    for view in snapshot.entities:
        if view.species not in seen:
            seen[view.species] = counter / 8.0
            counter += 1
        value = seen[view.species]
        for x, y in entity_cells(view):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = value
    return out


def channel_population(snapshot: WorldSnapshot, ego: EntityView | None) -> np.ndarray:
    out = _zeros(snapshot)
    total = max(1, sum(snapshot.population.values()))
    out[:, :] = total / 64.0
    return out


CHANNELS: dict[str, ChannelFn] = {
    "empty": channel_empty,
    "wall": channel_wall,
    "water": channel_water,
    "terrain": channel_terrain,
    "resource": channel_resource,
    "food": channel_food,
    "obstacle": channel_obstacle,
    "ego": channel_ego,
    "others": channel_others,
    "agents": channel_agents,
    "energy": channel_energy,
    "health": channel_health,
    "size": channel_size,
    "age": channel_age,
    "species": channel_species,
    "population": channel_population,
}

DEFAULT_CHANNEL_SET: tuple[str, ...] = (
    "empty",
    "wall",
    "water",
    "resource",
    "ego",
    "others",
    "energy",
    "health",
    "size",
    "age",
)


def resolve_channels(spec: int | list[str] | tuple[str, ...] | None) -> list[str]:
    if spec is None:
        return ["empty", "wall", "resource", "ego"]
    if isinstance(spec, int):
        count = max(1, min(spec, len(DEFAULT_CHANNEL_SET)))
        return list(DEFAULT_CHANNEL_SET[:count])
    names = list(spec)
    for name in names:
        if name not in CHANNELS:
            raise KeyError(f"Unknown observation channel '{name}'. Available: {sorted(CHANNELS)}")
    return names


@register_observation("grid")
@register_observation("multichannel")
@register_observation("multi_channel_grid")
class MultiChannelGridObservation(ObservationBuilder):
    kind = "grid"

    def __init__(self, channels: int | list[str] | tuple[str, ...] | None = None, **cfg):
        super().__init__(low=0.0, high=1.0, **cfg)
        self.channels = resolve_channels(channels)
        grid_size = cfg.get("grid_size")
        if grid_size:
            self._shape = (int(grid_size[0]), int(grid_size[1]), len(self.channels))
        else:
            self._shape = (0, 0, len(self.channels))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        if self._shape[0] == 0:
            self._shape = (snapshot.height, snapshot.width, len(self.channels))
            self._space = None
        ego = ego_view(snapshot, agent_id)
        planes = [CHANNELS[name](snapshot, ego) for name in self.channels]
        return np.stack(planes, axis=-1).astype(np.float32)

    def channel_index(self, name: str) -> int | None:
        try:
            return self.channels.index(name)
        except ValueError:
            return None


@register_observation("grid_codes")
class GridObservation(ObservationBuilder):
    kind = "grid"

    def __init__(self, **cfg):
        super().__init__(low=0.0, high=1.0, **cfg)
        grid_size = cfg.get("grid_size")
        if grid_size:
            self._shape = (int(grid_size[0]), int(grid_size[1]), 1)
        else:
            self._shape = (0, 0, 1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        if self._shape[0] == 0:
            self._shape = (snapshot.height, snapshot.width, 1)
            self._space = None
        ego = ego_view(snapshot, agent_id)
        codes = np.zeros((snapshot.height, snapshot.width), dtype=np.float32)
        for view in snapshot.entities:
            value = 0.4
            if view.edible:
                value = 0.35
            elif view.obstacle or (view.solid and not view.movable):
                value = 0.2
            elif view.movable:
                value = 1.0
            for x, y in entity_cells(view):
                if 0 <= x < snapshot.width and 0 <= y < snapshot.height:
                    if not (view.movable and ego is not None and view.id == ego.id):
                        codes[y, x] = value
                    else:
                        codes[y, x] = 1.0
        terrain_wall = (snapshot.layers.terrain == 1).astype(np.float32) * 0.2
        codes = np.where(codes == 0.0, terrain_wall, codes).astype(np.float32)
        if ego is not None:
            for x, y in entity_cells(ego):
                if 0 <= x < snapshot.width and 0 <= y < snapshot.height:
                    codes[y, x] = 1.0
        return codes[..., None]
