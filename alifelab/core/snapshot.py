from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .types import Event

CODE_EMPTY = 0
CODE_WALL = 1
CODE_WATER = 2
CODE_RESOURCE = 3
CODE_EGO = 4
CODE_OTHER = 5


@dataclass(frozen=True)
class WorldLayers:
    terrain: np.ndarray
    resources: np.ndarray
    entities: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.terrain.shape


@dataclass(frozen=True)
class EntityView:
    id: int
    kind: str
    species: str
    x: int
    y: int
    direction: tuple[int, int] | None = None
    velocity: tuple[int, int] | None = None
    health: float | None = None
    max_health: float | None = None
    energy: float | None = None
    max_energy: float | None = None
    size: float = 1.0
    age: float = 0.0
    lifetime: float | None = None
    body: tuple[tuple[int, int], ...] = ()
    genome: Mapping[str, float] = field(default_factory=dict)
    controller: str | None = None
    edible: bool = False
    obstacle: bool = False
    solid: bool = False
    movable: bool = False
    predator: bool = False


@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    episode: int
    width: int
    height: int
    wrap: bool
    entities: tuple[EntityView, ...]
    by_id: Mapping[int, EntityView]
    layers: WorldLayers
    events: tuple[Event, ...]
    population: Mapping[str, int] = field(default_factory=dict)

    def get(self, entity_id: int | None) -> EntityView | None:
        if entity_id is None:
            return None
        return self.by_id.get(entity_id)

    def at(self, x: int, y: int) -> tuple[EntityView, ...]:
        return tuple(e for e in self.entities if e.x == x and e.y == y)

    @property
    def codes(self) -> np.ndarray:
        return self.layers.entities
