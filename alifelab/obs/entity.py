from __future__ import annotations

from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import register_observation
from ..core.snapshot import EntityView, WorldSnapshot
from .common import (
    chebyshev,
    ego_view,
    normalized_age,
    normalized_energy,
    normalized_health,
    normalized_size,
    shortest_delta,
)

ENTITY_FEATURE_DIM = 12


def entity_features(snapshot: WorldSnapshot, ego: EntityView | None, view: EntityView) -> list[float]:
    if ego is None:
        rel_x = view.x / max(1, snapshot.width)
        rel_y = view.y / max(1, snapshot.height)
    else:
        dx, dy = shortest_delta(snapshot, ego.x, ego.y, view.x, view.y)
        rel_x = dx / max(1, snapshot.width)
        rel_y = dy / max(1, snapshot.height)
    same_species = 0.0
    if ego is not None and view.species == ego.species:
        same_species = 1.0
    direction_dx = (view.direction[0] + 1) / 2.0 if view.direction else 0.5
    direction_dy = (view.direction[1] + 1) / 2.0 if view.direction else 0.5
    return [
        float(rel_x),
        float(rel_y),
        same_species,
        1.0 if view.edible else 0.0,
        1.0 if view.obstacle or (view.solid and not view.movable) else 0.0,
        1.0 if view.predator else 0.0,
        normalized_energy(view),
        normalized_health(view),
        normalized_size(view),
        normalized_age(view),
        float(direction_dx),
        float(direction_dy),
    ]


def ranked_neighbors(
    snapshot: WorldSnapshot,
    ego: EntityView | None,
    top_k: int,
) -> list[EntityView]:
    if ego is None:
        others = list(snapshot.entities)[:top_k]
        others.sort(key=lambda v: v.id)
        return others
    others = [v for v in snapshot.entities if v.id != ego.id]
    others.sort(key=lambda v: (chebyshev(snapshot, ego.x, ego.y, v.x, v.y), v.id))
    return others[:top_k]


@register_observation("entity")
@register_observation("entities")
@register_observation("entity_list")
class EntityObservation(ObservationBuilder):
    kind = "vector"

    def __init__(self, top_k: int = 8, include_self: bool = True, **cfg: Any):
        super().__init__(low=-1.0, high=1.0, **cfg)
        self.top_k = int(top_k)
        self.include_self = bool(include_self)
        blocks = self.top_k + (1 if self.include_self else 0)
        self._shape: tuple[int, ...] = (blocks * ENTITY_FEATURE_DIM,)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ego = ego_view(snapshot, agent_id)
        out = np.zeros(self._shape, dtype=np.float32)
        index = 0
        if self.include_self:
            if ego is not None:
                out[index : index + ENTITY_FEATURE_DIM] = entity_features(snapshot, ego, ego)
            index += ENTITY_FEATURE_DIM
        for view in ranked_neighbors(snapshot, ego, self.top_k):
            feats = entity_features(snapshot, ego, view)
            out[index : index + ENTITY_FEATURE_DIM] = feats
            index += ENTITY_FEATURE_DIM
        return out


@register_observation("entity_tokens")
@register_observation("tokens")
class EntityTokenObservation(ObservationBuilder):
    kind = "tokens"

    def __init__(self, top_k: int = 8, include_self: bool = True, **cfg: Any):
        super().__init__(low=-1.0, high=1.0, **cfg)
        self.top_k = int(top_k)
        self.include_self = bool(include_self)
        blocks = self.top_k + (1 if self.include_self else 0)
        self._shape: tuple[int, ...] = (blocks, ENTITY_FEATURE_DIM)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ego = ego_view(snapshot, agent_id)
        out = np.zeros(self._shape, dtype=np.float32)
        index = 0
        if self.include_self:
            if ego is not None:
                out[index] = entity_features(snapshot, ego, ego)
            index += 1
        for view in ranked_neighbors(snapshot, ego, self.top_k):
            out[index] = entity_features(snapshot, ego, view)
            index += 1
        return out
