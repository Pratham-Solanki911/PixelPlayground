from __future__ import annotations

from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import register_observation
from ..core.snapshot import EntityView, WorldSnapshot
from .common import ego_view, normalized_energy, normalized_health, normalized_size, shortest_delta
from .entity import ranked_neighbors

RELATIVE_FEATURES_PER_ENTITY = 3


@register_observation("relative")
@register_observation("relative_coordinates")
class RelativeCoordinateObservation(ObservationBuilder):
    kind = "vector"

    def __init__(self, top_k: int = 6, **cfg: Any):
        super().__init__(low=-1.0, high=1.0, **cfg)
        self.top_k = int(top_k)
        self._shape: tuple[int, ...] = (4 + self.top_k * RELATIVE_FEATURES_PER_ENTITY,)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def _type_code(self, ego: EntityView | None, view: EntityView) -> float:
        if view.edible:
            return 1.0
        if view.obstacle or (view.solid and not view.movable):
            return 2.0
        if view.predator:
            return 3.0
        if ego is not None and view.species == ego.species:
            return 4.0
        if view.movable:
            return 5.0
        return 0.0

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ego = ego_view(snapshot, agent_id)
        out = np.zeros(self._shape, dtype=np.float32)
        if ego is not None:
            out[0] = normalized_energy(ego)
            out[1] = normalized_health(ego)
            out[2] = normalized_size(ego)
            out[3] = min(1.0, ego.age / 256.0)
        else:
            out[3] = 1.0
        offset = 4
        for view in ranked_neighbors(snapshot, ego, self.top_k):
            if ego is None:
                dx = view.x / max(1, snapshot.width)
                dy = view.y / max(1, snapshot.height)
            else:
                dx, dy = shortest_delta(snapshot, ego.x, ego.y, view.x, view.y)
                dx /= max(1, snapshot.width)
                dy /= max(1, snapshot.height)
            out[offset] = float(np.clip(dx, -1.0, 1.0))
            out[offset + 1] = float(np.clip(dy, -1.0, 1.0))
            out[offset + 2] = self._type_code(ego, view) / 5.0
            offset += RELATIVE_FEATURES_PER_ENTITY
        return out
