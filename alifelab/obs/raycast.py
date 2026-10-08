from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import register_observation
from ..core.snapshot import EntityView, WorldSnapshot
from .common import ego_view


@register_observation("raycast")
@register_observation("vision")
class RaycastObservation(ObservationBuilder):
    kind = "vector"

    def __init__(self, rays: int = 8, max_range: int | None = None, fov: float = 360.0, **cfg: Any):
        super().__init__(low=0.0, high=1.0, **cfg)
        self.rays = int(rays)
        self.max_range = max_range
        self.fov = float(fov)
        self._shape: tuple[int, ...] = (self.rays * 2,)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ego = ego_view(snapshot, agent_id)
        out = np.zeros(self._shape, dtype=np.float32)
        if ego is None:
            return out
        limit = self.max_range or max(snapshot.width, snapshot.height)
        start = 0.0 if self.fov >= 360.0 else -self.fov / 2.0
        span = math.radians(self.fov) / self.rays
        for ray in range(self.rays):
            angle = math.radians(start) + span * ray + (0.0 if self.fov < 360.0 else span * 0.0)
            if self.fov >= 360.0:
                angle = (2.0 * math.pi / self.rays) * ray
            dx = math.cos(angle)
            dy = math.sin(angle)
            distance, hit = self._trace(snapshot, ego, dx, dy, limit)
            out[ray * 2] = distance
            out[ray * 2 + 1] = hit
        return out

    def _trace(
        self,
        snapshot: WorldSnapshot,
        ego: EntityView,
        dx: float,
        dy: float,
        limit: int,
    ) -> tuple[float, float]:
        by_id = snapshot.by_id
        for step in range(1, limit + 1):
            x = int(round(ego.x + dx * step))
            y = int(round(ego.y + dy * step))
            if snapshot.wrap:
                x %= snapshot.width
                y %= snapshot.height
            elif not (0 <= x < snapshot.width and 0 <= y < snapshot.height):
                return (step - 1) / max(1, limit), 1.0
            terrain_code = int(snapshot.layers.terrain[y, x])
            if terrain_code == 1:
                return step / max(1, limit), 1.0
            if terrain_code == 2:
                return step / max(1, limit), 1.0
            entity_id = int(snapshot.layers.entities[y, x])
            if entity_id >= 0:
                view = by_id.get(entity_id)
                if view is not None:
                    if view.edible:
                        return step / max(1, limit), 3.0 / 5.0
                    if view.movable:
                        return step / max(1, limit), 4.0 / 5.0
                    if view.obstacle or view.solid:
                        return step / max(1, limit), 2.0 / 5.0
            if snapshot.layers.resources[y, x] > 0:
                return step / max(1, limit), 3.0 / 5.0
        return 1.0, 0.0
