"""RGB (H, W, 3) uint8 renderer of a WorldSnapshot."""
from __future__ import annotations

import numpy as np

from ..obs.common import entity_cells

_COLORS = {
    "snake": (194, 156, 14),
    "predator": (210, 40, 40),
    "prey": (40, 180, 200),
    "food": (60, 190, 70),
    "plant": (150, 230, 90),
    "rock": (140, 130, 120),
}

_TERRAIN_COLORS = {0: (25, 28, 32), 1: (70, 70, 78), 2: (40, 90, 170)}


class RGBRenderer:
    def __init__(self, scale: int = 1):
        self.scale = max(1, int(scale))

    def render(self, snapshot) -> np.ndarray:
        height, width = snapshot.layers.terrain.shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        terrain = snapshot.layers.terrain
        resources = snapshot.layers.resources
        maximum = float(resources.max()) if resources.size else 0.0
        for y in range(height):
            for x in range(width):
                canvas[y, x] = _TERRAIN_COLORS.get(int(terrain[y, x]), (25, 28, 32))
        for value, _ in np.ndenumerate(resources):
            if maximum > 0 and resources[value] > 0:
                canvas[value] = (200, 160, 40)

        for view in snapshot.entities:
            base = _COLORS.get(view.kind, (200, 200, 200))
            if view.controller:
                base = (240, 240, 240)
            for x, y in entity_cells(view):
                if 0 <= x < width and 0 <= y < height:
                    canvas[y, x] = base

        if self.scale > 1:
            return canvas.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return canvas