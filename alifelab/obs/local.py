from __future__ import annotations

from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import OBSERVATIONS, register_observation
from ..core.snapshot import WorldSnapshot
from .common import ego_view
from .grid import CHANNELS, resolve_channels


@register_observation("local")
@register_observation("local_vision")
class LocalVisionObservation(ObservationBuilder):
    kind = "grid"

    def __init__(
        self,
        radius: int = 4,
        channels: int | list[str] | tuple[str, ...] | None = None,
        egocentric: bool = False,
        **cfg: Any,
    ):
        super().__init__(low=0.0, high=1.0, **cfg)
        self.radius = int(radius)
        self.channels = resolve_channels(channels)
        self.egocentric = bool(egocentric)
        self._shape: tuple[int, ...] = (2 * self.radius + 1, 2 * self.radius + 1, len(self.channels))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        ego = ego_view(snapshot, agent_id)
        full = np.stack(
            [CHANNELS[name](snapshot, ego) for name in self.channels],
            axis=-1,
        ).astype(np.float32)
        height, width, _ = full.shape
        if ego is None:
            return np.zeros(self._shape, dtype=np.float32)

        window = np.zeros(self._shape, dtype=np.float32)
        for oy in range(-self.radius, self.radius + 1):
            for ox in range(-self.radius, self.radius + 1):
                sx, sy = ego.x + ox, ego.y + oy
                if snapshot.wrap:
                    sx %= width
                    sy %= height
                elif not (0 <= sx < width and 0 <= sy < height):
                    cell = window[oy + self.radius, ox + self.radius, :]
                    cell[:] = 0.0
                    if "wall" in self.channels:
                        cell[self.channels.index("wall")] = 1.0
                    continue
                window[oy + self.radius, ox + self.radius, :] = full[sy, sx, :]

        if self.egocentric and ego.direction is not None:
            dx, dy = ego.direction
            k = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}.get((dx, dy))
            if k:
                window = np.rot90(window, k=k, axes=(0, 1))
        return np.ascontiguousarray(window, dtype=np.float32)

    def channel_index(self, name: str) -> int | None:
        try:
            return self.channels.index(name)
        except ValueError:
            return None
