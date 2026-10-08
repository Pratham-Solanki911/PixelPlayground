from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder
from ..core.registry import OBSERVATIONS, register_observation
from ..core.snapshot import WorldSnapshot


@register_observation("temporal")
@register_observation("stack")
class TemporalObservation(ObservationBuilder):
    kind = "temporal"

    def __init__(self, base: str | dict[str, Any] | None = None, stack: int = 4, **cfg: Any):
        base_cfg: dict[str, Any] = dict(base or {"type": "entity"})
        base_type = str(base_cfg.pop("type", "entity"))
        self.base = OBSERVATIONS.create(base_type, **base_cfg)
        self.stack = int(stack)
        self._frames: deque[np.ndarray] = deque(maxlen=self.stack)
        super().__init__(low=self.base._low, high=self.base._high, **cfg)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.stack, *self.base.shape)

    def build(self, snapshot: WorldSnapshot, agent_id: int | None) -> np.ndarray:
        frame = self.base.build(snapshot, agent_id)
        if not self._frames:
            for _ in range(self.stack):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        return np.stack(list(self._frames), axis=0).astype(np.float32)

    def reset(self) -> None:
        self._frames.clear()
        self.base.reset()

    def channel_index(self, name: str) -> int | None:
        return self.base.channel_index(name)
