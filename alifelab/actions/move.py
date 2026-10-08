from __future__ import annotations

from typing import Any

from ..core.interfaces import ActionSpace
from ..core.registry import register_action
from ..core.spaces import BoxSpace, DiscreteSpace, Space
from ..core.types import IdleIntent, Intent, MoveIntent

NORTH_SOUTH_EAST_WEST = [(0, -1), (1, 0), (0, 1), (-1, 0)]
EIGHT_WAY = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


@register_action("discrete_move")
@register_action("grid")
class DiscreteMoveActions(ActionSpace):
    def __init__(self, n: int = 5, **cfg: Any):
        self.n = int(n)
        if self.n == 1:
            specs: list[Intent] = [IdleIntent()]
        elif self.n == 4:
            specs = [MoveIntent(dx, dy) for dx, dy in NORTH_SOUTH_EAST_WEST]
        elif self.n == 5:
            specs = [IdleIntent()] + [MoveIntent(dx, dy) for dx, dy in NORTH_SOUTH_EAST_WEST]
        elif self.n == 8:
            specs = [MoveIntent(dx, dy) for dx, dy in EIGHT_WAY]
        elif self.n == 9:
            specs = [IdleIntent()] + [MoveIntent(dx, dy) for dx, dy in EIGHT_WAY]
        else:
            raise ValueError(f"discrete_move supports n in {{1,4,5,8,9}}, got {n}")
        self.intents: list[Intent] = specs
        self._space = DiscreteSpace(len(specs))
        self._encode = {intent: i for i, intent in enumerate(specs)}

    @property
    def space(self) -> Space:
        return self._space

    @property
    def deltas(self) -> list[tuple[int, int]]:
        out = []
        for intent in self.intents:
            if isinstance(intent, MoveIntent):
                out.append((intent.dx, intent.dy))
            else:
                out.append((0, 0))
        return out

    def decode(self, action: Any) -> Intent:
        index = int(action)
        if not 0 <= index < len(self.intents):
            raise IndexError(f"Action {index} out of range for {len(self.intents)} actions")
        return self.intents[index]

    def encode(self, intent: Intent) -> int:
        return self._encode.get(intent, 0)


@register_action("continuous_move")
@register_action("continuous")
class ContinuousMoveActions(ActionSpace):
    def __init__(self, threshold: float = 0.34, **cfg: Any):
        self.threshold = float(threshold)
        self._space = BoxSpace(-1.0, 1.0, shape=(2,), dtype=float)

    @property
    def space(self) -> Space:
        return self._space

    @property
    def deltas(self) -> list[tuple[int, int]]:
        return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def decode(self, action: Any) -> Intent:
        values = [float(v) for v in action]
        dx = 0
        dy = 0
        if len(values) >= 1 and abs(values[0]) >= self.threshold:
            dx = 1 if values[0] > 0 else -1
        if len(values) >= 2 and abs(values[1]) >= self.threshold:
            dy = 1 if values[1] > 0 else -1
        if dx == 0 and dy == 0:
            return IdleIntent()
        return MoveIntent(dx, dy)


@register_action("custom")
class CustomActionSpace(ActionSpace):
    def __init__(self, actions: list[dict[str, Any]] | None = None, **cfg: Any):
        specs = actions or [{"intent": "idle"}]
        intents: list[Intent] = []
        for spec in specs:
            kind = str(spec.get("intent", "idle")).lower()
            if kind in {"idle", "stay", "noop"}:
                intents.append(IdleIntent())
            elif kind in {"move", "move_intent"}:
                intents.append(MoveIntent(int(spec.get("dx", 0)), int(spec.get("dy", 0))))
            else:
                raise ValueError(f"Unknown custom intent '{kind}'")
        self.intents = intents
        self._space = DiscreteSpace(len(intents))

    @property
    def space(self) -> Space:
        return self._space

    @property
    def deltas(self) -> list[tuple[int, int]]:
        out = []
        for intent in self.intents:
            if isinstance(intent, MoveIntent):
                out.append((intent.dx, intent.dy))
            else:
                out.append((0, 0))
        return out

    def decode(self, action: Any) -> Intent:
        return self.intents[int(action)]
