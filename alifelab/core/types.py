from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Vec2:
    x: int
    y: int

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def manhattan(self, other: "Vec2") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def chebyshev(self, other: "Vec2") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))


class Intent:
    """Marker base class for action intents produced by action spaces."""


@dataclass(frozen=True, slots=True)
class IdleIntent(Intent):
    pass


@dataclass(frozen=True, slots=True)
class MoveIntent(Intent):
    dx: int
    dy: int


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    tick: int
    actor_id: int | None = None
    target_id: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tick": self.tick,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "data": dict(self.data),
        }
