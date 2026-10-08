from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import Vec2
from .components import Body, Growth, Position, Velocity

if TYPE_CHECKING:
    from .engine import WorldEngine
    from .entity import Entity


def commit_move(world: "WorldEngine", entity: "Entity", tx: int, ty: int) -> None:
    pos = entity.require(Position)
    prev = (pos.x, pos.y)
    world.move_entity(entity, prev, (tx, ty))
    velocity = entity.get(Velocity)
    if velocity is not None:
        velocity.dx = tx - prev[0]
        velocity.dy = ty - prev[1]
    trail_body(entity, prev)


def trail_body(entity: "Entity", prev: tuple[int, int]) -> None:
    body = entity.get(Body)
    if body is None or not body.segments:
        return
    growth = entity.get(Growth)
    keep_tail = growth is not None and growth.enabled and growth.body_pending > 0
    neck = Vec2(prev[0], prev[1])
    if keep_tail:
        growth.body_pending -= 1
        body.segments = (neck,) + body.segments
    else:
        body.segments = (neck,) + body.segments[:-1]
