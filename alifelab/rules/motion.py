from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.registry import register_rule
from ..core.types import MoveIntent
from ..world.components import Direction, MotionState, Position, Action
from ..world.motion import commit_move
from .interfaces import CollisionRule, MovementRule

if TYPE_CHECKING:
    from ..world.engine import WorldEngine
    from ..world.entity import Entity


@register_rule("movement")
class DefaultMovementRule(MovementRule):
    def __init__(self, face: bool = True):
        self.face = bool(face)

    def apply(self, world: "WorldEngine") -> None:
        defer = world.has_rule("collision")
        for entity in sorted(world.entities.values(), key=lambda e: e.id):
            pos = entity.get(Position)
            action = entity.get(Action)
            if pos is None or action is None:
                continue
            state = entity.get(MotionState)
            if state is None:
                state = entity.add(MotionState())
            dx = dy = 0
            intent = action.intent
            if isinstance(intent, MoveIntent):
                dx = 1 if intent.dx > 0 else -1 if intent.dx < 0 else 0
                dy = 1 if intent.dy > 0 else -1 if intent.dy < 0 else 0
            if dx == 0 and dy == 0:
                state.active = False
                state.moved = False
                continue
            if self.face:
                direction = entity.get(Direction)
                if direction is not None:
                    direction.dx, direction.dy = dx, dy
            tx, ty = world.grid.normalize(pos.x + dx, pos.y + dy)
            state.prev_x, state.prev_y = pos.x, pos.y
            state.target_x, state.target_y = tx, ty
            state.active = True
            state.moved = False
            if not defer:
                if world.blocked(tx, ty, exclude={entity.id}):
                    state.active = False
                    world.emit("blocked", actor_id=entity.id, x=tx, y=ty, reason="static")
                else:
                    commit_move(world, entity, tx, ty)
                    state.moved = True
                    state.active = False


@register_rule("collision")
class DefaultCollisionRule(CollisionRule):
    def __init__(self, allow_swap: bool = False):
        self.allow_swap = bool(allow_swap)

    def apply(self, world: "WorldEngine") -> None:
        movers: list[Entity] = []
        for entity in world.entities.values():
            state = entity.get(MotionState)
            if state is not None and state.active:
                movers.append(entity)
        if not movers:
            return
        movers.sort(key=lambda e: e.id)

        swap_ids: set[int] = set()
        if not self.allow_swap:
            for i, a in enumerate(movers):
                a_state = a.get(MotionState)
                for b in movers[i + 1 :]:
                    b_state = b.get(MotionState)
                    a_target = (a_state.target_x, a_state.target_y)
                    b_target = (b_state.target_x, b_state.target_y)
                    if a_target == (b_state.prev_x, b_state.prev_y) and b_target == (a_state.prev_x, a_state.prev_y):
                        if a_target != (a_state.prev_x, a_state.prev_y):
                            swap_ids.add(a.id)
                            swap_ids.add(b.id)

        claimed: dict[tuple[int, int], int] = {}
        for entity in movers:
            state = entity.get(MotionState)
            state.active = False
            target = (state.target_x, state.target_y)
            reason: str | None = None
            other: int | None = None
            if entity.id in swap_ids:
                reason = "head_on"
            elif target in claimed:
                reason = "occupied"
                other = claimed[target]
            elif not world.walkable(*target):
                reason = "terrain"
            elif world.solid_at(*target):
                reason = "solid"
            if reason is not None:
                state.moved = False
                world.emit(
                    "bump",
                    actor_id=entity.id,
                    x=target[0],
                    y=target[1],
                    reason=reason,
                    other=other,
                )
            else:
                claimed[target] = entity.id
                commit_move(world, entity, *target)
                state.moved = True
