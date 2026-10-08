from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.snapshot import EntityView, WorldSnapshot


def ego_view(snapshot: "WorldSnapshot", agent_id: int | None) -> "EntityView | None":
    if agent_id is None:
        return None
    return snapshot.by_id.get(agent_id)


def shortest_delta(snapshot: "WorldSnapshot", ax: int, ay: int, bx: int, by: int) -> tuple[int, int]:
    dx = bx - ax
    dy = by - ay
    if snapshot.wrap:
        if abs(dx) > snapshot.width // 2:
            dx -= snapshot.width if dx > 0 else -snapshot.width
        if abs(dy) > snapshot.height // 2:
            dy -= snapshot.height if dy > 0 else -snapshot.height
    return (dx, dy)


def chebyshev(snapshot: "WorldSnapshot", ax: int, ay: int, bx: int, by: int) -> int:
    dx, dy = shortest_delta(snapshot, ax, ay, bx, by)
    return max(abs(dx), abs(dy))


def normalized_energy(view: "EntityView | None") -> float:
    if view is None or view.energy is None or not view.max_energy:
        return 0.0
    return float(np.clip(view.energy / view.max_energy, 0.0, 1.0))


def normalized_health(view: "EntityView | None") -> float:
    if view is None or view.health is None or not view.max_health:
        return 0.0
    return float(np.clip(view.health / view.max_health, 0.0, 1.0))


def normalized_size(view: "EntityView | None") -> float:
    if view is None:
        return 0.0
    return float(np.clip(view.size / 8.0, 0.0, 1.0))


def normalized_age(view: "EntityView | None") -> float:
    if view is None:
        return 0.0
    return float(np.clip(view.age / 256.0, 0.0, 1.0))


def entity_cells(view: "EntityView") -> tuple[tuple[int, int], ...]:
    return ((view.x, view.y),) + view.body
