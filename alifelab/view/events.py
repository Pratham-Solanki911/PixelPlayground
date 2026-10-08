"""Generic, engine-free event log + genealogy tracking.

Events are formatted from their ``kind`` plus resolved entity names. Any
future rule that emits new kinds still produces a readable line via the
generic fallback, so the log never needs to be taught about a specific game.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .model import ViewEvent, ViewWorld

_TEMPLATES: Mapping[str, str] = {
    "consume": "{actor} ate {target}",
    "predation": "{actor} attacked {target}",
    "predation_attempt": "{actor} attacked {target}",
    "death": "{actor} died ({reason})",
    "reproduce": "{actor} reproduced; born {child}",
    "reproduction_failed": "{actor} reproduction failed ({reason})",
    "birth": "{actor} gave birth to {child}",
    "spawn": "{child} spawned ({preset})",
    "grow": "{actor} grew (+{size})",
    "blocked": "{actor} blocked",
    "move": "{actor} moved",
    "mutate": "{actor} mutated",
}

_HIDDEN_KINDS = {"blocked", "move"}


@dataclass(frozen=True)
class LogEntry:
    tick: int
    message: str
    kind: str
    actor_id: int | None = None
    target_id: int | None = None


class EventLog:
    """Accumulates ``[t=N] message`` lines and resolves entity display names."""

    def __init__(self, capacity: int = 2000):
        self.capacity = max(1, int(capacity))
        self.entries: list[LogEntry] = []
        self.names: dict[int, str] = {}

    @property
    def message_count(self) -> int:
        return len(self.entries)

    def attach(self, world: ViewWorld | None) -> None:
        if world is None:
            return
        for entity in world.entities:
            self.names[entity.id] = entity.display_name

    def push_many(self, events, world: ViewWorld | None) -> int:
        self.attach(world)
        pushed = 0
        for event in events:
            if event.kind in _HIDDEN_KINDS:
                continue
            message = self._format(event)
            if message is None:
                continue
            self.entries.append(
                LogEntry(tick=event.tick, message=message, kind=event.kind, actor_id=event.actor_id, target_id=event.target_id)
            )
            pushed += 1
        if len(self.entries) > self.capacity:
            del self.entries[: len(self.entries) - self.capacity]
        return pushed

    def lines(self, limit: int | None = None) -> list[str]:
        selection = self.entries if limit is None else self.entries[-limit:]
        return [f"[t={entry.tick}] {entry.message}" for entry in selection]

    def reset(self) -> None:
        self.entries.clear()
        self.names.clear()

    # ------------------------------------------------------------------ names

    def _name(self, entity_id: int | None) -> str:
        if entity_id is None:
            return "world"
        name = self.names.get(entity_id)
        if name is not None:
            return name
        return f"Entity {entity_id}"

    def _target_name(self, event: ViewEvent) -> str:
        if event.target_id is not None:
            name = self.names.get(event.target_id)
            if name is not None:
                return name
            category = event.data.get("category")
            if category:
                return str(category)
            preset = event.data.get("preset")
            if preset:
                return str(preset)
        return "entity"

    def _format(self, event: ViewEvent) -> str | None:
        template = _TEMPLATES.get(event.kind)
        if template is None:
            return f"{event.kind}: actor={event.actor_id} target={event.target_id}"
        tokens = {
            "actor": self._name(event.actor_id),
            "target": self._target_name(event),
            "child": self._name(event.target_id),
            "reason": str(event.data.get("reason", "?")),
            "preset": str(event.data.get("preset", "entity")),
            "size": f"{float(event.data.get('size_gain', 0.0)):+.1f}",
        }
        message = template
        for key, value in tokens.items():
            message = message.replace("{" + key + "}", value)
        return message


class Genealogy:
    """Derives parent IDs + offspring counts from reproduce events."""

    def __init__(self):
        self.parents: dict[int, tuple[int, ...]] = {}
        self.offspring: dict[int, int] = {}

    def update(self, events) -> None:
        for event in events:
            if event.kind not in {"reproduce", "birth"}:
                continue
            child = event.target_id
            parent_ids = tuple(sorted(int(p) for p in (event.data.get("parents") or ())))
            if child is None:
                continue
            if parent_ids:
                self.parents[child] = parent_ids
            if event.actor_id is not None and not parent_ids:
                parent_ids = (event.actor_id,)
                self.parents[child] = parent_ids
            for parent in parent_ids:
                self.offspring[parent] = self.offspring.get(parent, 0) + 1

    def lineage(self, entity_id: int) -> tuple[tuple[int, ...], int]:
        return (self.parents.get(entity_id, ()), self.offspring.get(entity_id, 0))

    def reset(self) -> None:
        self.parents.clear()
        self.offspring.clear()