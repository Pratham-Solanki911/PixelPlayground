"""Generic entity inspector -- works for any species, not just snakes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .events import Genealogy
from .model import ViewEntity, ViewFrame

_FMT_NUM = "{:.2f}"


@dataclass(frozen=True)
class InspectorField:
    label: str
    value: str
    key: str
    kind: str = "text"


@dataclass(frozen=True)
class InspectorReport:
    entity_id: int | None
    name: str
    species: str
    alive: bool
    fields: tuple[InspectorField, ...] = ()

    def get(self, key: str) -> str | None:
        for f in self.fields:
            if f.key == key:
                return f.value
        if key == "id":
            return "" if self.entity_id is None else str(self.entity_id)
        return None

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"name": self.name, "species": self.species, "alive": str(self.alive)}
        for f in self.fields:
            out[f.key] = f.value
        return out

    def render(self) -> str:
        lines = [f"{self.name} ({'alive' if self.alive else 'dead'})"]
        lines.extend(f"{f.label}: {f.value}" for f in self.fields)
        return "\n".join(lines)


def _num(value: float | None) -> str:
    return "n/a" if value is None else _FMT_NUM.format(value)


def _pair(value: float | None, maximum: float | None) -> str:
    if value is None:
        return "n/a"
    if maximum is None:
        return _FMT_NUM.format(value)
    return f"{_FMT_NUM.format(value)} / {_FMT_NUM.format(maximum)}"


class Inspector:
    """Builds an ordered :class:`InspectorReport` for a generic entity."""

    def __init__(self, genealogy: Genealogy | None = None):
        self.genealogy = genealogy or Genealogy()
        self.rewards: dict[int, float] = {}

    def inspect(
        self,
        frame: ViewFrame,
        entity_id: int | None,
        *,
        memory: Mapping[int, ViewEntity] | None = None,
    ) -> InspectorReport:
        if entity_id is None:
            return InspectorReport(None, "none", "none", alive=False)
        entity = frame.world.entity_by_id(entity_id)
        if entity is None and memory:
            entity = memory.get(entity_id)
        if entity is None:
            return InspectorReport(entity_id, f"Entity {entity_id}", "unknown", alive=False)

        parents, offspring = self.genealogy.lineage(entity_id)
        reward = self.rewards.get(entity_id)
        fields = [
            InspectorField("ID", str(entity.id), "id"),
            InspectorField("Species", entity.species or entity.kind or "default", "species"),
            InspectorField("Kind", entity.kind, "kind"),
            InspectorField("Position", f"({entity.x}, {entity.y})", "position"),
            InspectorField("Size", _FMT_NUM.format(entity.size), "size"),
            InspectorField("Health", _pair(entity.health, entity.max_health), "health"),
            InspectorField("Energy", _pair(entity.energy, entity.max_energy), "energy"),
            InspectorField("Age", f"{_FMT_NUM.format(entity.age)}{f' / {_FMT_NUM.format(entity.lifetime)}' if entity.lifetime else ''}", "age"),
            InspectorField("Velocity", str(entity.velocity) if entity.velocity else "n/a", "velocity"),
            InspectorField("Direction", str(entity.direction) if entity.direction else "n/a", "direction"),
            InspectorField("Status", "alive" if entity.alive else "dead", "alive"),
            InspectorField("State", entity.state or "n/a", "state"),
            InspectorField("Action", entity.action or "n/a", "action"),
            InspectorField("Controller", entity.controller or "none", "controller"),
            InspectorField("Predator", "yes" if entity.predator else "no", "predator"),
            InspectorField("Edible", "yes" if entity.edible else "no", "edible"),
            InspectorField("Obstacle", "yes" if entity.obstacle else "no", "obstacle"),
            InspectorField("Solid", "yes" if entity.solid else "no", "solid"),
            InspectorField("Movable", "yes" if entity.movable else "no", "movable"),
            InspectorField("Genome", str(dict(entity.genome)) if entity.genome else "n/a", "genome"),
            InspectorField("Parents", ", ".join(str(p) for p in parents) if parents else "n/a", "parents"),
            InspectorField("Offspring", str(offspring) if offspring else "0", "offspring"),
            InspectorField("Reward contribution", "n/a" if reward is None else _FMT_NUM.format(reward), "reward_contribution"),
        ]
        return InspectorReport(entity_id, entity.display_name, entity.species or entity.kind, alive=entity.alive, fields=tuple(fields))