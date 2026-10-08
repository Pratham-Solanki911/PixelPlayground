from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..core.registry import register_rule
from ..world.components import (
    Action,
    ActionCapabilities,
    Edible,
    Energy,
    Genome,
    Growth,
    Position,
)
from ..world.engine import WorldFullError
from ..world.presets import spawn_preset
from .interfaces import FoodRule, SpawnRule

if TYPE_CHECKING:
    from ..world.engine import WorldEngine


@register_rule("food")
class DefaultFoodRule(FoodRule):
    def __init__(
        self,
        categories: tuple[str, ...] | list[str] = ("food",),
        energy_per_nutrition: float = 1.0,
        use_genome_efficiency: bool = True,
        remove: bool = True,
        max_per_step: int | None = None,
        growth: bool = True,
    ):
        self.categories = tuple(categories)
        self.energy_per_nutrition = float(energy_per_nutrition)
        self.use_genome_efficiency = bool(use_genome_efficiency)
        self.remove = bool(remove)
        self.max_per_step = max_per_step
        self.growth = bool(growth)

    def apply(self, world: "WorldEngine") -> None:
        eaten = 0
        for entity in world.get_entities(Position, Action):
            caps = entity.get(ActionCapabilities)
            if caps is not None and not caps.consume:
                continue
            pos = entity.get(Position)
            for target in world.entities_at(pos.x, pos.y):
                if target.id == entity.id:
                    continue
                edible = target.get(Edible)
                if edible is None or edible.category not in self.categories:
                    continue
                efficiency = 1.0
                genome = entity.get(Genome)
                if self.use_genome_efficiency and genome is not None:
                    efficiency = max(0.05, genome.get("efficiency", 1.0))
                nutrition = edible.nutrition
                gained = 0.0
                energy = entity.get(Energy)
                if energy is not None:
                    gained = nutrition * self.energy_per_nutrition * efficiency
                    energy.value = min(energy.max, energy.value + gained)
                grew = False
                growth = entity.get(Growth)
                if self.growth and growth is not None and growth.enabled and growth.amount:
                    growth.queue()
                    grew = True
                world.emit(
                    "consume",
                    actor_id=entity.id,
                    target_id=target.id,
                    category=edible.category,
                    nutrition=nutrition,
                    energy=gained,
                    growth=grew,
                )
                if self.remove:
                    world.remove_entity(target.id)
                eaten += 1
                if self.max_per_step is not None and eaten >= self.max_per_step:
                    return


@register_rule("spawn")
class DefaultSpawnRule(SpawnRule):
    def __init__(
        self,
        spawns: list[Mapping[str, Any]] | None = None,
        preset: str | None = None,
        count: int | None = None,
        per_step: float | None = None,
        at_start: int | None = None,
        controller: str | None = None,
        params: Mapping[str, Any] | None = None,
    ):
        if spawns is None and preset is not None:
            spawns = [
                {
                    "preset": preset,
                    "count": count or 0,
                    "per_step": per_step or 0.0,
                    "at_start": at_start or 0,
                    "controller": controller,
                    "params": params,
                }
            ]
        self.specs: list[dict[str, Any]] = []
        for entry in spawns or []:
            spec = dict(entry)
            spec.setdefault("preset", spec.get("name"))
            spec.setdefault("count", 0)
            spec.setdefault("per_step", 0.0)
            spec.setdefault("at_start", 0)
            self.specs.append(spec)
        self._acc: dict[int, float] = {}
        self._first = True

    def reset(self) -> None:
        self._acc = {i: 0.0 for i in range(len(self.specs))}
        self._first = True

    def apply(self, world: "WorldEngine") -> None:
        counts = world.population_counts()
        first = self._first
        self._first = False
        for index, spec in enumerate(self.specs):
            preset = str(spec["preset"])
            target = int(spec["count"])
            current = counts.get(preset, 0)
            if first:
                at_start = min(int(spec["at_start"]), target)
                while current < at_start:
                    try:
                        spawn_preset(world, preset, spec.get("params"), spec.get("controller"))
                    except WorldFullError:
                        break
                    current += 1
                counts[preset] = current
            if current >= target:
                continue
            self._acc[index] = self._acc.get(index, 0.0) + float(spec["per_step"])
            while self._acc[index] >= 1.0 and current < target:
                try:
                    spawn_preset(world, preset, spec.get("params"), spec.get("controller"))
                except WorldFullError:
                    self._acc[index] = 0.0
                    break
                current += 1
                self._acc[index] -= 1.0
            counts[preset] = current
