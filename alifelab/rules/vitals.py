from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.registry import register_rule
from ..world.components import (
    Age,
    Edible,
    Energy,
    Genome,
    Growth,
    Health,
    Position,
    Size,
)
from .interfaces import AgingRule, DeathRule, EnergyRule, GrowthRule

if TYPE_CHECKING:
    from ..world.engine import WorldEngine


@register_rule("energy")
class DefaultEnergyRule(EnergyRule):
    def __init__(
        self,
        drain: float = 0.3,
        per_size: float = 0.05,
        regen: float = 0.0,
        regen_threshold: float = 0.0,
        starvation_threshold: float = 0.0,
        starvation_damage: float = 1.0,
        terrain_damage: bool = True,
    ):
        self.drain = float(drain)
        self.per_size = float(per_size)
        self.regen = float(regen)
        self.regen_threshold = float(regen_threshold)
        self.starvation_threshold = float(starvation_threshold)
        self.starvation_damage = float(starvation_damage)
        self.terrain_damage = bool(terrain_damage)

    def apply(self, world: "WorldEngine") -> None:
        for entity in world.get_entities(Energy, Position):
            if entity.has(Edible):
                continue
            energy = entity.get(Energy)
            pos = entity.get(Position)
            size = entity.get(Size)
            genome = entity.get(Genome)
            efficiency = max(0.05, genome.get("efficiency", 1.0)) if genome is not None else 1.0
            vigor = max(0.0, genome.get("vigor", 1.0)) if genome is not None else 1.0
            drain = (self.drain + self.per_size * (size.value if size is not None else 1.0)) / efficiency
            energy.value = max(0.0, energy.value - drain)
            health = entity.get(Health)
            if health is None:
                continue
            if energy.value <= self.starvation_threshold and self.starvation_damage > 0.0:
                health.value -= self.starvation_damage
            elif self.regen > 0.0 and energy.value >= self.regen_threshold:
                health.value = min(health.max, health.value + self.regen * vigor)
            if self.terrain_damage:
                terrain = world.terrain.get(pos.x, pos.y)
                if terrain.damage:
                    health.value -= terrain.damage


@register_rule("growth")
class DefaultGrowthRule(GrowthRule):
    def __init__(self, apply_to_size: bool = True):
        self.apply_to_size = bool(apply_to_size)

    def apply(self, world: "WorldEngine") -> None:
        for entity in world.get_entities(Growth):
            growth = entity.get(Growth)
            if growth.size_pending <= 0.0:
                continue
            if not growth.enabled:
                growth.size_pending = 0.0
                continue
            gain = growth.size_pending
            growth.size_pending = 0.0
            if not self.apply_to_size:
                continue
            size = entity.get(Size)
            if size is None:
                continue
            before = size.value
            size.apply(gain)
            world.emit("grow", actor_id=entity.id, size_gain=size.value - before, size=size.value)


@register_rule("aging")
class DefaultAgingRule(AgingRule):
    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def apply(self, world: "WorldEngine") -> None:
        for entity in world.get_entities(Age):
            age = entity.get(Age)
            age.value += self.dt


@register_rule("death")
class DefaultDeathRule(DeathRule):
    def __init__(
        self,
        starve: bool = True,
        check_health: bool = True,
        check_lifetime: bool = True,
        check_size: bool = False,
        min_size: float = 0.0,
    ):
        self.starve = bool(starve)
        self.check_health = bool(check_health)
        self.check_lifetime = bool(check_lifetime)
        self.check_size = bool(check_size)
        self.min_size = float(min_size)

    def apply(self, world: "WorldEngine") -> None:
        doomed: list[tuple[int, str]] = []
        for entity in sorted(world.entities.values(), key=lambda e: e.id):
            reason: str | None = None
            health = entity.get(Health)
            energy = entity.get(Energy)
            age = entity.get(Age)
            size = entity.get(Size)
            if self.check_health and health is not None and health.value <= 0.0:
                reason = "health"
            elif self.starve and energy is not None and energy.value <= 0.0:
                reason = "starve"
            elif self.check_lifetime and age is not None and age.lifetime is not None and age.value >= age.lifetime:
                reason = "lifetime"
            elif self.check_size and size is not None and size.value <= self.min_size:
                reason = "size"
            if reason is not None:
                doomed.append((entity.id, reason))
        for entity_id, reason in doomed:
            world.emit("death", actor_id=entity_id, reason=reason)
            world.remove_entity(entity_id)
