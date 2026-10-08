from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.interfaces import Rule

if TYPE_CHECKING:
    from ..world.engine import WorldEngine


class MovementRule(Rule):
    name = "movement"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class CollisionRule(Rule):
    name = "collision"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class FoodRule(Rule):
    name = "food"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class EnergyRule(Rule):
    name = "energy"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class GrowthRule(Rule):
    name = "growth"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class PredationRule(Rule):
    name = "predation"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class ReproductionRule(Rule):
    name = "reproduction"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class DeathRule(Rule):
    name = "death"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class SpawnRule(Rule):
    name = "spawn"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class MutationRule(Rule):
    name = "mutation"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError


class AgingRule(Rule):
    name = "aging"

    def apply(self, world: "WorldEngine") -> None:
        raise NotImplementedError
