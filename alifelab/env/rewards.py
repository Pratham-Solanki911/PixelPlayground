from __future__ import annotations

from typing import Any, Mapping

from ..core.interfaces import RewardContext, RewardFunction
from ..core.registry import register_reward

DEFAULT_WEIGHTS: dict[str, float] = {
    "energy": 0.1,
    "food": 1.0,
    "predation": 2.0,
    "growth": 0.0,
    "birth": 1.0,
    "survival": 0.01,
    "death": -10.0,
    "step": 0.0,
}

_EVENT_WEIGHT_KEYS = {
    "consume": "food",
    "predation": "predation",
    "grow": "growth",
    "reproduce": "birth",
}


@register_reward("composite")
class CompositeReward(RewardFunction):
    def __init__(self, weights: Mapping[str, float] | None = None, **cfg: Any):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def evaluate(self, ctx: RewardContext) -> float:
        reward = 0.0
        weights = self.weights
        if ctx.prev_energy is not None and ctx.energy is not None:
            reward += weights["energy"] * (ctx.energy - ctx.prev_energy)
        if ctx.prev_health is not None and ctx.health is not None:
            reward += weights.get("health", 0.0) * (ctx.health - ctx.prev_health)
        if ctx.prev_size is not None and ctx.size is not None:
            reward += weights.get("size", 0.0) * (ctx.size - ctx.prev_size)
        counts = {"food": 0, "predation": 0, "birth": 0, "growth": 0}
        for event in ctx.events:
            if event.actor_id != ctx.agent_id:
                continue
            key = _EVENT_WEIGHT_KEYS.get(event.kind)
            if key is not None:
                counts[key] += 1
        for key, count in counts.items():
            reward += weights.get(key, 0.0) * count
        if not ctx.died:
            reward += weights.get("survival", 0.0)
        reward += weights.get("step", 0.0)
        if ctx.died:
            reward += weights.get("death", 0.0)
        return float(reward)


@register_reward("energy")
class EnergyReward(RewardFunction):
    def __init__(self, weight: float = 0.1, **cfg: Any):
        self.weight = float(weight)

    def evaluate(self, ctx: RewardContext) -> float:
        if ctx.prev_energy is None or ctx.energy is None:
            return 0.0
        return float(self.weight * (ctx.energy - ctx.prev_energy))


@register_reward("sparse_food")
class SparseFoodReward(RewardFunction):
    def __init__(self, weight: float = 1.0, categories: tuple[str, ...] | None = None, **cfg: Any):
        self.weight = float(weight)
        self.categories = tuple(categories) if categories else None

    def evaluate(self, ctx: RewardContext) -> float:
        total = 0.0
        for event in ctx.events:
            if event.kind != "consume" or event.actor_id != ctx.agent_id:
                continue
            if self.categories and event.data.get("category") not in self.categories:
                continue
            total += self.weight
        return float(total)


@register_reward("survival")
class SurvivalReward(RewardFunction):
    def __init__(self, alive: float = 0.05, death: float = -10.0, **cfg: Any):
        self.alive = float(alive)
        self.death = float(death)

    def evaluate(self, ctx: RewardContext) -> float:
        reward = 0.0 if ctx.died else self.alive
        if ctx.died:
            reward += self.death
        return float(reward)


@register_reward("zero")
@register_reward("none")
class ZeroReward(RewardFunction):
    def evaluate(self, ctx: RewardContext) -> float:
        return 0.0
