"""Live population panel data, derived generically from frames + events."""
from __future__ import annotations

from typing import Mapping

from .model import ViewFrame


class PopulationTracker:
    """Tracks live totals/species breakdowns plus cumulative event counters."""

    def __init__(self):
        self.by_species: dict[str, int] = {}
        self.total = 0
        self.births = 0
        self.deaths = 0
        self.predation = 0
        self.attack_attempts = 0
        self.reproduction = 0
        self.food_consumed = 0
        self.spawns = 0
        self.mutations = 0
        self.last_deltas: dict[str, int] = {}

    def update(self, frame: ViewFrame) -> None:
        by_species: dict[str, int] = {}
        for entity in frame.world.entities:
            key = entity.species or entity.kind or "entity"
            by_species[key] = by_species.get(key, 0) + 1
        self.by_species = by_species
        self.total = frame.world.total_population

        deltas: dict[str, int] = {
            "births": 0,
            "deaths": 0,
            "predation": 0,
            "reproduction": 0,
            "food_consumed": 0,
            "spawns": 0,
            "mutations": 0,
            "attack_attempts": 0,
        }
        for event in frame.events:
            kind = event.kind
            if kind == "reproduce" or kind == "birth":
                deltas["births"] += int(event.data.get("count", 1))
                deltas["reproduction"] += int(event.data.get("count", 1))
            elif kind == "death":
                deltas["deaths"] += 1
            elif kind == "predation":
                deltas["predation"] += 1
            elif kind == "predation_attempt":
                deltas["attack_attempts"] += 1
            elif kind == "consume":
                deltas["food_consumed"] += 1
            elif kind == "spawn":
                deltas["spawns"] += 1
            elif kind == "mutate":
                deltas["mutations"] += 1
        self.last_deltas = deltas
        self.births += deltas["births"]
        self.deaths += deltas["deaths"]
        self.predation += deltas["predation"]
        self.attack_attempts += deltas["attack_attempts"]
        self.reproduction += deltas["reproduction"]
        self.food_consumed += deltas["food_consumed"]
        self.spawns += deltas["spawns"]
        self.mutations += deltas["mutations"]

    def summary(self) -> dict[str, object]:
        return {
            "total": self.total,
            "by_species": dict(self.by_species),
            "births": self.births,
            "deaths": self.deaths,
            "predation": self.predation,
            "attack_attempts": self.attack_attempts,
            "reproduction": self.reproduction,
            "food_consumed": self.food_consumed,
            "spawns": self.spawns,
            "mutations": self.mutations,
            "last_deltas": dict(self.last_deltas),
        }

    def reset(self) -> None:
        self.by_species = {}
        self.total = 0
        self.births = self.deaths = 0
        self.predation = self.attack_attempts = 0
        self.reproduction = self.food_consumed = self.spawns = self.mutations = 0
        self.last_deltas = {}