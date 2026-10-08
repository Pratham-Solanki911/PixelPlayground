from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from ..core.interfaces import ObservationBuilder, Rule
from ..core.snapshot import EntityView, WorldLayers, WorldSnapshot
from ..core.types import Event, Intent, describe_intent
from .components import (
    Action,
    Age,
    Controller,
    Direction,
    Edible,
    Energy,
    Genome,
    Health,
    MotionState,
    Obstacle,
    Predator,
    Position,
    Size,
    Solid,
    Species,
    Body,
    Velocity,
)
from .entity import Entity
from .grid import Grid
from .terrain import Terrain


class WorldFullError(RuntimeError):
    pass


class WorldEngine:
    def __init__(
        self,
        width: int,
        height: int,
        wrap: bool = True,
        seed: int | None = None,
        rules: Sequence[Rule] = (),
        initializer: Callable[["WorldEngine"], None] | None = None,
        terrain: Terrain | None = None,
        episode: int = 0,
    ):
        self.grid = Grid(width, height, wrap)
        self.terrain = terrain if terrain is not None else Terrain(width, height)
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        self.rng = np.random.default_rng(self.seed)
        self.rules: list[Rule] = list(rules)
        self.rule_names = frozenset(rule.name for rule in self.rules)
        self.initializer = initializer
        self.episode = int(episode)
        self.entities: dict[int, Entity] = {}
        self.tick = 0
        self.events: list[Event] = []
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._next_id = 1
        self.reset()

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    @property
    def wrap(self) -> bool:
        return self.grid.wrap

    def reset(self, seed: int | None = None) -> "WorldEngine":
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.entities.clear()
        self._cells.clear()
        self.tick = 0
        self.events = []
        self._next_id = 1
        for rule in self.rules:
            rule.reset()
        if self.initializer is not None:
            self.initializer(self)
        return self

    def has_rule(self, name: str) -> bool:
        return name in self.rule_names

    def step(self) -> tuple[Event, ...]:
        self.events = []
        for rule in self.rules:
            rule.apply(self)
        self.tick += 1
        return tuple(self.events)

    def emit(self, kind: str, actor_id: int | None = None, target_id: int | None = None, **data: Any) -> Event:
        event = Event(kind=kind, tick=self.tick, actor_id=actor_id, target_id=target_id, data=data)
        self.events.append(event)
        return event

    def spawn_entity(
        self,
        kind: str,
        components: Iterable[Any] = (),
        x: int | None = None,
        y: int | None = None,
    ) -> Entity:
        if x is None or y is None:
            cell = self.free_cell()
            if cell is None:
                raise WorldFullError(f"No free cell to spawn '{kind}'")
            x, y = cell
        else:
            x, y = self.grid.normalize(int(x), int(y))
        entity = Entity(self._next_id, kind)
        self._next_id += 1
        for component in components:
            entity.add(copy.deepcopy(component))
        if not entity.has(Position):
            entity.add(Position(x, y))
        self.entities[entity.id] = entity
        self._index_add(entity)
        self.emit("spawn", actor_id=entity.id, preset=kind)
        return entity

    def spawn_clone(self, parent: Entity, x: int, y: int) -> Entity:
        x, y = self.grid.normalize(int(x), int(y))
        components = []
        for component in parent.components.values():
            if isinstance(component, (Position, MotionState, Action)):
                continue
            components.append(copy.deepcopy(component))
        components.append(Action())
        return self.spawn_entity(parent.kind, components, x, y)

    def remove_entity(self, entity_id: int) -> bool:
        entity = self.entities.pop(entity_id, None)
        if entity is None:
            return False
        self._index_remove(entity)
        return True

    def get_entity(self, entity_id: int) -> Entity | None:
        return self.entities.get(entity_id)

    def get_entities(self, *types: type) -> list[Entity]:
        found = [e for e in self.entities.values() if all(e.has(t) for t in types)]
        found.sort(key=lambda e: e.id)
        return found

    def entities_at(self, x: int, y: int) -> list[Entity]:
        x, y = self.grid.normalize(int(x), int(y))
        ids = self._cells.get((x, y), ())
        return sorted((self.entities[i] for i in ids if i in self.entities), key=lambda e: e.id)

    def occupied_cells(self) -> set[tuple[int, int]]:
        return set(self._cells.keys())

    def free_cells(self) -> list[tuple[int, int]]:
        occupied = self.occupied_cells()
        return [c for c in self.grid.cells() if c not in occupied]

    def free_cell(self, near: tuple[int, int] | None = None, max_dist: int | None = None) -> tuple[int, int] | None:
        cells = self.free_cells()
        if near is not None:
            cells = [c for c in cells if max_dist is None or self.grid.chebyshev(near, c) <= max_dist]
        if not cells:
            return None
        index = int(self.rng.integers(0, len(cells)))
        return cells[index]

    def walkable(self, x: int, y: int) -> bool:
        x, y = self.grid.normalize(int(x), int(y))
        return self.terrain.walkable(x, y)

    def solid_at(self, x: int, y: int, exclude: Iterable[int] = ()) -> bool:
        excluded = set(exclude)
        for entity_id in self._cells.get((x, y), ()):  # noqa: SIM118
            if entity_id in excluded:
                continue
            entity = self.entities.get(entity_id)
            if entity is None:
                continue
            solid = entity.get(Solid)
            if solid is not None and solid.blocking:
                return True
        return False

    def blocked(self, x: int, y: int, exclude: Iterable[int] = ()) -> bool:
        if not self.walkable(x, y):
            return True
        return self.solid_at(x, y, exclude)

    def move_entity(self, entity: Entity, old: tuple[int, int], new: tuple[int, int]) -> None:
        new = self.grid.normalize(*new)
        if old == new:
            return
        pos = entity.require(Position)
        pos.x, pos.y = new
        self._cells.get(old, []).remove(entity.id)
        if not self._cells.get(old):
            self._cells.pop(old, None)
        self._cells.setdefault(new, []).append(entity.id)

    def set_intent(self, entity_id: int, intent: Intent) -> None:
        entity = self.entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Unknown entity {entity_id}")
        action = entity.get(Action)
        if action is None:
            entity.add(Action(intent))
        else:
            action.set(intent)

    def population_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities.values():
            counts[entity.kind] = counts.get(entity.kind, 0) + 1
        return counts

    def species_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities.values():
            species = entity.get(Species)
            key = species.name if species is not None else entity.kind
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_observation(self, agent_id: int | None, builder: ObservationBuilder) -> np.ndarray:
        return builder.build(self.snapshot(), agent_id)

    def snapshot(self) -> WorldSnapshot:
        views: list[EntityView] = []
        for entity in sorted(self.entities.values(), key=lambda e: e.id):
            views.append(self._view(entity))
        by_id = {v.id: v for v in views}

        terrain = self.terrain.codes.copy()
        resources = np.zeros((self.height, self.width), dtype=np.float32)
        entity_layer = np.full((self.height, self.width), -1, dtype=np.int32)

        for entity in sorted(self.entities.values(), key=lambda e: e.id):
            pos = entity.get(Position)
            if pos is None:
                continue
            edible = entity.get(Edible)
            if edible is not None:
                resources[pos.y, pos.x] += edible.nutrition

        def paint(entity: Entity) -> None:
            pos = entity.get(Position)
            if pos is None:
                return
            body = entity.get(Body)
            cells = [(pos.x, pos.y)]
            if body is not None:
                cells.extend(s.as_tuple() for s in body.segments)
            for cx, cy in cells:
                cx, cy = self.grid.normalize(cx, cy)
                entity_layer[cy, cx] = entity.id

        for entity in sorted(self.entities.values(), key=lambda e: e.id):
            if entity.has(Controller):
                continue
            paint(entity)
        for entity in sorted(self.entities.values(), key=lambda e: e.id):
            if entity.has(Controller):
                paint(entity)

        return WorldSnapshot(
            tick=self.tick,
            episode=self.episode,
            width=self.width,
            height=self.height,
            wrap=self.wrap,
            entities=tuple(views),
            by_id=by_id,
            layers=WorldLayers(terrain=terrain, resources=resources, entities=entity_layer),
            events=tuple(self.events),
            population=self.population_counts(),
        )

    def get_state(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        entities = []
        for view in snapshot.entities:
            entities.append(
                {
                    "id": view.id,
                    "kind": view.kind,
                    "species": view.species,
                    "position": [view.x, view.y],
                    "direction": list(view.direction) if view.direction else None,
                    "velocity": list(view.velocity) if view.velocity else None,
                    "energy": view.energy,
                    "health": view.health,
                    "size": view.size,
                    "age": view.age,
                    "lifetime": view.lifetime,
                    "body": [list(c) for c in view.body],
                    "genome": dict(view.genome),
                    "controller": view.controller,
                    "edible": view.edible,
                    "obstacle": view.obstacle,
                    "solid": view.solid,
                    "predator": view.predator,
                }
            )
        return {
            "tick": self.tick,
            "episode": self.episode,
            "width": self.width,
            "height": self.height,
            "wrap": self.wrap,
            "seed": self.seed,
            "terrain": self.terrain.to_ascii(),
            "population": dict(snapshot.population),
            "entities": entities,
            "events": [e.to_dict() for e in snapshot.events],
        }

    def _view(self, entity: Entity) -> EntityView:
        pos = entity.require(Position)
        direction = entity.get(Direction)
        vel = entity.get(Velocity)
        health = entity.get(Health)
        energy = entity.get(Energy)
        size = entity.get(Size)
        age = entity.get(Age)
        species = entity.get(Species)
        genome = entity.get(Genome)
        controller = entity.get(Controller)
        edible = entity.get(Edible)
        obstacle = entity.get(Obstacle)
        solid = entity.get(Solid)
        predator = entity.get(Predator)
        body = entity.get(Body)
        action = entity.get(Action)
        return EntityView(
            id=entity.id,
            kind=entity.kind,
            species=species.name if species is not None else entity.kind,
            x=pos.x,
            y=pos.y,
            direction=(direction.dx, direction.dy) if direction is not None else None,
            velocity=(vel.dx, vel.dy) if vel is not None else None,
            health=health.value if health is not None else None,
            max_health=health.max if health is not None else None,
            energy=energy.value if energy is not None else None,
            max_energy=energy.max if energy is not None else None,
            size=size.value if size is not None else 1.0,
            age=age.value if age is not None else 0.0,
            lifetime=age.lifetime if age is not None else None,
            body=tuple(s.as_tuple() for s in body.segments) if body is not None else (),
            genome=dict(genome.traits) if genome is not None else {},
            controller=controller.name if controller is not None else None,
            edible=edible is not None,
            obstacle=obstacle is not None,
            solid=solid is not None,
            movable=entity.has(Action),
            predator=predator is not None and predator.enabled,
            state=_derive_state(entity),
            action=describe_intent(action.intent) if action is not None else None,
            alive=True,
        )

    def _index_add(self, entity: Entity) -> None:
        pos = entity.get(Position)
        if pos is None:
            return
        self._cells.setdefault((pos.x, pos.y), []).append(entity.id)

    def _index_remove(self, entity: Entity) -> None:
        pos = entity.get(Position)
        if pos is None:
            return
        bucket = self._cells.get((pos.x, pos.y))
        if bucket and entity.id in bucket:
            bucket.remove(entity.id)
            if not bucket:
                self._cells.pop((pos.x, pos.y), None)


def _derive_state(entity: Entity) -> str:
    """Generic, data-driven state descriptor for visualization."""
    if entity.get(Obstacle) is not None:
        return "static"
    predator = entity.get(Predator)
    if predator is not None and predator.enabled:
        return "predator"
    if entity.get(Edible) is not None:
        return "consumable"
    controller = entity.get(Controller)
    if controller is not None:
        return f"controlled:{controller.name}"
    return "alive"
