from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..core.registry import register_rule
from ..world.components import (
    Action,
    ActionCapabilities,
    Age,
    Body,
    Controller,
    Direction,
    Energy,
    Genome,
    Health,
    Mutation,
    Obstacle,
    Position,
    Predator,
    Reproduction,
    Size,
    Species,
)
from ..world.engine import WorldFullError
from ..world.presets import init_body
from .interfaces import MutationRule, PredationRule, ReproductionRule

if TYPE_CHECKING:
    from ..world.engine import WorldEngine
    from ..world.entity import Entity


def _species_of(entity: "Entity") -> str:
    species = entity.get(Species)
    return species.name if species is not None else entity.kind


def _pos(entity: "Entity") -> tuple[int, int]:
    pos = entity.require(Position)
    return (pos.x, pos.y)


@register_rule("predation")
class DefaultPredationRule(PredationRule):
    def apply(self, world: "WorldEngine") -> None:
        predators = [e for e in world.get_entities(Position, Predator) if e.get(Predator).enabled]
        if not predators:
            return
        prey_pool = [
            e
            for e in world.get_entities(Position)
            if (e.has(Energy) or e.has(Health)) and not e.has(Obstacle) and not e.has(Predator)
        ]
        for predator in predators:
            caps = predator.get(ActionCapabilities)
            if caps is not None and not caps.attack:
                continue
            cfg = predator.get(Predator)
            origin = _pos(predator)
            consumed = 0
            ordered = sorted(prey_pool, key=lambda t: (world.grid.chebyshev(origin, _pos(t)), t.id))
            for prey in ordered:
                if consumed >= cfg.max_per_step:
                    break
                if prey.id not in world.entities:
                    continue
                if not self._conditions_met(world, predator, prey, cfg):
                    continue
                world.emit("predation_attempt", actor_id=predator.id, target_id=prey.id)
                if cfg.attack_prob < 1.0 and float(world.rng.random()) >= cfg.attack_prob:
                    continue
                self._resolve(world, predator, prey, cfg)
                consumed += 1

    def _conditions_met(self, world: "WorldEngine", predator: "Entity", prey: "Entity", cfg: Predator) -> bool:
        dist = world.grid.chebyshev(_pos(predator), _pos(prey))
        if dist > cfg.attack_range:
            return False
        prey_species = _species_of(prey)
        predator_species = _species_of(predator)
        if prey_species == predator_species and not cfg.allow_same_species:
            return False
        if cfg.allowed_species is not None and prey_species not in cfg.allowed_species:
            return False
        if prey_species in cfg.blocked_species:
            return False
        p_size = predator.get(Size)
        t_size = prey.get(Size)
        p_size_v = p_size.value if p_size is not None else 1.0
        t_size_v = t_size.value if t_size is not None else 1.0
        if p_size_v < t_size_v * cfg.min_size_ratio:
            return False
        if cfg.min_energy_diff is not None:
            p_energy = predator.get(Energy)
            t_energy = prey.get(Energy)
            p_val = p_energy.value if p_energy is not None else 0.0
            t_val = t_energy.value if t_energy is not None else 0.0
            if p_val - t_val < cfg.min_energy_diff:
                return False
        return True

    def _resolve(self, world: "WorldEngine", predator: "Entity", prey: "Entity", cfg: Predator) -> None:
        frac = min(1.0, max(0.0, cfg.partial))
        p_energy = predator.get(Energy)
        p_health = predator.get(Health)
        p_size = predator.get(Size)
        t_energy = prey.get(Energy)
        t_health = prey.get(Health)
        t_size = prey.get(Size)

        energy_gain = 0.0
        if t_energy is not None:
            energy_gain = cfg.energy_gain + t_energy.value * frac * cfg.energy_mult
        else:
            energy_gain = cfg.energy_gain
        health_gain = cfg.health_gain
        size_gain = 0.0
        if cfg.grow:
            size_gain = cfg.size_gain
            if t_size is not None:
                size_gain += t_size.value * cfg.transfer_size

        if p_energy is not None:
            p_energy.value = min(p_energy.max, p_energy.value + energy_gain)
        if p_health is not None:
            p_health.value = min(p_health.max, p_health.value + health_gain)
        if cfg.grow and p_size is not None and size_gain:
            p_size.apply(size_gain)

        full = frac >= 1.0
        if t_energy is not None:
            t_energy.value = 0.0 if full else t_energy.value * (1.0 - frac)
        if t_health is not None:
            t_health.value = 0.0 if full else t_health.value * (1.0 - frac)

        world.emit(
            "predation",
            actor_id=predator.id,
            target_id=prey.id,
            energy_gain=energy_gain,
            health_gain=health_gain,
            size_gain=size_gain,
            killed=full,
            partial=frac,
        )
        if full:
            world.remove_entity(prey.id)


def _combine_traits(
    first: Mapping[str, float],
    second: Mapping[str, float],
    mode: str,
    alpha: float,
) -> dict[str, float]:
    keys = set(first) | set(second)
    out: dict[str, float] = {}
    for key in keys:
        va = first.get(key)
        vb = second.get(key)
        if va is None:
            out[key] = float(vb)  # type: ignore[arg-type]
        elif vb is None:
            out[key] = float(va)
        elif mode == "first":
            out[key] = float(va)
        elif mode == "second":
            out[key] = float(vb)
        elif mode == "blend":
            out[key] = float(alpha * va + (1.0 - alpha) * vb)
        else:
            out[key] = float(0.5 * (va + vb))
    return out


@register_rule("reproduction")
class DefaultReproductionRule(ReproductionRule):
    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def apply(self, world: "WorldEngine") -> None:
        candidates = [
            e for e in world.get_entities(Position, Reproduction, Action) if e.get(Reproduction).enabled
        ]
        for entity in candidates:
            cfg = entity.get(Reproduction)
            if cfg.cooldown_remaining > 0.0:
                cfg.cooldown_remaining = max(0.0, cfg.cooldown_remaining - self.dt)
        for entity in candidates:
            cfg = entity.get(Reproduction)
            if cfg.cooldown_remaining > 0.0 or not self._ready(entity, cfg):
                continue
            parents: list[Entity] = [entity]
            if cfg.mode == "sexual":
                partner = self._find_partner(world, entity, cfg, candidates)
                if partner is None:
                    world.emit("reproduction_failed", actor_id=entity.id, reason="no_partner")
                    continue
                parents.append(partner)
            elif cfg.mode != "asexual":
                world.emit("reproduction_failed", actor_id=entity.id, reason="bad_mode")
                continue
            self._reproduce(world, parents, cfg)

    def _ready(self, entity: "Entity", cfg: Reproduction) -> bool:
        caps = entity.get(ActionCapabilities)
        if caps is not None and not caps.reproduce:
            return False
        energy = entity.get(Energy)
        if energy is not None and energy.value < cfg.min_energy:
            return False
        if energy is None and cfg.min_energy > 0.0:
            return False
        age = entity.get(Age)
        age_value = age.value if age is not None else 0.0
        if age_value < cfg.min_age:
            return False
        return True

    def _find_partner(
        self,
        world: "WorldEngine",
        entity: "Entity",
        cfg: Reproduction,
        candidates: list[Entity],
    ) -> Entity | None:
        origin = _pos(entity)
        own_species = _species_of(entity)
        options: list[tuple[int, int]] = []
        by_id: dict[int, Entity] = {}
        for other in candidates:
            if other.id == entity.id:
                continue
            if other.get(Reproduction).mode != "sexual":
                continue
            if not self._ready(other, other.get(Reproduction)):
                continue
            other_species = _species_of(other)
            if cfg.compatible_species is not None:
                if other_species not in cfg.compatible_species:
                    continue
            elif other_species != own_species:
                continue
            dist = world.grid.chebyshev(origin, _pos(other))
            if dist > max(cfg.spawn_distance, 1) * 3 and dist > 6:
                continue
            options.append((dist, other.id))
            by_id[other.id] = other
        if not options:
            return None
        options.sort()
        return by_id[options[0][1]]

    def _reproduce(self, world: "WorldEngine", parents: list[Entity], cfg: Reproduction) -> None:
        costs: list[float] = []
        for index, parent in enumerate(parents):
            cost = cfg.energy_cost if index == 0 else (cfg.second_parent_cost if cfg.second_parent_cost is not None else cfg.energy_cost)
            energy = parent.get(Energy)
            if energy is None or energy.value < cost:
                world.emit("reproduction_failed", actor_id=parent.id, reason="energy")
                return
            costs.append(cost)

        first = parents[0]
        genome = first.get(Genome)
        traits: dict[str, float] | None = dict(genome.traits) if genome is not None else None
        if len(parents) == 2:
            second = parents[1]
            other_genome = second.get(Genome)
            if traits is None and other_genome is None:
                traits = None
            else:
                traits = _combine_traits(
                    traits or {},
                    dict(other_genome.traits) if other_genome is not None else {},
                    cfg.inherit,
                    cfg.blend_alpha,
                )

        for parent, cost in zip(parents, costs):
            parent.get(Energy).value -= cost
        total_cost = sum(costs)

        first_pos = _pos(first)
        first_dir = first.get(Direction)
        dx = first_dir.dx if first_dir is not None else 1
        dy = first_dir.dy if first_dir is not None else 0
        controller = first.get(Controller)

        spawned = 0
        for _ in range(max(1, cfg.offspring_count)):
            cell = world.free_cell(near=first_pos, max_dist=cfg.spawn_distance)
            if cell is None:
                world.emit("reproduction_failed", actor_id=first.id, reason="no_space")
                break
            try:
                child = world.spawn_clone(first, *cell)
            except WorldFullError:
                world.emit("reproduction_failed", actor_id=first.id, reason="no_space")
                break
            spawned += 1
            self._configure_child(world, child, parents, cfg, traits, total_cost, dx, dy, controller)

        if spawned:
            world.emit(
                "reproduce",
                actor_id=first.id,
                target_id=child.id,
                parents=[p.id for p in parents],
                count=spawned,
                mode=cfg.mode,
                energy_cost=total_cost,
            )
            for parent in parents:
                parent.get(Reproduction).cooldown_remaining = cfg.cooldown

    def _configure_child(
        self,
        world: "WorldEngine",
        child: Entity,
        parents: list[Entity],
        cfg: Reproduction,
        traits: dict[str, float] | None,
        child_energy: float,
        dx: int,
        dy: int,
        controller: Any,
    ) -> None:
        age = child.get(Age)
        if age is not None:
            age.value = 0.0
        energy = child.get(Energy)
        if energy is not None:
            energy.value = min(energy.max, child_energy)
        health = child.get(Health)
        if health is not None:
            health.value = health.max
        size = child.get(Size)
        if size is not None:
            parent_size = parents[0].get(Size)
            if parent_size is not None:
                size.value = max(size.minimum, min(size.maximum, parent_size.value * cfg.offspring_size_scale))
        if traits is not None:
            child_genome = child.get(Genome)
            if child_genome is None:
                child.add(Genome(traits=dict(traits)))
            else:
                child_genome.traits = dict(traits)
        repro = child.get(Reproduction)
        if repro is not None:
            repro.cooldown_remaining = cfg.child_cooldown if cfg.child_cooldown is not None else cfg.cooldown
        if cfg.inherit_controller and controller is not None:
            child.add(Controller(name=controller.name, policy=controller.policy))
        elif not cfg.inherit_controller:
            child.components.pop(Controller, None)
        if world.has_rule("mutation"):
            child.add(Mutation(pending=True))
        if child.has(Body) or cfg.offspring_body_length > 0:
            init_body(child, cfg.offspring_body_length, dx, dy, world.grid.normalize)


@register_rule("mutation")
class DefaultMutationRule(MutationRule):
    def __init__(
        self,
        rate: float = 0.1,
        scale: float = 0.1,
        relative: bool = True,
        bounds: Mapping[str, list[float]] | None = None,
    ):
        self.rate = float(rate)
        self.scale = float(scale)
        self.relative = bool(relative)
        self.bounds: dict[str, tuple[float, float]] = {}
        for key, value in (bounds or {}).items():
            self.bounds[key] = (float(value[0]), float(value[1]))

    def apply(self, world: "WorldEngine") -> None:
        for entity in world.get_entities(Genome):
            mutation = entity.get(Mutation)
            if mutation is None or not mutation.pending:
                continue
            mutation.pending = False
            genome = entity.get(Genome)
            changed: dict[str, float] = {}
            for trait, value in list(genome.traits.items()):
                if float(world.rng.random()) >= self.rate:
                    continue
                sigma = self.scale * abs(value) if self.relative and value != 0.0 else self.scale
                new_value = float(value + world.rng.normal(0.0, sigma))
                if trait in self.bounds:
                    lo, hi = self.bounds[trait]
                    new_value = max(lo, min(hi, new_value))
                genome.traits[trait] = new_value
                changed[trait] = new_value - float(value)
            if changed:
                world.emit("mutate", actor_id=entity.id, traits=sorted(changed), deltas=changed)
