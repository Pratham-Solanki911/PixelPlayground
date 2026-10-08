from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.registry import COMPONENTS, register_component
from ..core.types import IdleIntent, Intent, Vec2


class Component:
    pass


@register_component("position")
@dataclass
class Position(Component):
    x: int = 0
    y: int = 0

    @property
    def vec(self) -> Vec2:
        return Vec2(self.x, self.y)


@register_component("direction")
@dataclass
class Direction(Component):
    dx: int = 1
    dy: int = 0


@register_component("velocity")
@dataclass
class Velocity(Component):
    dx: int = 0
    dy: int = 0


@register_component("size")
@dataclass
class Size(Component):
    value: float = 1.0
    minimum: float = 0.1
    maximum: float = 100.0

    def apply(self, delta: float) -> None:
        self.value = max(self.minimum, min(self.maximum, self.value + delta))


@register_component("health")
@dataclass
class Health(Component):
    value: float = 100.0
    max: float = 100.0


@register_component("energy")
@dataclass
class Energy(Component):
    value: float = 100.0
    max: float = 100.0


@register_component("age")
@dataclass
class Age(Component):
    value: float = 0.0
    lifetime: float | None = None


@register_component("species")
@dataclass
class Species(Component):
    name: str = "default"


@register_component("solid")
@dataclass
class Solid(Component):
    blocking: bool = True


@register_component("obstacle")
@dataclass
class Obstacle(Component):
    pass


@register_component("edible")
@dataclass
class Edible(Component):
    nutrition: float = 10.0
    category: str = "food"


@register_component("controller")
@dataclass
class Controller(Component):
    name: str = "agent"
    policy: str | None = None


@register_component("action")
@dataclass
class Action(Component):
    intent: Intent = field(default_factory=IdleIntent)

    def set(self, intent: Intent) -> None:
        self.intent = intent


@register_component("motion_state")
@dataclass
class MotionState(Component):
    prev_x: int = 0
    prev_y: int = 0
    target_x: int = 0
    target_y: int = 0
    active: bool = False
    moved: bool = False

    @property
    def target(self) -> tuple[int, int]:
        return (self.target_x, self.target_y)

    @property
    def prev(self) -> tuple[int, int]:
        return (self.prev_x, self.prev_y)


@register_component("growth")
@dataclass
class Growth(Component):
    enabled: bool = True
    amount: float = 1.0
    body_pending: int = 0
    size_pending: float = 0.0

    def queue(self, amount: float | None = None) -> None:
        amt = self.amount if amount is None else amount
        if not self.enabled:
            return
        self.body_pending += int(amt)
        self.size_pending += float(amt)


@register_component("reproduction")
@dataclass
class Reproduction(Component):
    enabled: bool = True
    mode: str = "asexual"
    cooldown: float = 25.0
    cooldown_remaining: float = 0.0
    min_energy: float = 50.0
    min_age: float = 10.0
    offspring_count: int = 1
    spawn_distance: int = 2
    energy_cost: float = 30.0
    second_parent_cost: float | None = None
    inherit: str = "mean"
    blend_alpha: float = 0.5
    offspring_size_scale: float = 1.0
    offspring_body_length: int = 1
    inherit_controller: bool = True
    compatible_species: tuple[str, ...] | None = None
    child_cooldown: float | None = None


@register_component("genome")
@dataclass
class Genome(Component):
    traits: dict[str, float] = field(default_factory=dict)

    def get(self, trait: str, default: float = 1.0) -> float:
        return float(self.traits.get(trait, default))


@register_component("mutation")
@dataclass
class Mutation(Component):
    pending: bool = False


@register_component("predator")
@dataclass
class Predator(Component):
    enabled: bool = True
    min_size_ratio: float = 1.5
    min_energy_diff: float | None = None
    allowed_species: tuple[str, ...] | None = None
    blocked_species: tuple[str, ...] = ()
    allow_same_species: bool = False
    attack_range: int = 1
    attack_prob: float = 1.0
    energy_mult: float = 1.0
    energy_gain: float = 0.0
    health_gain: float = 0.0
    size_gain: float = 0.0
    transfer_size: float = 0.0
    grow: bool = True
    partial: float = 1.0
    max_per_step: int = 1


@register_component("sensory")
@dataclass
class Sensory(Component):
    radius: int = 3
    rays: int = 8
    fov: float = 360.0


@register_component("action_capabilities")
@dataclass
class ActionCapabilities(Component):
    move: bool = True
    attack: bool = True
    consume: bool = True
    reproduce: bool = True


@register_component("cooldowns")
@dataclass
class Cooldowns(Component):
    items: dict[str, float] = field(default_factory=dict)

    def ready(self, name: str) -> bool:
        return self.items.get(name, 0.0) <= 0.0

    def set(self, name: str, duration: float) -> None:
        self.items[name] = float(duration)

    def tick(self, dt: float = 1.0) -> None:
        for key in list(self.items):
            self.items[key] = max(0.0, self.items[key] - dt)


@dataclass
class ComponentContext:
    x: int
    y: int
    dx: int = 1
    dy: int = 0
    kind: str = ""
    species: str = ""
    normalize: Callable[[int, int], tuple[int, int]] | None = None

    def norm(self, x: int, y: int) -> tuple[int, int]:
        return self.normalize(x, y) if self.normalize else (x, y)


def build_body(data: Mapping[str, Any], ctx: ComponentContext) -> Body:
    length = int(data.get("length", 0))
    segments: list[Vec2] = []
    x, y = ctx.x, ctx.y
    for _ in range(max(0, length)):
        x -= ctx.dx
        y -= ctx.dy
        x, y = ctx.norm(x, y)
        segments.append(Vec2(x, y))
    return Body(segments=tuple(segments))


@dataclass
class Body(Component):
    segments: tuple[Vec2, ...] = ()

    @property
    def length(self) -> int:
        return len(self.segments)

    def cells(self, head: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        return (head,) + tuple(s.as_tuple() for s in self.segments)


@register_component("body")
def _register_body(data: Mapping[str, Any], ctx: ComponentContext) -> Body:
    return build_body(data, ctx)


def build_component(name: str, params: Mapping[str, Any] | None, ctx: ComponentContext) -> Component:
    factory = COMPONENTS.get(name)
    data = dict(params or {})
    if isinstance(factory, type):
        return factory(**data)
    return factory(data, ctx)


def build_components(spec: Mapping[str, Any], ctx: ComponentContext) -> list[Component]:
    components: list[Component] = []
    for name, params in spec.items():
        if params is None:
            continue
        components.append(build_component(name, params, ctx))
    return components


def clone_components(components: list[Component]) -> list[Component]:
    return [copy.deepcopy(c) for c in components]
