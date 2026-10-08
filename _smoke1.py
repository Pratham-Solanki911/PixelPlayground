import numpy as np

from alifelab.core.types import MoveIntent
from alifelab.world import WorldSpec, build_world, rules_from_config
from alifelab.world.components import Body, Position, Energy, Growth

spec = WorldSpec(
    width=8,
    height=8,
    seed=7,
    wrap=True,
    terrain={"layout": "random", "wall": 0.05, "water": 0.0},
    rules=rules_from_config(
        {
            "movement": True,
            "collision": True,
            "food": True,
            "growth": True,
            "energy": True,
            "aging": True,
            "spawn": {"preset": "food", "count": 5, "per_step": 0.2},
            "death": True,
        }
    ),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "food", "count": 5},
    ],
)
w = build_world(spec)
print("tick", w.tick, "pop", w.population_counts())
snake = [e for e in w.get_entities() if e.kind == "snake"][0]
print("snake id", snake.id, "body", len(snake.get(Body).segments), "pos", snake.get(Position))

for _ in range(20):
    w.set_intent(snake.id, MoveIntent(1, 0))
    w.step()
print(
    "after 20 ticks pos",
    snake.get(Position),
    "body",
    len(snake.get(Body).segments),
    "energy",
    round(snake.get(Energy).value, 2),
)
st = w.get_state()
print("state keys", sorted(st.keys()))
print("terrain row0:", st["terrain"][0])
snap = w.snapshot()
print("layers", snap.layers.terrain.shape, float(snap.layers.resources.sum()), int(snap.layers.entities.max()))
print("entity view", snap.by_id[snake.id])
print("events last tick", [e.kind for e in snap.events])

w2 = build_world(spec)
for _ in range(20):
    w2.set_intent(snake.id, MoveIntent(1, 0))
    w2.step()
print("deterministic:", w.get_state() == w2.get_state())
