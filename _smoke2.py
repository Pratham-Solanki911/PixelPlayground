import numpy as np

from alifelab.actions import DiscreteMoveActions
from alifelab.env import CompositeReward, LifeEnv
from alifelab.obs import (
    EntityObservation,
    EntityTokenObservation,
    GridObservation,
    LocalVisionObservation,
    MultiChannelGridObservation,
    RaycastObservation,
    RelativeCoordinateObservation,
    TemporalObservation,
)
from alifelab.world import WorldSpec, build_world, rules_from_config

spec = WorldSpec(
    width=12,
    height=10,
    seed=42,
    wrap=True,
    terrain={"layout": "random", "wall": 0.08},
    rules=rules_from_config(
        {
            "movement": True,
            "collision": True,
            "food": True,
            "growth": True,
            "energy": {"drain": 0.2},
            "aging": True,
            "spawn": {"preset": "food", "count": 6, "per_step": 0.3},
            "death": True,
        }
    ),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "prey", "count": 2},
        {"preset": "rock", "count": 5},
        {"preset": "food", "count": 6},
    ],
)

builders = {
    "grid_codes": GridObservation(grid_size=(10, 12)),
    "grid8": MultiChannelGridObservation(channels=8, grid_size=(10, 12)),
    "local": LocalVisionObservation(radius=3, channels=5),
    "entity": EntityObservation(top_k=6),
    "tokens": EntityTokenObservation(top_k=6),
    "relative": RelativeCoordinateObservation(top_k=5),
    "raycast": RaycastObservation(rays=8),
    "temporal": TemporalObservation(base={"type": "entity", "top_k": 4}, stack=3),
}

world = build_world(spec)
env = LifeEnv(
    world,
    DiscreteMoveActions(n=5),
    builders["grid8"],
    CompositeReward(),
    max_steps=50,
)
obs, info = env.reset(seed=1)
print("reset obs", obs.shape, obs.dtype, "space", env.observation_space)
print("info keys", sorted(info.keys()))

for name, builder in builders.items():
    builder.reset()
    out = builder.build(world.snapshot(), env.agent_id)
    print(f"{name:10s} shape={out.shape} range=({out.min():.2f},{out.max():.2f})")

rng = np.random.default_rng(0)
returns = 0.0
for _ in range(30):
    action = env.action_space.sample(rng)
    obs, reward, term, trunc, info = env.step(action)
    returns += reward
    if term or trunc:
        break
print("steps", info["steps"], "return", round(returns, 3), "term", term, "trunc", trunc)
print("final metrics", {k: v for k, v in info["metrics"].items() if k != "population"})

# gym-style API determinism check
def rollout(seed):
    w = build_world(spec)
    e = LifeEnv(w, DiscreteMoveActions(n=5), EntityObservation(top_k=6), CompositeReward(), max_steps=30)
    o, _ = e.reset(seed=seed)
    total = 0.0
    actions = [0, 1, 2, 3, 4, 1, 1, 2]
    for i in range(30):
        o, r, t, tr, _ = e.step(actions[i % len(actions)])
        total += r
        if t or tr:
            break
    return total, e.world.get_state()

a = rollout(5)
b = rollout(5)
print("env deterministic:", a == b)
