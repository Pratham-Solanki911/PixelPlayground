import os
import tempfile

import numpy as np
import torch

from alifelab.actions import DiscreteMoveActions
from alifelab.agents import DQNAgent, build_agent
from alifelab.core.interfaces import Transition
from alifelab.env import CompositeReward, LifeEnv
from alifelab.obs import EntityObservation
from alifelab.world import WorldSpec, build_world, rules_from_config

spec = WorldSpec(
    width=10,
    height=10,
    seed=42,
    wrap=True,
    terrain={"layout": "random", "wall": 0.06},
    rules=rules_from_config(
        {
            "movement": True,
            "collision": True,
            "food": True,
            "growth": True,
            "energy": {"drain": 0.15},
            "aging": True,
            "spawn": {"preset": "food", "count": 5, "per_step": 0.2},
            "death": True,
        }
    ),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "prey", "count": 2},
        {"preset": "food", "count": 5},
    ],
)

env = LifeEnv(
    build_world(spec),
    DiscreteMoveActions(n=5),
    EntityObservation(top_k=6),
    CompositeReward(),
    max_steps=40,
)


def run_agent(agent, episodes=4):
    returns = []
    losses = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=1)
        agent.reset()
        total = 0.0
        for _ in range(env.max_steps):
            action = agent.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.observe(Transition(obs, action, reward, next_obs, term, trunc, info))
            metrics = agent.train()
            if metrics.get("loss") == metrics.get("loss") and "loss" in metrics:
                losses.append(metrics["loss"])
            obs = next_obs
            total += reward
            if term or trunc:
                break
        returns.append(total)
    return returns, losses


agent = DQNAgent(
    env.observation_space,
    env.action_space,
    architecture="mlp",
    feature_dim=64,
    lr=3e-4,
    gamma=0.99,
    batch_size=32,
    tau=0.01,
    epsilon_decay=0.99,
)
print("agent:", type(agent).__name__, "params:", agent.parameter_count())
print("obs space:", env.observation_space, "action space:", env.action_space)
returns, losses = run_agent(agent, episodes=4)
print("returns:", [round(r, 2) for r in returns])
print("losses[1:]:", [round(l, 3) for l in losses[1:10]])

replays = agent.replay if hasattr(agent, "replay") else None

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "agent.pt")
    agent.save(path)
    agent2 = DQNAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=64)
    agent2.load(path)
    same = all(
        torch.equal(a, b)
        for a, b in zip(agent.net.state_dict().values(), agent2.net.state_dict().values())
    )
    print("save/load roundtrip:", same)

# registry-driven construction
reg_agent = build_agent("random", env.observation_space, env.action_space, seed=3)
o, _ = env.reset(seed=1)
print("registered random agent action:", reg_agent.act(o))

# actor-critic path
from alifelab.agents import A2CAgent

ac = A2CAgent(
    env.observation_space,
    env.action_space,
    architecture="mlp",
    feature_dim=64,
    lr=3e-4,
    entropy_coef=0.05,
)
ac_returns, ac_losses = run_agent(ac, episodes=3)
print("a2c returns:", [round(r, 2) for r in ac_returns])
print("a2c losses[1:]:", [round(l, 3) for l in ac_losses[1:6]])