import os
import tempfile

import numpy as np
import torch

from alifelab.agents import A2CAgent, A2CGRUAgent, A2CLSTMAgent, DQNAgent, build_agent
from alifelab.core.interfaces import Transition
from alifelab.train import Trainer, build_env
from alifelab.world import WorldSpec, rules_from_config

spec = WorldSpec(
    width=10,
    height=10,
    seed=7,
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
        {"preset": "food", "count": 5},
    ],
)

env = build_env(spec, observation="entity", max_steps=40, seed=1)

ac = A2CAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=64, recurrent="lstm")
print("a2c+lstm params:", ac.parameter_count(), "recurrent:", ac.net.is_recurrent)
print("registered:", [n for n in ("a2c", "a2c_lstm", "a2c_gru") if n in __import__("alifelab").core.registry.AGENTS])

obs, _ = env.reset(seed=1)
for _ in range(3):
    a = ac.act(obs)
    prev = ac._state
assert prev is not None, "LSTM state must persist across steps"
state_before_reset = ac._state
ac.reset()
print("state cleared on reset:", ac._state is None)
obs, _ = env.reset(seed=1)

trainer = Trainer(env, ac, episodes=6, seed=1, max_steps=40)
history = trainer.run()
losses = [l for r in history for l in r.losses if l == l]
print("lstm losses[:5]:", [round(l, 3) for l in losses[:5]], "n=", len(losses))
print("returns:", [round(r.return_, 2) for r in history])

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "a2c_lstm.pt")
    ac.save(path)
    ac2 = v = A2CLSTMAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=64)
    ac2.load(path)
    ok = all(torch.equal(a, b) for a, b in zip(ac.net.state_dict().values(), ac2.net.state_dict().values()))
    print("save/load roundtrip:", ok)

gru = A2CGRUAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=64)
print("gru registered build ok, params:", gru.parameter_count())

try:
    DQNAgent(env.observation_space, env.action_space, recurrent="lstm")
    print("dqn+recurrent: NO ERROR (unexpected)")
except ValueError as exc:
    print("dqn+recurrent rejected:", exc)