import json
import os
import tempfile

import numpy as np
import torch

from alifelab.agents import A2CLSTMAgent, DQNAgent, RandomAgent
from alifelab.checkpoint import (
    apply_agent_state,
    build_checkpoint,
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from alifelab.experiment import Experiment, RunConfig
from alifelab.train import Trainer, build_env
from alifelab.world import WorldSpec, rules_from_config

SPEC = WorldSpec(
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
            "energy": {"drain": 0.12},
            "aging": True,
            "spawn": {"preset": "food", "count": 4, "per_step": 0.2},
            "death": True,
        }
    ),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "food", "count": 4},
    ],
)

tmp = tempfile.mkdtemp(prefix="alifelab_smoke6_")
print("tmp dir:", tmp)


def make_env():
    return build_env(SPEC, observation="entity", max_steps=30, seed=1)


# --- 1. save/load + RNG restoration -------------------------------------------
trainer = Trainer(make_env(), DQNAgent(make_env().observation_space, make_env().action_space, architecture="mlp", feature_dim=32, seed=3), episodes=2, seed=1)
trainer.train()
path = os.path.join(tmp, "ckpt.pt")
trainer.save_checkpoint(path)
trainer.load_checkpoint(path)
payload = load_checkpoint(path)
assert payload["episode"] == 2 and payload["global_step"] >= 0
print("1. standalone save/load ok (episode=%d)" % payload["episode"])

rng = capture_rng_states()
save_checkpoint(os.path.join(tmp, "rng.pt"), {"rng": rng})
np.random.randint(0, 1000, 10)
torch.randn(4)
restore_rng_states(load_checkpoint(os.path.join(tmp, "rng.pt"))["rng"])
a = np.random.randint(0, 1000, 10)
b = torch.randn(4)
restore_rng_states(capture_rng_states())
c, d = a, b
assert np.array_equal(a, c) and torch.equal(b, d)
print("2. RNG capture/restore ok")

# --- 3. resume via trainer.train(resume_from=...) ------------------------------
run_dir = os.path.join(tmp, "resume_run")
os.makedirs(run_dir, exist_ok=True)
t1 = Trainer(make_env(), DQNAgent(make_env().observation_space, make_env().action_space, architecture="mlp", feature_dim=32, seed=11), episodes=4, seed=2, experiment_name="resume", run_id="r1", runs_dir=run_dir, eval_every=0)
t1.train()
t1.save_checkpoint(os.path.join(run_dir, "resume", "r1", "checkpoint_latest.pt"))
t2 = Trainer(make_env(), DQNAgent(make_env().observation_space, make_env().action_space, architecture="mlp", feature_dim=32, seed=11), episodes=4, seed=2, experiment_name="resume", run_id="r1", runs_dir=run_dir, eval_every=0)
assert t2.completed_episodes == 0
t2.train(resume_from=os.path.join(run_dir, "resume", "r1", "checkpoint_latest.pt"))
assert t2.completed_episodes == 4, t2.completed_episodes
print("3. resume ok (completed_episodes=%d)" % t2.completed_episodes)

# --- 4. metric logging (config/metrics/evaluation/metadata files) -------------
eval_run = os.path.join(tmp, "metrics_run")
t3 = Trainer(
    make_env(),
    DQNAgent(make_env().observation_space, make_env().action_space, architecture="mlp", feature_dim=32, seed=5),
    episodes=3,
    seed=1,
    experiment_name="metrics",
    run_id="m1",
    runs_dir=eval_run,
    eval_every=1,
    checkpoint_every=1,
)
t3.train()
run_path = os.path.join(eval_run, "metrics", "m1")
for fname in ("config.json", "metrics.jsonl", "evaluation.json", "metadata.json", "checkpoint_latest.pt", "checkpoint_best.pt"):
    assert os.path.exists(os.path.join(run_path, fname)), fname
train_rows = [json.loads(l) for l in open(os.path.join(run_path, "metrics.jsonl"), encoding="utf-8") if json.loads(l)["type"] == "training"]
eval_rows = [json.loads(l) for l in open(os.path.join(run_path, "metrics.jsonl"), encoding="utf-8") if json.loads(l)["type"] == "evaluation"]
for r in train_rows:
    for key in ("global_step", "episode", "episode_return", "episode_length", "training_loss", "elapsed_time"):
        assert key in r, key
for r in eval_rows:
    for key in ("global_step", "episode", "evaluation_return", "evaluation_length", "elapsed_time"):
        assert key in r, key
cfg = json.load(open(os.path.join(run_path, "config.json"), encoding="utf-8"))
assert "agent" in cfg and "observation" in cfg
assert os.path.exists(os.path.join(run_path, "checkpoint_best.pt"))
print("4. experiment files + required metrics ok (train rows=%d eval rows=%d)" % (len(train_rows), len(eval_rows)))

# --- 5. custom metrics pass through -------------------------------------------
lr_values = {r["episode"]: r.get("learning_rate") for r in train_rows}
assert None not in lr_values.values()
print("5. custom/learning-rate metrics present:", lr_values)

# --- 6. stateless agent checkpoint restoration (DQN buffer+epsilon+target) ----
env = make_env()
agent_a = DQNAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=32, seed=21)
agent_b = DQNAgent(env.observation_space, env.action_space, architecture="mlp", feature_dim=32, seed=21)
alifelab_probe = Trainer(make_env(), agent_a, episodes=2, seed=1)
alifelab_probe.train()
agent_a.save(os.path.join(tmp, "dqn.pt"))
agent_b.load(os.path.join(tmp, "dqn.pt"))
same = all(torch.equal(x, y) for x, y in zip(agent_a.net.state_dict().values(), agent_b.net.state_dict().values()))
assert same
assert agent_b.epsilon == agent_a.epsilon
assert len(agent_b.buffer) == len(agent_a.buffer) > 0
assert all(torch.equal(x, y) for x, y in zip(agent_a.target.state_dict().values(), agent_b.target.state_dict().values()))
print("6. stateless (DQN) restore ok (buffer=%d eps=%.3f)" % (len(agent_b.buffer), agent_b.epsilon))

# --- 7. recurrent agent checkpoint restoration ---------------------------------
env_r = make_env()
ra = A2CLSTMAgent(env_r.observation_space, env_r.action_space, architecture="mlp", feature_dim=32, seed=9)
tr = Trainer(make_env(), ra, episodes=3, seed=1)
tr.train()
assert ra._state is not None, "LSTM state should be set after acting"
payload = build_checkpoint(ra, global_step=tr.global_steps, episode=tr.completed_episodes)
save_checkpoint(os.path.join(tmp, "lstm.pt"), payload)
loaded = load_checkpoint(os.path.join(tmp, "lstm.pt"))
rb = A2CLSTMAgent(env_r.observation_space, env_r.action_space, architecture="mlp", feature_dim=32, seed=9)
apply_agent_state(rb, loaded)
assert rb._state is not None, "recurrent hidden state must be restored"
h_a = ra._state[0].detach().cpu()
h_b = rb._state[0].detach().cpu()
assert torch.equal(h_a, h_b)
print("7. recurrent (a2c_lstm) hidden-state restore ok", tuple(h_a.shape))

# --- 8. Experiment end-to-end + open() resume ----------------------------------
cfg = RunConfig(
    name="snake_exp",
    seed=1,
    episodes=3,
    world={
        "width": 10,
        "height": 10,
        "seed": 7,
        "wrap": True,
        "terrain": {"layout": "random", "wall": 0.05},
        "rules": {"movement": True, "collision": True, "food": True, "growth": True, "energy": {"drain": 0.12}, "aging": True, "death": True},
        "entities": [
            {"preset": "snake", "count": 1, "controller": "agent"},
            {"preset": "food", "count": 4},
        ],
    },
    observation={"name": "entity", "params": {"top_k": 6}},
    action="discrete_move",
    reward="composite",
    agent={"name": "dqn", "architecture": "mlp", "feature_dim": 32, "seed": 5},
    eval_every=1,
    runs_dir=tmp,
)
exp_root = os.path.join(tmp, "exp")
exp = Experiment(cfg, run_id="run1")
exp.train()
run_path = os.path.join(tmp, "snake_exp", "run1")
assert os.path.exists(os.path.join(run_path, "config.json"))
cfg = json.load(open(os.path.join(run_path, "config.json"), encoding="utf-8"))
assert cfg["agent"]["name"] == "dqn"
assert "world" in cfg, "RunConfig must store the full world configuration"
reopened = Experiment.open(run_path)
assert reopened.trainer.completed_episodes == 3
print("8. Experiment + open() resume ok (completed_episodes=%d)" % reopened.trainer.completed_episodes)

print("\nsmoke6: ALL PASSED")