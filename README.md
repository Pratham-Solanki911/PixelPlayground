<div align="center">

# PixelPlayground

**`alifelab` — a modular artificial-life + reinforcement-learning experimentation framework for grid worlds.**

<a href="#"><span style="display:inline-block;padding:3px 10px;border-radius:14px;background:#0d1117;color:#7ee787;border:1px solid #238636;font-size:12px;margin:2px;">Python 3.11+</span></a>
<a href="#"><span style="display:inline-block;padding:3px 10px;border-radius:14px;background:#0d1117;color:#79c0ff;border:1px solid #1f6feb;font-size:12px;margin:2px;">PyTorch</span></a>
<a href="#"><span style="display:inline-block;padding:3px 10px;border-radius:14px;background:#0d1117;color:#d2a8ff;border:1px solid #8957e5;font-size:12px;margin:2px;">Algorithm-agnostic</span></a>
<a href="#"><span style="display:inline-block;padding:3px 10px;border-radius:14px;background:#0d1117;color:#ffa657;border:1px solid #bc4c00;font-size:12px;margin:2px;">Deterministic checkpoints</span></a>
<a href="#"><span style="display:inline-block;padding:3px 10px;border-radius:14px;background:#0d1117;color:#c9d1d9;border:1px solid #484f58;font-size:12px;margin:2px;">Zero-config experiment dirs</span></a>

Build worlds from pluggable **rules**, feed agents through pluggable **observations / actions / rewards**, and drive everything through a **single algorithm-agnostic `Trainer`** that logs metrics, evaluates, checkpoints, and resumes — deterministically.

**Flagship demo: the classic Snake world** (`snake` entity preset + `grid` observation + DQN).

[Quickstart](#quickstart) | [Architecture](#architecture) | [Building blocks](#building-blocks) | [Experiment tracking](#experiment-tracking--reproducibility) | [Configuration reference](#configuration-reference) | [Roadmap](#roadmap)

</div>

---

## What is PixelPlayground?

PixelPlayground hosts **alifelab**, a small research-oriented framework for experimentation with reinforcement learning agents living in simulated grid worlds. Every piece of the pipeline — world, rules, observation space, action space, reward function, neural architecture, and learning algorithm — is a **named, runtime-registered component** that can be switched by a string in a JSON config. That makes it trivial to sweep experiments, mirror a training run exactly, and resume from disk.

The framework deliberately separates three concerns:

1. **The world** — an entity-component-system grid engine with pluggable rule systems (motion, collisions, energy, food, predation, reproduction, mutation, ...).
2. **The agent** — the learning algorithm + neural architecture. Agents expose everything needed to resume through a single `state_dict()` / `load_state_dict()` protocol, so the trainer never needs to know what is inside.
3. **The run** — a `Trainer` that runs episodes, records metrics, evaluates exploration-free, and writes checkpoints, all into a self-describing run directory.

<p align="center">
<span style="font-size:13px;"><b>A train run is just a folder:</b> config, metrics, evaluations, checkpoints, metadata — read it later, resume it, or hand it to a colleague.</span>
</p>

---

## Highlights

<table align="center">
<tr>
<td align="center" width="33%">
<span style="font-size:18px;color:#7ee787;font-weight:bold;">Composable world</span><br/>
<span style="font-size:13px;color:#8b949e;">11 rule systems, 6 entity presets, terrain, wrap-around grids, and an ECS core that snapshots into clean NumPy state.</span>
</td>
<td align="center" width="33%">
<span style="font-size:18px;color:#79c0ff;font-weight:bold;">Drop-in algorithms</span><br/>
<span style="font-size:13px;color:#8b949e;">DQN, A2C, A2C-LSTM/GRU (recurrent hidden states), random baselines. 10 neural architectures incl. transformer & SSM blocks.</span>
</td>
<td align="center" width="33%">
<span style="font-size:18px;color:#d2a8ff;font-weight:bold;">Deterministic resume</span><br/>
<span style="font-size:13px;color:#8b949e;">Checkpoints carry weights, optimizers, replay buffers, epsilon, RNN hidden states, and Python/NumPy/Torch RNG states.</span>
</td>
</tr>
<tr>
<td align="center" width="33%">
<span style="font-size:18px;color:#ffa657;font-weight:bold;">Single Trainer</span><br/>
<span style="font-size:13px;color:#8b949e;">Training, evaluation, and checkpoints are algorithm-agnostic. Add a new agent class; tracking comes for free.</span>
</td>
<td align="center" width="33%">
<span style="font-size:18px;color:#ff7b72;font-weight:bold;">Registry-driven config</span><br/>
<span style="font-size:13px;color:#8b949e;">Everything is JSON-serializable. A run's <code>config.json</code> is enough to rebuild the world and agent exactly.</span>
</td>
<td align="center" width="33%">
<span style="font-size:18px;color:#e3b341;font-weight:bold;">Self-describing runs</span><br/>
<span style="font-size:13px;color:#8b949e;">metrics.jsonl, evaluation.json, metadata.json, latest + best checkpoints — no database needed.</span>
</td>
</tr>
</table>

---

## Architecture

### Layered stack

<div align="center">

<div style="display:inline-block;text-align:center;padding:10px 18px;margin:2px;background:#f0f6ff;border:1px solid #79c0ff;border-radius:8px;min-width:420px;max-width:640px;">
<span style="font-weight:bold;color:#0969da;">Interface Factory</span><br/>
<span style="font-size:12px;color:#57606a;">Observation space &middot; Action space &middot; Reward function<br/><code>grid</code> &middot; <code>raycast</code> &middot; <code>relative</code> &middot; <code>discrete_move</code> &middot; <code>continuous_move</code> &middot; <code>composite</code></span>
</div>
<br/>
<span style="color:#79c0ff;font-size:16px;">&#8595;</span><br/>
<div style="display:inline-block;text-align:center;padding:10px 18px;margin:2px;background:#effaf1;border:1px solid #7ee787;border-radius:8px;min-width:420px;max-width:640px;">
<span style="font-weight:bold;color:#1a7f37;">World Engine (ECS)</span><br/>
<span style="font-size:12px;color:#57606a;">Grid + entities + terrain | snapshotter / state<br/>Rules: movement &middot; collision &middot; energy &middot; growth &middot; food &middot; aging &middot; death &middot; spawn &middot; predation &middot; reproduction &middot; mutation</span>
</div>
<br/>
<span style="color:#7ee787;font-size:16px;">&#8595;</span><br/>
<div style="display:inline-block;text-align:center;padding:10px 18px;margin:2px;background:#f8f0ff;border:1px solid #d2a8ff;border-radius:8px;min-width:420px;max-width:640px;">
<span style="font-weight:bold;color:#8957e5;">Agent (learning algorithm + architecture)</span><br/>
<span style="font-size:12px;color:#57606a;">random &middot; dqn &middot; a2c &middot; a2c_lstm &middot; a2c_gru<br/>mlp &middot; cnn &middot; lstm &middot; gru &middot; transformer &middot; ssm &middot; cnn-* hybrids</span>
</div>
<br/>
<span style="color:#d2a8ff;font-size:16px;">&#8595;</span><br/>
<div style="display:inline-block;text-align:center;padding:10px 18px;margin:2px;background:#fff8ef;border:1px solid #ffa657;border-radius:8px;min-width:420px;max-width:640px;">
<span style="font-weight:bold;color:#bc4c00;">Trainer (algorithm-agnostic)</span><br/>
<span style="font-size:12px;color:#57606a;">episodes &middot; exploration-free evaluation &middot; metric recording &middot; best/latest checkpoints &middot; resume</span>
</div>
<br/>
<span style="color:#ffa657;font-size:16px;">&#8595;</span><br/>
<div style="display:inline-block;text-align:center;padding:10px 18px;margin:2px;background:#f6f8fa;border:1px solid #8b949e;border-radius:8px;min-width:420px;max-width:640px;">
<span style="font-weight:bold;color:#57606a;">Run artifacts</span><br/>
<span style="font-size:12px;color:#57606a;"><code>config.json</code> &middot; <code>metrics.jsonl</code> &middot; <code>evaluation.json</code> &middot; <code>metadata.json</code> &middot; <code>checkpoint_latest.pt</code> &middot; <code>checkpoint_best.pt</code></span>
</div>

</div>

### Episode data flow

Every interaction between the agent and the world passes through a small pipeline. The agent only ever sees observation tensors and emits actions; the environment only ever returns a reward and the next snapshot.

<div align="center">

<table style="border-collapse:collapse;text-align:center;">
<tr>
<td style="padding:6px;"><div style="padding:8px 12px;background:#effaf1;border:1px solid #7ee787;border-radius:6px;font-size:12px;color:#1a7f37;"><b>LifeEnv.reset</b></div></td>
<td style="padding:6px;color:#8b949e;font-size:16px;">&#8594;</td>
<td style="padding:6px;"><div style="padding:8px 12px;background:#f0f6ff;border:1px solid #79c0ff;border-radius:6px;font-size:12px;color:#0969da;"><b>Observation</b><br/>grid / raycast / ...</div></td>
<td style="padding:6px;color:#8b949e;font-size:16px;">&#8594;</td>
<td style="padding:6px;"><div style="padding:8px 12px;background:#f8f0ff;border:1px solid #d2a8ff;border-radius:6px;font-size:12px;color:#8957e5;"><b>Agent.act</b><br/>policy / epsilon-greedy</div></td>
<td style="padding:6px;color:#8b949e;font-size:16px;">&#8594;</td>
<td style="padding:6px;"><div style="padding:8px 12px;background:#effaf1;border:1px solid #7ee787;border-radius:6px;font-size:12px;color:#1a7f37;"><b>World.step</b><br/>rules run in order</div></td>
<td style="padding:6px;color:#8b949e;font-size:16px;">&#8594;</td>
<td style="padding:6px;"><div style="padding:8px 12px;background:#fff8ef;border:1px solid #ffa657;border-radius:6px;font-size:12px;color:#bc4c00;"><b>Reward + Transition</b><br/>terminal? truncated?</div></td>
</tr>
<tr>
<td colspan="9" style="color:#8b949e;font-size:13px;">&#8635; &nbsp; Agent.observe(transition) &rarr; Agent.train() every <code>train_every</code> steps &rarr; repeat until <code>episodes</code> done</td>
</tr>
</table>

</div>

### Training loop

<div align="center" style="max-width:560px;margin:0 auto;">

1&nbsp;<span style="padding:4px 10px;background:#f0f6ff;border:1px solid #79c0ff;border-radius:6px;font-size:13px;color:#0969da;">Build world &amp; agent from config</span> &nbsp;&rarr;&nbsp; <span style="padding:4px 10px;background:#effaf1;border:1px solid #7ee787;border-radius:6px;font-size:13px;color:#1a7f37;">Reset; collect starts</span> &nbsp;&rarr;&nbsp; <span style="padding:4px 10px;background:#f8f0ff;border:1px solid #d2a8ff;border-radius:6px;font-size:13px;color:#8957e5;">Step until done</span> &nbsp;&rarr;&nbsp; <span style="padding:4px 10px;background:#fff8ef;border:1px solid #ffa657;border-radius:6px;font-size:13px;color:#bc4c00;">Record episode metrics</span> &nbsp;&rarr;&nbsp; <span style="padding:4px 10px;background:#f6f8fa;border:1px solid #8b949e;border-radius:6px;font-size:13px;color:#57606a;">Checkpoint</span>

<span style="font-size:12px;color:#8b949e;">Evaluation is interleaved every <code>eval_every</code> episodes with exploration disabled; the best eval score is tracked for <code>checkpoint_best.pt</code>.</span>

</div>

---

## Quickstart

### Install

```bash
# Requires Python 3.11+ and PyTorch
pip install -e .          # from the repo root
```

### Run your first Snake agent

```python
from alifelab.experiment import Experiment, RunConfig

config = RunConfig(
    name="snake_ai",
    seed=7,
    episodes=100,
    world={
        "width": 10,
        "height": 10,
        "seed": 7,
        "wrap": True,
        "terrain": {"layout": "random", "wall": 0.05},
        "rules": {
            "movement": True,
            "collision": True,
            "food": True,
            "growth": True,
            "energy": {"drain": 0.12},
            "aging": True,
            "death": True,
        },
        "entities": [
            {"preset": "snake", "count": 1, "controller": "agent"},
            {"preset": "food", "count": 4},
        ],
    },
    observation={"name": "grid"},
    action={"name": "discrete_move"},
    reward={"name": "composite"},
    agent={"name": "dqn", "architecture": "cnn", "feature_dim": 64},
    eval_every=10,
    eval_episodes=5,
)

experiment = Experiment(config, run_id="run1")
experiment.train()
```

That is the whole loop. After it finishes, your run lives in:

```
runs/snake_ai/run1/
```

### Simulate a world without any agent

```python
from alifelab.world import WorldSpec, build_world, rules_from_config

spec = WorldSpec(
    width=8, height=8, seed=7, wrap=True,
    terrain={"layout": "random", "wall": 0.05},
    rules=rules_from_config({"movement": True, "collision": True,
                             "food": True, "growth": True}),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "food", "count": 5},
    ],
)
world = build_world(spec)
print(world.population_counts())   # e.g. {'snake': 1, 'food': 5}
for _ in range(50):
    world.set_intent(world.get_entities()[0].id, (1, 0))
    world.step()
```

### Render a frame

```python
from alifelab.render import AsciiRenderer, RGBRenderer

ascii_view = AsciiRenderer().render(world.snapshot())   # str
rgb_frames = RGBRenderer().render(world.snapshot())     # np.ndarray
```

---

## Building blocks

Everything below is selectable **by name** in config. That is the core idea: swap any part of the stack without touching code.

### Agents

| Agent | Experience | Recurrence | What `state_dict()` captures |
|---|---|---|---|
| `random` | none | no | its own RNG stream |
| `dqn` | off-policy (replay buffer) | no | network, target net, optimizer, epsilon, replay buffer, steps, RNGs |
| `a2c` | on-policy | no | actor-critic network, optimizer, steps, RNGs |
| `a2c_lstm` | on-policy | LSTM | everything above **plus carried LSTM hidden state (h + c)** |
| `a2c_gru` | on-policy | GRU | everything above **plus carried GRU hidden state** |

### Architectures

| Architecture | Notes |
|---|---|
| `mlp` | dense feed-forward, the default |
| `cnn` | 2D convolutions over grid observations |
| `lstm` / `gru` | sequence models with memory |
| `transformer` | attention over the step sequence |
| `ssm` | state-space sequence block |
| `cnn_lstm` / `cnn_gru` / `cnn_transformer` / `cnn_ssm` | CNN vision + sequence head hybrids |

### Observations

| Name | Produces | Best for |
|---|---|---|
| `grid` / `multichannel` | multi-channel grid tensor | vision-style policies |
| `grid_codes` | integer-coded grid | compact embeddings |
| `entity` / `entities` / `entity_list` | per-entity feature table | object-centric models |
| `entity_tokens` / `tokens` | tokenized entity table | transformer / attention |
| `relative` / `relative_coordinates` | positions relative to self | spatial targeting |
| `local` / `local_vision` | patch around the agent | partial observability |
| `raycast` / `vision` | distance rays | classic sensing |
| `temporal` / `stack` | concatenated history | handling occlusion |

### Actions, rewards, rules, presets

| Action spaces | Reward functions | Rule systems | Entity presets |
|---|---|---|---|
| `discrete_move` / `grid` | `composite` | `movement` | `snake` |
| `continuous_move` / `continuous` | `energy` | `collision` | `food` |
| `custom` | `sparse_food` | `energy`, `growth`, `aging`, `death` | `plant` |
| | `survival` | `food`, `spawn` | `predator` |
| | `zero` / `none` | `predation`, `reproduction`, `mutation` | `prey`, `rock` |

---

## Experiment tracking & reproducibility

The `Trainer` writes a complete, self-describing run directory. No external services required.

### Run directory layout

<pre style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:12px;font-size:12px;line-height:1.5;">
runs/
&#9492;&#9472;&#9472; snake_ai/                          <span style="color:#57606a;"># one folder per experiment_name</span>
   &#9492;&#9472;&#9472; run1/                          <span style="color:#57606a;"># one folder per run_id</span>
      &#9500;&#9472;&#9472; config.json                  <span style="color:#57606a;"># full world + interfaces + agent + training config</span>
      &#9500;&#9472;&#9472; metadata.json               <span style="color:#57606a;"># python, torch, numpy, platform, start time</span>
      &#9500;&#9472;&#9472; metrics.jsonl               <span style="color:#57606a;"># one JSON line per training / eval episode</span>
      &#9500;&#9472;&#9472; evaluation.json             <span style="color:#57606a;"># last + best evaluation summaries</span>
      &#9500;&#9472;&#9472; checkpoint_latest.pt        <span style="color:#57606a;"># most recent full state</span>
      &#9492;&#9472;&#9472; checkpoint_best.pt          <span style="color:#57606a;"># best by best_metric (default evaluation_return)</span>
</pre>

### What a checkpoint contains

<div align="center">

<table align="center" style="border-collapse:collapse;">
<tr style="background:#0d1117;color:#7ee787;">
<th style="padding:6px 12px;border:1px solid #30363d;">Category</th>
<th style="padding:6px 12px;border:1px solid #30363d;">Contents</th>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">Format</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;"><code>alifelab-checkpoint-v1</code> + timestamp + platform</td>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">Progress</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;">episode, global step</td>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">Agent</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;">network + optimizer state, steps, agent RNG, algorithm extras (epsilon, replay buffer, target net, RNN hidden states)</td>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">World</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;">JSON-safe world state for immediate re-instantiation</td>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">Randomness</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;">Python RNG, NumPy RNG, Torch RNG, CUDA RNG (when present) &mdash; all restorable</td>
</tr>
<tr>
<td style="padding:6px 12px;border:1px solid #d0d7de;font-weight:bold;">Config</td>
<td style="padding:6px 12px;border:1px solid #d0d7de;">environment / observation / reward / agent configuration (full run config)</td>
</tr>
</table>

</div>

Because RNG states are snapshotted and restored, a paused and resumed run continues the **same** pseudo-random stream as if it never stopped.

### Metrics recorded per episode

| Key | Meaning |
|---|---|
| `episode` | episode index |
| `global_step` | cumulative steps across all episodes |
| `episode_return` | undiscounted return |
| `episode_length` | steps taken |
| `training_loss` | mean loss from this episode's updates (deserialized if a tensor) |
| `evaluation_return` / `evaluation_length` | on eval episodes (exploration off) |
| `learning_rate` | current learning rate |
| `elapsed_time` | seconds since run start |

Any custom metric returned by `agent.train()` (e.g. `epsilon`, `value_loss`) is recorded automatically — the `Trainer` never needs to know your algorithm's internals.

### Resume from disk

```python
from alifelab.experiment import Experiment

# anywhere later, on any machine:
experiment = Experiment.open("runs/snake_ai/run1")
experiment.train()          # continues from checkpoint_best.pt / latest state

# or resume from an explicit path with a raw Trainer:
trainer.train(resume_from="runs/snake_ai/run1/checkpoint_latest.pt")
```

---

## Configuration reference

`RunConfig` is the single declarative entry point. Every field is JSON-serializable, so a run can be echoed verbatim into `config.json`.

| Field | Default | Description |
|---|---|---|
| `name` | — | experiment name (folder under `runs/`) |
| `world` | — | grid + terrain + rules + entities (see below) |
| `seed` | `0` | master seed for world & agent |
| `episodes` | `100` | number of training episodes |
| `observation` | `None` | observation space name or `{name, params...}` |
| `action` | `"discrete_move"` | action space name or dict |
| `reward` | `"composite"` | reward function name or dict |
| `agent` | `{"name": "random"}` | agent + architecture + hyperparams |
| `train_every` | `1` | train the agent every N steps |
| `max_steps` | `None` | cap steps per episode |
| `eval_every` | `10` | run evaluation every N episodes (`0` disables) |
| `eval_episodes` | `5` | evaluation episodes each time |
| `checkpoint_every` | `1` | write `checkpoint_latest.pt` every N episodes |
| `best_metric` | `evaluation_return` | metric that defines `checkpoint_best.pt` |
| `run_id` | auto | run folder name |
| `runs_dir` | `"runs"` | root directory for all experiments |
| `extra` | `{}` | free-form metadata stored in `config.json` |

A full `world` config looks like:

```json
{
  "width": 10, "height": 10, "seed": 7, "wrap": true,
  "terrain": {"layout": "random", "wall": 0.05},
  "rules": {
    "movement": true, "collision": true, "food": true,
    "growth": true, "energy": {"drain": 0.12},
    "aging": true, "death": true
  },
  "entities": [
    {"preset": "snake", "count": 1, "controller": "agent"},
    {"preset": "food", "count": 4}
  ]
}
```

---

## Tests

The suite is a set of self-contained smoke scripts at the repo root (each `python _smokeN.py` runs green):

| Script | Covers |
|---|---|
| `_smoke_arch.py` | every neural architecture forward pass + parameter counts |
| `_smoke1.py` | world build, rule execution, state & snapshot extraction |
| `_smoke2.py` | observation spaces |
| `_smoke3.py` | action spaces |
| `_smoke4.py` | agents: DQN, A2C, A2C-LSTM/GRU, random |
| `_smoke5.py` | renderers (ASCII + RGB) |
| `_smoke6.py` | checkpoint save/load, RNG restore, resume, best checkpoint, `Experiment.open()` |

```bash
python _smoke1.py   # ... run any of them
```

---

## Roadmap

- PPO and SAC algorithm ports
- Hyperparameter sweeps (`Sweeper`)
- Episode video recording through the renderers
- Multi-agent controllers in a single world
- Distributed / async rollout workers

---

<div align="center">

Built as **PixelPlayground** &middot; core library `alifelab` v0.1.0

</div>