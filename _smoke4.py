from alifelab.agents import DQNAgent
from alifelab.render import AsciiRenderer, RGBRenderer
from alifelab.train import Trainer, build_env
from alifelab.world import WorldSpec, rules_from_config

spec = WorldSpec(
    width=14,
    height=10,
    seed=7,
    wrap=True,
    terrain={"layout": "random", "wall": 0.07},
    rules=rules_from_config(
        {
            "movement": True,
            "collision": True,
            "food": True,
            "growth": True,
            "energy": {"drain": 0.15},
            "aging": True,
            "spawn": {"preset": "food", "count": 6, "per_step": 0.2},
            "death": True,
        }
    ),
    entities=[
        {"preset": "snake", "count": 1, "controller": "agent"},
        {"preset": "prey", "count": 1},
        {"preset": "food", "count": 6},
    ],
)

env = build_env(spec, observation="entity", max_steps=40, seed=1)
env.set_renderer(AsciiRenderer(color=True))

agent = DQNAgent(
    env.observation_space,
    env.action_space,
    architecture="mlp",
    feature_dim=64,
    lr=3e-4,
    batch_size=32,
    tau=0.01,
    epsilon_decay=0.995,
)

trainer = Trainer(env, agent, episodes=12, seed=1, max_steps=40)
history = trainer.run()
print("final 3 returns:", [round(r.return_, 2) for r in history[-3:]])
print("mean return over last 5:", round(trainer.mean_return(window=5), 2))
print("eval:", {k: round(v, 3) for k, v in trainer.evaluate(5).items()})

env.reset(seed=1)
rgb = RGBRenderer()
frame = rgb.render(env.world.snapshot())
print("rgb frame shape:", frame.shape, frame.dtype)
ascii_frame = AsciiRenderer(color=False).render(env.world.snapshot())
print(ascii_frame)