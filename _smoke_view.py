"""Engine-free smoke run of the visualizer (no pygame, no simulation).

Builds a scripted multi-event demo frame, runs a headless visualizer loop,
snapshots the composed display, and exercises key bindings + recording + the
debug harness. Prints a short report.

Run:  python _smoke_view.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from alifelab.view import (
    AgentEphemera,
    EventPanel,
    InspectorPanel,
    LineagePanel,
    Recorder,
    StaticFrameSource,
    ViewEntity,
    ViewEvent,
    ViewFrame,
    ViewWorld,
    Visualizer,
    debug_tests,
)


def demo_frame(step: int) -> ViewFrame:
    world = ViewWorld(
        width=5,
        height=4,
        wrap=True,
        tick=step,
        episode=1,
        terrain=np.zeros((4, 5), dtype=np.int8),
        resources=np.zeros((4, 5), dtype=np.int8),
        entities=(
            ViewEntity(id=1, kind="snake", species="snake", x=step % 5, y=0, direction=(1, 0), size=1.0, predator=True, controller="agent"),
            ViewEntity(id=2, kind="snake", species="snake", x=(step + 2) % 5, y=1, health=70.0, max_health=100.0, age=float(step)),
            ViewEntity(id=3, kind="prey", species="prey", x=(step + 1) % 5, y=2, edible=True),
        ),
        population={"snake": 2, "prey": 1},
    )
    events = (
        ViewEvent(kind="move", tick=step, actor_id=1),
        ViewEvent(kind="consume", tick=step, actor_id=1, target_id=7, data={"category": "food"}) if step % 3 == 0 else
        ViewEvent(kind="grow", tick=step, actor_id=1, data={"size_gain": 0.4}),
    )
    return ViewFrame(
        world=world,
        events=(events[1],),
        population={"snake": 2, "prey": 1},
        episode=1,
        step=step,
        mode="train",
        episode_return=float(step) * 0.25,
        episode_length=step,
        agent=AgentEphemera(active=True, mode="train", agent_id=1, reward=0.1, episode_return=float(step) * 0.25, loss=0.42, epsilon=0.05, value=1.23),
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="view_smoke_"))
    frames = [demo_frame(i) for i in range(8)]
    source = StaticFrameSource(frames[0], advance=lambda: frames[min(int(getattr(source, "_i", 0)), len(frames) - 1)])

    viz = Visualizer.build(source, headless=True, speed=1.0, mode="train", recording_dir=str(tmp))
    viz.verify()
    viz.set_rgb_provider(lambda: viz.display_frame())
    viz.set_ascii_provider(lambda: "smoke ascii")

    viz.toggle_rgb_recording()
    viz.toggle_ascii_recording()
    for index in range(8):
        viz.step_once()
    viz.save_rgb(name="demo")
    viz.save_ascii(name="demo")

    print("selftests:")
    for line in debug_tests(verbose=True):
        if line and not line.startswith("  "):
            print("  " + line)
        else:
            print("    " + line)

    print("\nkey effects:")
    for key in ("play", "speed_up", "mode", "step", "fast", "panels", "overlay"):
        print(f"  {key!r:14} -> {viz.handle_key(key)}")

    print(f"\nsummary: {viz.summary()}")
    print("overview:")
    print("\n".join(viz.debug_overview()))
    print(f"\nartifacts in {tmp}")
    for path in sorted(tmp.iterdir()):
        print(f"  {path.name}  {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())