"""Generate demo media (PNG / GIF / MP4 / ASCII) for the README.

Drives the REAL engine headlessly: builds a lively 15x15 world with a
scripted snake agent (heads for food, dodges obstacles), fleeing prey, a
stalking predator, food, plants and rocks. Every tick is rendered through the
engine-free Visualizer into the composed display frame, and the collected
frames are exported as:

    assets/demo.png     hero screenshot (large scale, selection on the snake)
    assets/demo.gif     animated preview (README friendly)
    assets/demo.mp4     video
    assets/demo_ascii.txt  static ASCII render of the world

Requires Pillow + imageio + imageio-ffmpeg (media only; the view package and
all selftests work without them).

Run:  python _media_view.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from alifelab.core.types import IdleIntent, MoveIntent
from alifelab.render.ascii import AsciiRenderer
from alifelab.view import AgentEphemera, Visualizer, snapshot_to_frame
from alifelab.view.model import ViewFrame
from alifelab.world import WorldSpec, build_world, rules_from_config

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

SEED = 20260214
SIZE = 15
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
NAMES = {v: k for k, v in DIRS.items()} | {(0, 0): "IDLE"}


def build_session():
    spec = WorldSpec(
        width=SIZE,
        height=SIZE,
        wrap=False,
        seed=SEED,
        rules=rules_from_config(None),
        entities=[
            {
                "preset": "snake",
                "count": 1,
                "controller": "agent",
                "params": {
                    "body": {"length": 4},
                    "reproduction": {"enabled": False},
                    "energy": {"value": 120.0, "max": 150.0},
                },
            },
            {"preset": "prey", "count": 2, "params": {"age": {"lifetime": 500.0}}},
            {"preset": "predator", "count": 1},
            {"preset": "food", "count": 10},
            {"preset": "plant", "count": 6},
            {"preset": "rock", "count": 4},
        ],
    )
    return build_world(spec)


class DemoSession:
    """Drives every entity with simple handcrafted behaviours each tick."""

    def __init__(self, world):
        self.world = world
        self.last_events = ()

    def _snap(self):
        return self.world.snapshot()

    def _controller_id(self, snapshot, name: str) -> int | None:
        for view in snapshot.entities:
            if view.controller == name:
                return view.id
        return None

    def _ids(self, snapshot, kinds: set[str]) -> list[int]:
        return [v.id for v in snapshot.entities if v.kind in kinds]

    @staticmethod
    def _toward(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        dx = 1 if b[0] > a[0] else -1 if b[0] < a[0] else 0
        dy = 1 if b[1] > a[1] else -1 if b[1] < a[1] else 0
        return dx, dy

    @staticmethod
    def _away(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return -DemoSession._toward(a, b)[0], -DemoSession._toward(a, b)[1]

    def _cell_state(self, snapshot) -> dict[tuple[int, int], bool]:
        occupied = {}
        for view in snapshot.entities:
            solid = view.solid and not view.edible
            occupied[(view.x, view.y)] = occupied.get((view.x, view.y), False) or solid
            for bx, by in view.body or ():
                occupied[(bx, by)] = True
        return occupied

    def _safe(self, snapshot, occupied, fx, fy, exclude: set[int]) -> tuple[int, int] | None:
        w, h = self.world.width, self.world.height
        for dx, dy in ((fx, fy), (fy, fx), (-fy, -fx), (-fx, -fy), (dx := 0, 0) if False else (0, 0)):
            if (dx, dy) == (0, 0):
                continue
            tx, ty = (int(self.world.grid.normalize(0 + dx if False else 0, 0)) if False else 0, 0)
        candidates = [(fx, fy), (fy, fx) if fx != 0 or fy != 0 else (0, 0), (-fy, -fx), (-fx, -fy)]
        for dx, dy in candidates:
            if (dx, dy) == (0, 0):
                continue
            tx, ty = self.world.grid.normalize(0 + dx, 0 + dy)  # placeholder, replaced below
            return (dx, dy)
        return None

    def _pick(self, snapshot, pos: tuple[int, int], bias: tuple[int, int], occupied: dict) -> tuple[int, int]:
        """Valid (dx, dy) closest to bias, else any valid move, else stay.

        On non-wrapping worlds ``normalize`` clamps to the border, so a
        candidate whose clamped target equals the current cell (move off-grid)
        is discarded.
        """
        tries = [bias, (bias[1], bias[0]), (-bias[1], -bias[0]), (-bias[0], -bias[1])]
        for dx, dy in tries:
            if (dx, dy) == (0, 0):
                continue
            tx, ty = self.world.grid.normalize(pos[0] + dx, pos[1] + dy)
            if (tx, ty) == pos:
                continue
            if not occupied.get((tx, ty), False):
                return (dx, dy)
        for dx, dy in DIRS.values():
            tx, ty = self.world.grid.normalize(pos[0] + dx, pos[1] + dy)
            if (tx, ty) == pos:
                continue
            if not occupied.get((tx, ty), False):
                return (dx, dy)
        return (0, 0)

    def step(self) -> tuple:
        snapshot = self._snap()
        occupied = self._cell_state(snapshot)
        positions = {v.id: (v.x, v.y) for v in snapshot.entities}
        snake = self._controller_id(snapshot, "agent")
        targets = [v for v in snapshot.entities if v.edible]
        predators = self._ids(snapshot, {"predator"})
        preys = self._ids(snapshot, {"prey"})

        for view in snapshot.entities:
            pos = positions[view.id]
            intent: object = IdleIntent()
            if view.id == snake:
                target = min(targets, key=lambda t: self.world.grid.chebyshev(pos, (t.x, t.y)), default=None)
                bias = self._toward(pos, (target.x, target.y)) if target else (0, 0)
                intent = MoveIntent(*self._pick(snapshot, pos, bias, occupied))
            elif view.kind == "predator" and snake is not None:
                bias = self._toward(pos, positions[snake])
                intent = MoveIntent(*self._pick(snapshot, pos, bias, occupied))
            elif view.kind == "prey":
                threats = [positions[s] for s in (predators + [snake]) if s in positions]
                bias = (0, 0)
                if threats:
                    bias = self._toward(pos, min(threats, key=lambda t: self.world.grid.chebyshev(pos, t)))
                    bias = (-bias[0], -bias[1])
                else:
                    bias = (0, 0)
                intent = MoveIntent(*self._pick(snapshot, pos, bias, occupied))
            elif view.movable:
                intent = IdleIntent()
            self.world.set_intent(view.id, intent)
        self.last_events = self.world.step()
        return self.last_events

    def reset(self) -> None:
        self.world.reset()

    def set_mode(self, mode: str) -> None:
        pass


class DemoSource:
    """FrameSource that feeds live engine snapshots into the visualizer."""

    def __init__(self, session: DemoSession):
        self.session = session
        self.return_so_far = 0.0
        self.last_reward = 0.0

    def _agent_id(self, snapshot) -> int | None:
        for view in snapshot.entities:
            if view.controller == "agent":
                return view.id
        return None

    def frame(self) -> ViewFrame:
        snapshot = self.session.world.snapshot()
        agent_id = self._agent_id(snapshot)
        agent_view = snapshot.by_id.get(agent_id) if agent_id is not None else None
        kind = set()
        for event in self.session.last_events:
            if event.actor_id == agent_id and event.kind in {"consume", "predation"}:
                kind.add(event.kind)
            elif event.actor_id == agent_id and event.kind == "death":
                kind.add("death")
        reward = float(len(kind) * 0.5) - (0.75 if "death" in kind else 0.0)
        self.last_reward = reward
        self.return_so_far += reward
        ephemera = AgentEphemera(
            active=agent_view is not None,
            mode="train",
            episode=snapshot.episode,
            step=snapshot.tick,
            global_step=snapshot.tick,
            reward=reward,
            episode_return=self.return_so_far,
            episode_length=snapshot.tick,
            loss=None,
            epsilon=0.0,
            value=agent_view.energy / agent_view.max_energy if agent_view and agent_view.max_energy else None,
            policy="greedy (scripted demo)",
            architecture="none",
            recurrent=False,
            recurrent_active=False,
            current_action_name="hunt" if "consume" in kind else "move",
            action_labels=tuple(NAMES.values()),
            agent_id=agent_id,
            agent_ids=(agent_id,) if agent_id is not None else (),
        )
        return snapshot_to_frame(snapshot, agent=ephemera, mode="train")

    def step_once(self) -> None:
        self.session.step()

    def reset(self) -> None:
        self.session.reset()
        self.return_so_far = 0.0

    def set_mode(self, mode: str) -> None:
        self.session.set_mode(mode)


def hero_screenshot(viz: Visualizer, snake_id: int | None, path: Path) -> None:
    viz.world_scale = 26
    viz.show_help = False
    viz.show_overlay = True
    viz.show_panels = True
    if snake_id is not None:
        viz.focus(snake_id)
    image = viz.render_display()
    from PIL import Image

    Image.fromarray(image).save(path)
    print(f"saved {path} ({image.shape[1]}x{image.shape[0]})")


def pick_hero_tick(session: DemoSession, total: int) -> int:
    """Last tick (snake alive, food remaining, decent population) for the hero shot."""
    chosen = 0
    for index in range(1, total + 1):
        session.step()
        snapshot = session.world.snapshot()
        snake_alive = any(v.controller == "agent" for v in snapshot.entities)
        counts: dict[str, int] = {}
        for view in snapshot.entities:
            counts[view.kind] = counts.get(view.kind, 0) + 1
        if index >= 20 and snake_alive and counts.get("food", 0) >= 3 and len(snapshot.entities) >= 15:
            chosen = index
    return chosen


def main() -> int:
    session = DemoSession(build_session())
    source = DemoSource(session)
    viz = Visualizer.build(source, headless=True, speed=1.0, mode="train", recording_dir="runs/view")
    session.world.episode = 1
    viz.verify()

    hero_tick = pick_hero_tick(DemoSession(build_session()), steps := 100)
    print(f"hero tick selected: {hero_tick or 'none (fallback to last)'}")

    frames: list[np.ndarray] = []
    ascii_writer = AsciiRenderer(color=False)
    ascii_frames: list[str] = []
    hero = 0

    for index in range(1, steps + 1):
        viz.step_once()
        if index in (1, steps // 3, steps // 2, steps):
            ascii_frames.append(f"--- tick {index} ---\n" + ascii_writer.render(session.world.snapshot()))
        if hero_tick and index == hero_tick:
            image = viz.render_display()
            hero = index
            snake = source._agent_id(session.world.snapshot())
            hero_screenshot(viz, snake, ASSETS / "demo.png")
            viz.world_scale = None
            viz.show_panels = True
            image = viz.render_display()
        frames.append(viz.render_display())

    if hero == 0:
        image = viz.render_display()
        hero_screenshot(viz, source._agent_id(session.world.snapshot()), ASSETS / "demo.png")
    first = viz.render_display()
    print(f"tick 000  display {first.shape[1]}x{first.shape[0]}")

    ascii_path = ASSETS / "demo_ascii.txt"
    ascii_path.write_text("\n\n".join(ascii_frames) + "\n", encoding="utf-8")
    print(f"saved {ascii_path}")

    # shrink for media (compose frame is tall; keep aspect, cap width)
    width = min(frames[0].shape[1], 420)
    scale = width / frames[0].shape[1]
    height = int(frames[0].shape[0] * scale)
    thumb = [imageio_resize(f, (width, height)) for f in frames]

    gif_path = ASSETS / "demo.gif"
    fps = 2  # 500 ms per world step
    save_gif(gif_path, thumb, fps=fps)
    print(f"saved {gif_path} ({len(thumb)} frames @ {fps} fps)")

    import imageio.v2 as iio

    mp4_path = ASSETS / "demo.mp4"
    iio.mimsave(mp4_path, thumb, format="FFMPEG", fps=fps, codec="libx264", quality=7, macro_block_size=1)
    print(f"saved {mp4_path}")

    for path in sorted(ASSETS.glob("demo.*")):
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")
    return 0


def save_gif(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    from PIL import Image

    images = [Image.fromarray(f).convert("RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=max(1, round(1000 / fps)),
        loop=0,
    )


def imageio_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


if __name__ == "__main__":
    sys.exit(main())