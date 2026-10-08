"""Diagnostics: lightweight self-tests + a structured debug report.

Everything here is engine-free (no simulation, RL, pygame, or torch imports)
so it runs literally anywhere the view package can be imported -- including on
a machine with no GUI at all. It is the visualizer's built-in sanity harness.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Mapping

import numpy as np

from .adapter import snapshot_to_frame
from .controller import PLAY_SPEEDS, PlaybackController
from .events import EventLog, Genealogy
from .gui import DashboardPanel  # noqa: F401  (import guard)
from .inspector import Inspector
from .model import AgentEphemera, ViewEntity, ViewEvent, ViewFrame, ViewWorld
from .population import PopulationTracker
from .recording import Recorder
from .renderers import OverlayRenderer, WorldRenderer
from .sources import StaticFrameSource

_TESTS = {
    "model": (
        ("world_basics", "build world, entities, by-id lookup, names"),
        ("frame_basics", "build frame with events + population"),
    ),
    "renderers": (
        ("world_render", "WorldRenderer produces correctly sized image"),
        ("overlay_render", "OverlayRenderer produces a readable strip"),
        ("observation_render", "ObservationRenderer handles 1D/2D/3D"),
    ),
    "events": (
        ("log_format", "EventLog formats known + unknown kinds"),
        ("genealogy", "Genealogy derives parents + offspring"),
        ("population", "tracker counts births/deaths/deltas"),
        ("inspector", "inspector builds a report for dead + alive"),
    ),
    "adapters": (
        ("snapshot_to_frame", "fake snapshot converts to ViewFrame"),
        ("playback", "speed clamp, cycling, step accumulation"),
        ("recorder", "RGB + ASCII capture and save round-trip"),
    ),
}


def debug_tests(verbose: bool = False) -> list[str]:
    """Run all self-tests, returning formatted output lines.

    Does not raise: failures are reported as ``FAIL`` lines. Returns a list of
    strings suitable for printing or returning from a debug command.
    """
    out: list[str] = []
    total = passed = 0
    for category, cases in _TESTS.items():
        out.append(f"[{category}]")
        for name, doc in cases:
            total += 1
            try:
                _run(_TESTS[category], name)
                passed += 1
                if verbose:
                    out.append(f"  ok  {name} -- {doc}")
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                out.append(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        if not verbose:
            out.append(f"  ({category}: {_counts(_TESTS[category])})")
    out.append(f"{passed}/{total} checks passed")
    return out


def _counts(cases) -> str:
    return f"{len(cases)} checks"


def _run(cases, name: str) -> None:
    actor = globals().get(f"_test_{name}")
    if actor is None:
        raise NotImplementedError(name)
    actor()


def _test_world_basics():
    entities = (
        ViewEntity(id=3, kind="snake", species="snake", x=0, y=0),
        ViewEntity(id=7, kind="prey", species="prey", x=2, y=3, body=((1, 3),)),
    )
    world = ViewWorld(
        width=4,
        height=4,
        wrap=False,
        tick=0,
        episode=0,
        terrain=np.zeros((4, 4), dtype=np.int8),
        resources=np.zeros((4, 4), dtype=np.int8),
        entities=entities,
        population={"snake": 1, "prey": 1},
    )
    assert world.total_population == 2
    assert world.entity_by_id(7).x == 2
    assert world.entity_by_id(99) is None
    assert len(world.entities_at(2, 3)) == 1
    assert entities[0].display_name == "Snake 3"
    assert entities[0].head == (0, 0)
    assert tuple(entities[1].cells) == ((2, 3), (1, 3))


def _test_frame_basics():
    events = (ViewEvent(kind="spawn", tick=0, actor_id=None, target_id=3, data={"preset": "snake"}),)
    frame = ViewFrame(
        world=_mini_world(),
        events=events,
        population={"snake": 1},
        episode=0,
        step=0,
        agent=AgentEphemera(active=True, reward=1.5),
    )
    assert frame.episode == 0 and frame.step == 0
    assert frame.agent.active and frame.agent.reward == 1.5
    assert frame.world.terrain.shape == (3, 3)


def _test_world_render():
    frame = _mini_frame()
    world = WorldRenderer().render(frame, scale=2)
    assert world.shape == (6, 6, 3) and world.dtype == np.uint8
    selected = WorldRenderer().render(frame, scale=1, selected_id=5)
    assert selected.shape == (3, 3, 3)


def _test_overlay_render():
    frame = _mini_frame()
    world = WorldRenderer().render(frame, scale=4)
    overlay = OverlayRenderer().render(frame, world)
    assert overlay.ndim == 3 and overlay.shape[1] == world.shape[1]
    assert overlay.shape[0] > world.shape[0]


def _test_observation_render():
    from .renderers import ObservationRenderer

    renderer = ObservationRenderer()
    assert renderer.render_array(np.zeros((4, 4), dtype=np.float32)).shape == (16, 16, 3)
    assert renderer.render_array(np.arange(6, dtype=np.float32), scale=2).shape == (2, 12, 3)
    assert renderer.render_array(np.zeros((2, 4, 3), dtype=np.float32), scale=1).shape == (2, 4, 3)


def _test_log_format():
    log = EventLog(capacity=8)
    world = _mini_world()
    events = (
        ViewEvent(kind="consume", tick=1, actor_id=5, target_id=9, data={"category": "food"}),
        ViewEvent(kind="predation", tick=2, actor_id=3, target_id=5),
        ViewEvent(kind="death", tick=3, actor_id=5, data={"reason": "predation"}),
        ViewEvent(kind="reproduce", tick=4, actor_id=3, target_id=12, data={"parents": [3, 7]}),
        ViewEvent(kind="weird", tick=5, actor_id=7),
    )
    pushed = log.push_many(events, world)
    assert pushed == 5
    lines = log.lines()
    assert any("[t=1]" in line and "ate" in line for line in lines)
    assert any("died" in line for line in lines)
    assert any("born" in line for line in lines)
    assert any("weird" in line for line in lines)


def _test_genealogy():
    gen = Genealogy()
    gen.update(
        (
            ViewEvent(kind="reproduce", tick=0, actor_id=3, target_id=12, data={"parents": (3, 7)}),
            ViewEvent(kind="reproduce", tick=0, actor_id=3, target_id=13, data={"parents": (3, 7)}),
        )
    )
    assert gen.lineage(12) == ((3, 7), 0)
    assert gen.offspring[3] == 2 and gen.offspring[7] == 2


def _test_population():
    tracker = PopulationTracker()
    frame = _mini_frame()
    frame = ViewFrame(
        world=_mini_world(extra=ViewEntity(id=12, kind="snake", species="snake", x=0, y=1)),
        events=(
            ViewEvent(kind="reproduce", tick=1, actor_id=3, target_id=12, data={"parents": [3, 7]}),
            ViewEvent(kind="death", tick=1, actor_id=5),
            ViewEvent(kind="consume", tick=1, actor_id=3, target_id=9),
        ),
    )
    tracker.update(frame)
    summary = tracker.summary()
    assert summary["births"] == 1 and summary["deaths"] == 1 and summary["food_consumed"] == 1
    assert summary["total"] == 4
    assert summary["by_species"]["snake"] == 3


def _test_inspector():
    gen = Genealogy()
    inspector = Inspector(gen)
    frame = _mini_frame()
    report = inspector.inspect(frame, 5)
    assert report.alive and report.species == "snake"
    assert report.get("size") is not None
    assert report.entity_id == 5
    dead = inspector.inspect(frame, 999, memory={})
    assert not dead.alive


def _test_snapshot_to_frame():
    snap = _FakeSnapshot()
    frame = snapshot_to_frame(snap, mode="idle")
    assert frame.world.width == 3 and frame.world.tick == 2
    assert len(frame.events) == 1
    assert frame.world.entity_by_id(5).species == "snake"
    assert frame.mode == "idle"
    ephemera = snapshot_to_frame(snap, agent=AgentEphemera(active=True, mode="train", reward=2.0))
    assert ephemera.mode == "train" and ephemera.agent.reward == 2.0


def _test_playback():
    source = StaticFrameSource(_mini_frame())
    controller = PlaybackController(source, speed=0.5)
    assert controller.speed == 0.5
    assert controller.set_speed(999) in PLAY_SPEEDS
    speed = controller.cycle_speed()
    assert speed in PLAY_SPEEDS
    controller.fast_forward()
    assert controller.is_fast()
    controller.pause()
    assert controller.toggle_play() is True
    controller.set_speed(10.0)
    assert controller.steps_per_tick(0.1) >= 1


def _test_recorder():
    recorder = Recorder()
    recorder.attach(rgb=lambda: np.full((2, 2, 3), 255, dtype=np.uint8), ascii=lambda: "ab\ncd")
    recorder.set_rgb_recording(True)
    recorder.set_ascii_recording(True)
    recorder.capture()
    assert recorder.frame_count() == 1 and recorder.line_count() == 1
    with tempfile.TemporaryDirectory() as tmp:
        npz = os.path.join(tmp, "f.npz")
        recorder.save_rgb(npz)
        with np.load(npz) as payload:
            assert payload["frames"].shape == (1, 2, 2, 3)
        ascii_path = os.path.join(tmp, "t.txt")
        recorder.save_ascii(ascii_path)
        assert os.path.getsize(ascii_path) > 0
    empty = Recorder()
    try:
        empty.save_ascii("unused")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError saving with no lines")


# ------------------------------------------------------------------ fixtures


def _mini_world(extra: ViewEntity | None = None) -> ViewWorld:
    terrain = np.zeros((3, 3), dtype=np.int8)
    resources = np.zeros((3, 3), dtype=np.int8)
    entities = [
        ViewEntity(id=3, kind="snake", species="snake", x=0, y=0, direction=(1, 0), size=1.0, predator=True),
        ViewEntity(id=5, kind="snake", species="snake", x=1, y=0, health=50.0, max_health=100.0, age=3.0),
        ViewEntity(id=7, kind="prey", species="prey", x=2, y=2, edible=True),
    ]
    if extra is not None:
        entities.append(extra)
    return ViewWorld(
        width=3,
        height=3,
        wrap=False,
        tick=1,
        episode=0,
        terrain=terrain,
        resources=resources,
        entities=tuple(entities),
        population={"snake": 2, "prey": 1},
    )


def _mini_frame() -> ViewFrame:
    return ViewFrame(
        world=_mini_world(),
        events=(ViewEvent(kind="spawn", tick=0, target_id=3, data={"preset": "snake"}),
                ViewEvent(kind="consume", tick=1, actor_id=3, target_id=9, data={"category": "food"})),
        population={"snake": 2, "prey": 1},
        episode=0,
        step=1,
        agent=AgentEphemera(active=True, mode="train", agent_id=3, reward=0.5),
    )


class _FakeSnapshot:
    tick = 2
    episode = 1
    width = 3
    height = 3
    wrap = False
    entities = (
        type("E", (), {"id": 5, "kind": "snake", "species": "snake", "x": 1, "y": 1, "alive": True,
                      "size": 1.0, "age": 0.0, "health": None, "max_health": None,
                      "energy": None, "max_energy": None, "lifetime": None, "velocity": None,
                      "direction": (1, 0), "body": (), "genome": {}, "controller": "agent",
                      "edible": False, "obstacle": False, "solid": True, "movable": True,
                      "predator": False, "state": None, "action": None})(),
    )
    by_id = {5: entities[0]}
    events = (type("EV", (), {"kind": "move", "tick": 2, "actor_id": 5, "target_id": None, "data": {}})(),)

    class _Layers:
        terrain = np.zeros((3, 3), dtype=np.int8)
        resources = np.zeros((3, 3), dtype=np.int8)

        @property
        def shape(self):
            return self.terrain.shape

    layers = _Layers()
    population = {"snake": 1}


# ------------------------------------------------------------------ debug report


def debug_report(visualizer) -> dict[str, Any]:
    """Aggregate the visualizer's live internal state into a plain dict."""
    frame = visualizer.current_frame()
    payload = {
        "package": "alifelab.view",
        "episode": frame.episode,
        "step": frame.step,
        "mode": frame.mode,
        "playing": visualizer.controller.is_playing(),
        "speed": visualizer.controller.speed,
        "world_size": [frame.world.width, frame.world.height],
        "population": visualizer.population.summary(),
        "focused": visualizer.focused_id,
        "help_shown": visualizer.show_help,
        "overlay": visualizer.show_overlay,
        "panels": visualizer.show_panels,
        "rgb_recording": visualizer.recorder.rgb_recording,
        "ascii_recording": visualizer.recorder.ascii_recording,
        "log_lines": visualizer.event_log.message_count,
        "debug_report_path": visualizer.report_path,
    }
    return payload


def debug_overview(visualizer) -> list[str]:
    """Human-readable one-screen digest of the visualizer run."""
    report = debug_report(visualizer)
    pop = report["population"]
    species = ", ".join(f"{k}:{v}" for k, v in pop["by_species"].items()) or "none"
    return [
        "=" * 56,
        "ALIFELAB VIEW DEBUG OVERVIEW",
        f"  mode={report['mode']}  episode={report['episode']}  step={report['step']}  "
        f"playing={report['playing']}  speed={report['speed']}",
        f"  world={report['world_size'][0]}x{report['world_size'][1]}  population={pop['total']}  "
        f"({species})",
        f"  births={pop['births']}  deaths={pop['deaths']}  predation={pop['predation']}  "
        f"repro={pop['reproduction']}  food={pop['food_consumed']}  spawns={pop['spawns']}  "
        f"mutations={pop['mutations']}  attacks={pop['attack_attempts']}",
        f"  focused_entity={report['focused']}  help={report['help_shown']}  "
        f"overlay={report['overlay']}  panels={report['panels']}",
        f"  rgb_recording={report['rgb_recording']}  ascii_recording={report['ascii_recording']}  "
        f"log={report['log_lines']}",
        f"  debug_report_file={report['debug_report_path']}",
        "-" * 56,
    ]