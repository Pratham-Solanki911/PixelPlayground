"""End-to-end tests for the engine-free visualizer package.

Run with either::

    python -m unittest discover -s tests -v
    pytest tests/test_view.py -v

No simulation, RL, pygame, or torch are touched -- these tests verify the
view package behaves correctly in isolation (the engine-free contract).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from alifelab.view import (
    DashboardPanel,
    EventLog,
    Genealogy,
    Inspector,
    OverlayRenderer,
    PLAY_SPEEDS,
    PlaybackController,
    PopulationTracker,
    Recorder,
    SnapshotSource,
    StaticFrameSource,
    Visualizer,
    WorldRenderer,
    debug_tests,
    snapshot_to_frame,
)
from alifelab.view.model import (
    AgentEphemera,
    ViewEntity,
    ViewEvent,
    ViewFrame,
    ViewWorld,
)


def _mini_world(extra: ViewEntity | None = None) -> ViewWorld:
    terrain = np.zeros((3, 3), dtype=np.int8)
    resources = np.zeros((3, 3), dtype=np.int8)
    entities = [
        ViewEntity(id=3, kind="snake", species="snake", x=0, y=0, direction=(1, 0), predator=True, size=1.0),
        ViewEntity(id=5, kind="snake", species="snake", x=1, y=0, health=50.0, max_health=100.0, age=3.0, controller="agent"),
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
        events=(
            ViewEvent(kind="spawn", tick=0, target_id=3, data={"preset": "snake"}),
            ViewEvent(kind="consume", tick=1, actor_id=3, target_id=9, data={"category": "food"}),
        ),
        population={"snake": 2, "prey": 1},
        episode=0,
        step=1,
        agent=AgentEphemera(active=True, mode="train", agent_id=3, reward=0.5),
    )


class ModelTests(unittest.TestCase):
    def test_world_lookup(self):
        world = _mini_world()
        self.assertEqual(world.total_population, 3)
        self.assertIsNotNone(world.entity_by_id(5))
        self.assertIsNone(world.entity_by_id(99))
        self.assertEqual(len(world.entities_at(2, 2)), 1)

    def test_display_name_agents(self):
        entity = _mini_world().entity_by_id(5)
        self.assertEqual(entity.display_name, "Snake 5")
        self.assertEqual(entity.is_agent, True)

    def test_frame_defaults(self):
        frame = _mini_frame()
        self.assertEqual(frame.episode, 0)
        self.assertEqual(frame.step, 1)
        self.assertTrue(frame.agent.active)


class RendererTests(unittest.TestCase):
    def test_world_size_and_selection(self):
        frame = _mini_frame()
        image = WorldRenderer().render(frame, scale=3)
        self.assertEqual(image.shape, (9, 9, 3))
        selected = WorldRenderer().render(frame, scale=3, selected_id=5)
        base = WorldRenderer().render(frame, scale=3)
        self.assertFalse(np.array_equal(selected, base))  # selection ring added

    def test_overlay_constructs(self):
        frame = _mini_frame()
        world = WorldRenderer().render(frame, scale=4)
        overlay = OverlayRenderer(font_scale=2).render(frame, world)
        self.assertGreater(overlay.shape[0], world.shape[0])
        self.assertEqual(overlay.shape[1], world.shape[1])

    def test_dashboard_panel_lines(self):
        tracker = PopulationTracker()
        tracker.update(_mini_frame())
        panel = DashboardPanel(tracker)
        self.assertTrue(any("snake" in line.lower() for line in panel.lines()))


class EventGenealogyTests(unittest.TestCase):
    def test_log_formatting_and_capacity(self):
        log = EventLog(capacity=4)
        world = _mini_world()
        events = (
            ViewEvent(kind="consume", tick=1, actor_id=3, target_id=9, data={"category": "food"}),
            ViewEvent(kind="death", tick=2, actor_id=5, data={"reason": "predation"}),
            ViewEvent(kind="reproduce", tick=3, actor_id=3, target_id=12, data={"parents": [3, 7]}),
            ViewEvent(kind="spawn", tick=4, target_id=12, data={"preset": "snake"}),
        )
        log.push_many(events, world)
        self.assertEqual(log.message_count, 4)
        lines = log.lines()
        self.assertIn("[t=1]", lines[0])
        # capacity trimmed: adding more keeps only the newest
        log.push_many((ViewEvent(kind="mutate", tick=5, actor_id=3, data={"traits": [1]}),), world)
        self.assertEqual(len(log.entries), 4)
        self.assertTrue(any("[t=5]" in line for line in log.lines()))

    def test_genealogy(self):
        gen = Genealogy()
        gen.update(
            (
                ViewEvent(kind="reproduce", tick=0, actor_id=3, target_id=12, data={"parents": (3, 7)}),
                ViewEvent(kind="reproduce", tick=0, actor_id=3, target_id=13, data={"parents": (3, 7)}),
            )
        )
        parents, _ = gen.lineage(12)
        self.assertEqual(parents, (3, 7))
        self.assertEqual(gen.offspring[3], 2)


class PopulationInspectorTests(unittest.TestCase):
    def test_population_counts(self):
        tracker = PopulationTracker()
        frame = _mini_frame()
        frame = ViewFrame(
            world=_mini_world(extra=ViewEntity(id=12, kind="snake", species="snake", x=0, y=1)),
            events=(
                ViewEvent(kind="reproduce", tick=1, actor_id=3, target_id=12, data={"parents": [3, 7]}),
                ViewEvent(kind="death", tick=1, actor_id=5),
                ViewEvent(kind="predation", tick=1, actor_id=3, target_id=7),
            ),
        )
        tracker.update(frame)
        summary = tracker.summary()
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["births"], 1)
        self.assertEqual(summary["deaths"], 1)
        self.assertEqual(summary["predation"], 1)

    def test_inspector_alive_and_dead(self):
        inspector = Inspector(Genealogy())
        frame = _mini_frame()
        report = inspector.inspect(frame, 5)
        self.assertTrue(report.alive)
        self.assertEqual(report.get("species"), "snake")
        dead = inspector.inspect(frame, 999, memory={})
        self.assertFalse(dead.alive)


class AdapterSourceControllerTests(unittest.TestCase):
    def test_snapshot_to_frame(self):
        class FakeLayer:
            terrain = np.zeros((2, 2), dtype=np.int8)
            resources = np.zeros((2, 2), dtype=np.int8)

            @property
            def shape(self):
                return self.terrain.shape

        Entity = type(
            "E",
            (),
            {
                "id": 1,
                "kind": "snake",
                "species": "snake",
                "x": 1,
                "y": 1,
                "alive": True,
                "size": 1.0,
                "age": 0.0,
                "health": None,
                "max_health": None,
                "energy": None,
                "max_energy": None,
                "lifetime": None,
                "velocity": None,
                "direction": (1, 0),
                "body": (),
                "genome": {},
                "controller": "agent",
                "edible": False,
                "obstacle": False,
                "solid": True,
                "movable": True,
                "predator": False,
                "state": None,
                "action": None,
            },
        )()
        Event = type("EV", (), {"kind": "spawn", "tick": 0, "actor_id": None, "target_id": 1, "data": {}})()

        class FakeSnapshot:
            tick = 1
            episode = 2
            width = 2
            height = 2
            wrap = False
            entities = (Entity,)
            by_id = {1: Entity}
            events = (Event,)
            layers = FakeLayer()
            population = {"snake": 1}

        frame = snapshot_to_frame(FakeSnapshot(), mode="idle")
        self.assertEqual(frame.world.width, 2)
        self.assertEqual(frame.world.tick, 1)
        self.assertEqual(frame.episode, 2)
        self.assertEqual(len(frame.events), 1)
        self.assertEqual(frame.world.entity_by_id(1).controller, "agent")

    def test_snapshot_source_steps(self):
        calls = {"step": 0, "reset": 0}

        class FakeSession:
            def __init__(self):
                self.tick = 0

            def snapshot(self):
                entity = _make_entity(self.tick)
                Event = type("EV", (), {"kind": "move", "tick": self.tick, "actor_id": 1, "target_id": None, "data": {}})()
                layers = type("L", (), {"terrain": np.zeros((3, 3), dtype=np.int8), "resources": np.zeros((3, 3), dtype=np.int8)})()
                return type("S", (), {
                    "tick": self.tick, "episode": 0, "width": 3, "height": 3, "wrap": False,
                    "entities": (entity,), "by_id": {1: entity}, "events": (Event,),
                    "layers": layers, "population": {"snake": 1},
                })

            def step(self):
                self.tick += 1

            def reset(self):
                self.tick = 0

        source = SnapshotSource(FakeSession())
        source.step_once()
        self.assertEqual(source.frame().world.tick, 1)

    def test_playback_speed_and_steps(self):
        source = StaticFrameSource(_mini_frame())
        controller = PlaybackController(source, speed=0.5, mode="train")
        self.assertEqual(controller.speed, 0.5)
        self.assertIn(controller.cycle_speed(1), PLAY_SPEEDS)
        controller.set_speed(10.0)
        self.assertGreaterEqual(controller.steps_per_tick(0.2), 1)
        self.assertTrue(controller.toggle_play())
        controller.fast_forward()
        self.assertTrue(controller.is_fast())


class RecorderTests(unittest.TestCase):
    def test_rgb_ascii_roundtrip(self):
        recorder = Recorder()
        recorder.attach(
            rgb=lambda: np.full((2, 4, 3), 200, dtype=np.uint8),
            ascii=lambda: "..\n..",
        )
        recorder.set_rgb_recording(True)
        recorder.set_ascii_recording(True)
        recorder.capture()
        recorder.capture()
        self.assertEqual(recorder.frame_count(), 2)
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "v.npz"
            recorder.save_rgb(str(npz))
            with np.load(npz) as payload:
                self.assertEqual(payload["frames"].shape, (2, 2, 4, 3))
            ascii_path = Path(tmp) / "t.txt"
            recorder.save_ascii(str(ascii_path))
            self.assertGreater(ascii_path.stat().st_size, 0)

    def test_save_without_capture_raises(self):
        with self.assertRaises(ValueError):
            Recorder().save_rgb("unused.npz")


class VisualizerTests(unittest.TestCase):
    def test_headless_step_loop(self):
        source = StaticFrameSource(_mini_frame())
        viz = Visualizer.build(source, headless=True, recording_dir=tempfile.mkdtemp())
        out = viz.run(frames=3, headless=True)
        self.assertIn("mode", out)
        self.assertIn("step", out)

    def test_key_bindings_headless(self):
        viz = Visualizer(StaticFrameSource(_mini_frame()), headless=True, recording_dir=tempfile.mkdtemp())
        viz.verify()
        self.assertEqual(viz.handle_key("play")["playing"], True)
        self.assertIn("speed", viz.handle_key("speed_up"))
        self.assertEqual(viz.handle_key("mode")["mode"], "evaluate")
        viz.focus(5)
        self.assertEqual(viz.focused_id, 5)
        viz.handle_key("clear")
        self.assertIsNone(viz.focused_id)
        self.assertFalse(viz.show_help)
        viz.handle_key("help")
        self.assertTrue(viz.show_help)

    def test_selection_pick(self):
        viz = Visualizer(StaticFrameSource(_mini_frame()), headless=True, recording_dir=tempfile.mkdtemp())
        viz.verify()
        viz.pick(2, 2)
        self.assertEqual(viz.focused_id, 7)

    def test_render_display_headless(self):
        viz = Visualizer.build(StaticFrameSource(_mini_frame()), headless=True, recording_dir=tempfile.mkdtemp())
        viz.verify()
        image = viz.render_display()
        self.assertTrue(image.ndim == 3 and image.shape[2] == 3)
        self.assertGreater(image.shape[0], 0)

    def test_debug_report_and_overview(self):
        viz = Visualizer(StaticFrameSource(_mini_frame()), headless=True, recording_dir=tempfile.mkdtemp())
        viz.verify()
        report = viz.debug_report()
        self.assertEqual(report["world_size"], [3, 3])
        overview = viz.debug_overview()
        self.assertTrue(any("OVERVIEW" in line for line in overview))

    def test_selftests_pass(self):
        lines = debug_tests()
        summary = lines[-1]
        self.assertIn("checks passed", summary)
        self.assertNotIn("FAIL", "\n".join(lines))


def _make_entity(tick):
    return type(
        "E",
        (),
        {
            "id": 1,
            "kind": "snake",
            "species": "snake",
            "x": (1 + tick) % 3,
            "y": 1,
            "alive": True,
            "size": 1.0,
            "age": float(tick),
            "health": None,
            "max_health": None,
            "energy": None,
            "max_energy": None,
            "lifetime": None,
            "velocity": None,
            "direction": (1, 0),
            "body": (),
            "genome": {},
            "controller": "agent",
            "edible": False,
            "obstacle": False,
            "solid": True,
            "movable": True,
            "predator": False,
            "state": None,
            "action": None,
        },
    )()


if __name__ == "__main__":
    unittest.main(verbosity=2)