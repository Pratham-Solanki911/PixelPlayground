"""The interactive visualizer facade.

The visualizer is engine-free: it drives any :class:`FrameSource` (a live
simulation via :class:`SnapshotSource`, a saved recording, a scripted demo),
renders :class:`ViewFrame` objects with the generic renderers, and pipes events
into the event log, population tracker, genealogy, and inspector. It owns
playback state, mode cycling, keyboard bindings, selection, and recording -- but
never imports the simulation, RL code, Trainer, or Agent.

The GUI (pygame) layer is optional and guarded: a headless ``tick()`` loop
works on machines without pygame, which is how the debug self-tests run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import palette as pal
from .controller import PlaybackController
from .debug import debug_overview, debug_report, debug_tests
from .events import EventLog, Genealogy
from .gui import DashboardPanel, EventPanel, InspectorPanel, LineagePanel, stack_horizontal
from .inspector import Inspector
from .model import ViewEntity, ViewFrame
from .population import PopulationTracker
from .recording import Recorder
from .renderers import OverlayRenderer, WorldRenderer
from .sources import FrameSource

MODES: tuple[str, ...] = ("train", "evaluate", "record", "idle")

HELP_LINES: tuple[str, ...] = (
    "space  pause/play    right/s  step once    r  reset episode",
    "f  fast-forward      e/c      speed up/down   m  cycle mode",
    "1-9   focus agent    z  select focused agent    x  clear selection",
    "o  toggle header overlay    p  toggle panels    h  toggle this help",
    "g  debug overview    d  save debug report    l  save event log",
    "t  rgb recording on/off    b  save rgb npz    a  ascii recording on/off",
    "q/esc  quit",
)

# pygame key bytes -> action name (used by the GUI loop; headless tests call
# handle_action directly).
_KEY_CODES: dict[str, str] = {
    "hover": "none",
    "space": "play",
    "enter": "play",
    "right": "step",
    "s": "step",
    "r": "reset",
    "f": "fast",
    "e": "speed_up",
    "c": "speed_down",
    "m": "mode",
    "o": "overlay",
    "p": "panels",
    "h": "help",
    "g": "debug",
    "d": "save_debug",
    "l": "save_log",
    "t": "rgb_recording",
    "b": "save_rgb",
    "a": "ascii_recording",
    "z": "focus_agent",
    "x": "clear",
    "q": "quit",
    "esc": "quit",
}


class Visualizer:
    """Drives a frame source, renders it, and exposes playback + data panels."""

    def __init__(
        self,
        source: FrameSource,
        *,
        speed: float = 1.0,
        mode: str = "train",
        world_scale: int | None = None,
        max_log: int = 2000,
        recording_dir: str = "runs/view",
        auto_retry_episode: bool = True,
        headless: bool = False,
    ):
        if mode not in MODES:
            mode = MODES[0]
        self.controller = PlaybackController(source, speed=speed, mode=mode, auto_retry_episode=auto_retry_episode)
        self.headless = bool(headless)
        self.world_scale = world_scale

        # data panels (engine-free)
        self.event_log = EventLog(capacity=max_log)
        self.genealogy = Genealogy()
        self.population = PopulationTracker()
        self.inspector = Inspector(self.genealogy)

        # renderers
        self.renderer = WorldRenderer()
        self.overlay = OverlayRenderer()

        # gui panels
        self.dashboard = DashboardPanel(self.population)
        self.events_panel = EventPanel(title="Events")
        self.inspector_panel = InspectorPanel()
        self.lineage_panel = LineagePanel(self.genealogy)

        # state
        self.focused_id: int | None = None
        self.memory: dict[int, ViewEntity] = {}
        self.show_help = False
        self.show_overlay = True
        self.show_panels = True
        self.last_reward_agent: int | None = None

        # recording (providers supplied by a driver phase)
        self.recorder = Recorder()
        self.recording_dir = Path(recording_dir)
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.report_path: str | None = None

        self._frame: ViewFrame | None = None
        self._display_frame: np.ndarray | None = None

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, source: FrameSource, **cfg: Any) -> "Visualizer":
        """Construct a fully-wired visualizer with sanity assertions.

        The assertions form the reproducible "graph" contract: if any required
        piece is missing, the visualizer refuses to start with a clear error.
        """
        visualizer = cls(source, **cfg)
        checks = (
            ("frame_source", source is not None and callable(getattr(source, "frame", None))),
            ("event_log", visualizer.event_log is not None),
            ("population", visualizer.population is not None),
            ("genealogy", visualizer.genealogy is not None),
            ("inspector", visualizer.inspector is not None),
            ("renderer", visualizer.renderer is not None),
            ("recorder", visualizer.recorder is not None),
        )
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise RuntimeError(f"Visualizer wiring incomplete: missing {', '.join(failed)}")
        visualizer.verify()
        return visualizer

    def verify(self) -> bool:
        """Probe the current source frame once; store it as the render target."""
        self._frame = self.controller.source.frame()
        self._refresh()
        return True

    # ---------------------------------------------------------------- current

    def current_frame(self) -> ViewFrame:
        if self._frame is None:
            self._frame = self.controller.source.frame()
        return self._frame

    def display_frame(self) -> np.ndarray:
        if self._display_frame is None:
            self.render_display()
        return self._display_frame

    # ----------------------------------------------------------------- refresh

    def _refresh(self) -> None:
        frame = self.current_frame()
        self.event_log.push_many(frame.events, frame.world)
        self.genealogy.update(frame.events)
        self.population.update(frame)
        agent = frame.agent
        if agent and agent.agent_id is not None and agent.reward is not None:
            self.inspector.rewards[agent.agent_id] = self.inspector.rewards.get(agent.agent_id, 0.0) + agent.reward
            self.last_reward_agent = agent.agent_id
        for entity in frame.world.entities:
            self.memory[entity.id] = entity
        self.focused_id = self._resident_selection(self.focused_id)

    def _resident_selection(self, focused_id: int | None) -> int | None:
        if focused_id is None:
            return None
        frame = self.current_frame()
        if frame.world.entity_by_id(focused_id) is not None:
            return focused_id
        return self.memory.get(focused_id).id if focused_id in self.memory else None

    # --------------------------------------------------------------- rendering

    def render_display(self) -> np.ndarray:
        frame = self.current_frame()
        w, h = frame.world.width, frame.world.height
        scale = self.world_scale or max(2, min(28, 760 // max(w, h, 1)))
        world = self.renderer.render(frame, scale=scale, selected_id=self.focused_id)
        width = world.shape[1]
        chunks = [world]
        if self.show_overlay:
            chunks.insert(0, self.overlay.render(frame, world))
        if self.show_panels:
            chunks.append(self._status_strip(width))
        if self.show_help:
            chunks.append(self._help_strip(width))
        image = np.vstack(chunks) if chunks else world
        self._display_frame = image
        return image

    def _status_strip(self, width: int) -> np.ndarray:
        report = self.inspector.inspect(self.current_frame(), self.focused_id, memory=self.memory)
        self.inspector_panel.set_report(report if self.focused_id is not None else None)
        self.lineage_panel.set_target(self.focused_id, report.name)
        events = self.events_panel.set_lines(self.event_log.lines(-8))
        panels: list = [self.dashboard]
        if self.focused_id is not None:
            panels += [self.inspector_panel, self.lineage_panel]
        else:
            panels += [events]
        height = 12 * (7 * 2 + 2) + 24
        return stack_horizontal(panels, width, height)

    def _help_strip(self, width: int) -> np.ndarray:
        fs = 2
        cell_h = 7 * fs + fs
        lines = list(HELP_LINES)
        image = np.full((len(lines) * cell_h + fs, width, 3), (16, 18, 22), dtype=np.uint8)
        from .font import draw_text

        for i, line in enumerate(lines):
            draw_text(image, fs // 2, i * cell_h + fs // 2, line, (205, 210, 215), scale=fs)
        return image

    # ----------------------------------------------------------------- manual

    def step_once(self) -> None:
        self.controller.step_once()
        self._frame = None
        self._refresh()
        self.recorder.capture()

    def reset_episode(self) -> None:
        self.controller.reset_episode()
        self._frame = None
        self._refresh()

    def cycle_mode(self) -> str:
        index = MODES.index(self.controller.mode)
        mode = MODES[(index + 1) % len(MODES)]
        self.controller.set_mode(mode)
        return mode

    def focus(self, entity_id: int | None) -> None:
        self.focused_id = self._resident_selection(entity_id)

    def focus_agent(self, index: int) -> int | None:
        """Focus the agent whose position in ``agent_ids`` is ``index`` (0-based)."""
        frame = self.current_frame()
        agent_ids = frame.agent.agent_ids or ()
        if agent_ids:
            if 0 <= index < len(agent_ids):
                self.focus(int(agent_ids[index]))
                return self.focused_id
            return self.focused_id
        agent_id = frame.agent.agent_id
        if agent_id is not None:
            self.focus(agent_id)
        return self.focused_id

    def pick(self, x: int, y: int) -> int | None:
        """Select the entity at world cell (x, y)."""
        frame = self.current_frame()
        world = frame.world
        if not (0 <= x < world.width and 0 <= y < world.height):
            return self.focused_id
        entities = world.entities_at(x, y)
        if not entities:
            return self.focused_id
        if self.focused_id in {e.id for e in entities}:
            self.focus(None)
            return None
        based_on_agent = [e for e in entities if e.is_agent] or list(entities)
        self.focus(based_on_agent[0].id)
        return self.focused_id

    # --------------------------------------------------------------- recording

    def set_rgb_provider(self, provider) -> None:
        self.recorder.attach(rgb=provider)

    def set_ascii_provider(self, provider) -> None:
        self.recorder.attach(ascii=provider)

    def toggle_rgb_recording(self) -> bool:
        enabled = not self.recorder.rgb_recording
        self.recorder.set_rgb_recording(enabled)
        return enabled

    def toggle_ascii_recording(self) -> bool:
        enabled = not self.recorder.ascii_recording
        self.recorder.set_ascii_recording(enabled)
        return enabled

    def save_rgb(self, name: str = "video") -> Path:
        path = self.recording_dir / f"{name}_{self.current_frame().episode}_{self.current_frame().step}.npz"
        self.recorder.save_rgb(str(path))
        return path

    def save_ascii(self, name: str = "log") -> Path:
        path = self.recording_dir / f"{name}_{self.current_frame().episode}_{self.current_frame().step}.txt"
        self.recorder.save_ascii(str(path))
        return path

    def save_log(self) -> Path:
        path = self.recording_dir / f"eventlog_{self.current_frame().episode}_{self.current_frame().step}.txt"
        path.write_text("\n".join(self.event_log.lines()), encoding="utf-8")
        return path

    # ------------------------------------------------------------------- keys

    def handle_key(self, name: str, value: Any = None) -> dict[str, Any]:
        """Map a normalized key action to a behavior. Returns effect info."""
        action = _KEY_CODES.get(name, name)
        effects: dict[str, Any] = {"action": action}
        if action == "play":
            effects["playing"] = self.controller.toggle_play()
        elif action == "step":
            self.step_once()
            effects["stepped"] = self.current_frame().step
        elif action == "reset":
            self.reset_episode()
        elif action == "fast":
            self.controller.fast_forward()
            effects["playing"] = True
        elif action == "speed_up":
            effects["speed"] = self.controller.cycle_speed(1)
        elif action == "speed_down":
            effects["speed"] = self.controller.cycle_speed(-1)
        elif action == "mode":
            effects["mode"] = self.cycle_mode()
        elif action == "overlay":
            self.show_overlay = not self.show_overlay
            effects["show_overlay"] = self.show_overlay
        elif action == "panels":
            self.show_panels = not self.show_panels
            effects["show_panels"] = self.show_panels
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "debug":
            effects["debug"] = debug_overview(self)
        elif action == "save_debug":
            path = self.save_debug_report()
            effects["saved"] = str(path)
        elif action == "save_log":
            effects["saved"] = str(self.save_log())
        elif action == "rgb_recording":
            effects["rgb_recording"] = self.toggle_rgb_recording()
        elif action == "save_rgb":
            if self.recorder.rgb_provider is None:
                effects["error"] = "no RGB provider attached"
            else:
                effects["saved"] = str(self.save_rgb())
        elif action == "ascii_recording":
            effects["ascii_recording"] = self.toggle_ascii_recording()
        elif action == "focus_agent":
            effects["focused"] = self.focus_agent(0)
        elif action == "clear":
            self.focus(None)
        elif action == "quit":
            effects["quit"] = True
        return effects

    # ------------------------------------------------------------------- debug

    def debug_report(self) -> dict[str, Any]:
        return debug_report(self)

    def debug_overview(self) -> list[str]:
        return debug_overview(self)

    def save_debug_report(self) -> Path:
        path = self.recording_dir / f"debug_{self.current_frame().episode}_{self.current_frame().step}.txt"
        path.write_text("\n".join(debug_overview(self)) + "\n", encoding="utf-8")
        self.report_path = str(path)
        return path

    # -------------------------------------------------------------------- loop

    def run(self, frames: int | None = None, headless: bool | None = None) -> dict[str, Any]:
        """Main loop. ``headless=True`` runs without a window (used by tests).

        Returns a summary dict. In GUI mode requires ``pygame`` (optional
        extra ``alifelab[ui]``).
        """
        headless = self.headless if headless is None else headless
        if headless:
            return self._run_headless(frames)
        return self._run_pygame(frames)

    def _run_headless(self, frames: int | None) -> dict[str, Any]:
        if frames is None:
            frames = 10
        for _ in range(max(0, int(frames))):
            self.step_once()
        return self.summary()

    def _run_pygame(self, frames: int | None) -> dict[str, Any]:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "the interactive visualizer needs pygame; install it with `pip install alifelab[ui]` "
                "(headless mode is available if you set headless=True)"
            ) from exc
        pygame.init()
        self._refresh()
        image = self.display_frame()
        screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
        clock = pygame.time.Clock()
        running = True
        steps = 0
        while running and (frames is None or steps < frames):
            interval = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(self._pygame_key(event.key))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
            if self.controller.is_playing():
                for _ in range(self.controller.steps_per_tick(interval)):
                    self.step_once()
                    steps += 1
                    if frames is not None and steps >= frames:
                        break
            image = self.render_display()
            surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        pygame.quit()
        return self.summary()

    def _pygame_key(self, key: int) -> str:  # pragma: no cover - pygame only
        import pygame

        special = {
            pygame.K_SPACE: "space",
            pygame.K_RETURN: "enter",
            pygame.K_RIGHT: "right",
            pygame.K_ESCAPE: "esc",
        }
        if key in special:
            return special[key]
        ch = pygame.key.name(key)
        return ch.lower()

    def _handle_click(self, pos: tuple[int, int]) -> None:  # pragma: no cover - pygame only
        frame = self.current_frame()
        scale = self.world_scale or max(2, min(28, 760 // max(frame.world.width, frame.world.height, 1)))
        height_offset = 0
        if self.show_overlay:
            height_offset += self.overlay.render(frame, self.renderer.render(frame, scale=1)).shape[0] - frame.world.height * scale
        world_h = frame.world.height * scale
        if not (height_offset <= pos[1] < height_offset + world_h):
            return
        x, y = pos[0] // scale, (pos[1] - height_offset) // scale
        self.pick(int(x), int(y))

    # ----------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        frame = self.current_frame()
        return {
            "mode": frame.mode,
            "episode": frame.episode,
            "step": frame.step,
            "playing": self.controller.is_playing(),
            "speed": self.controller.speed,
            "population_total": self.population.total,
            "events_logged": self.event_log.message_count,
            "focused": self.focused_id,
        }

    def help_lines(self) -> list[str]:
        return list(HELP_LINES)

    @staticmethod
    def run_selftests(verbose: bool = False) -> list[str]:
        return debug_tests(verbose=verbose)