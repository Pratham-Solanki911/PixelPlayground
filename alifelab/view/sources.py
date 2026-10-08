"""Frame sources: the thin interface the visualizer drives.

The visualizer never imports the simulation. It drives anything implementing
:class:`FrameSource` -- a live simulation harness (via
:class:`SnapshotSource`), a saved recording, or a scripted demo
(:class:`StaticFrameSource`).
"""
from __future__ import annotations

from typing import Callable, Protocol

from .adapter import snapshot_to_frame
from .model import AgentEphemera, ViewFrame


class FrameSource(Protocol):
    """Anything that can produce a fresh :class:`ViewFrame` and advance."""

    def frame(self) -> ViewFrame: ...
    def step_once(self) -> None: ...
    def reset(self) -> None: ...
    def set_mode(self, mode: str) -> None: ...


class SnapshotSource:
    """Live source wrapping a duck-typed engine/env with ``snapshot()``.

    The wrapped object is never imported by name -- any object exposing
    ``snapshot()``, ``step()``, ``reset()`` and ``set_mode()`` works. This is
    what the eventual Driver session connects to the visualizer.
    """

    def __init__(self, session: object, agent: AgentEphemera | None = None, **adapter_kwargs):
        self.session = session
        self.agent = agent
        self._adapter_kwargs = adapter_kwargs
        self._frame: ViewFrame | None = None

    def frame(self) -> ViewFrame:
        if self._frame is None:
            self._frame = snapshot_to_frame(
                self.session.snapshot(), agent=self.agent, **self._adapter_kwargs
            )
        return self._frame

    def step_once(self) -> None:
        step = getattr(self.session, "step", None)
        if callable(step):
            step()
        self._frame = None

    def reset(self) -> None:
        reset = getattr(self.session, "reset", None)
        if callable(reset):
            reset()
        self._frame = None

    def set_mode(self, mode: str) -> None:
        set_mode = getattr(self.session, "set_mode", None)
        if callable(set_mode):
            set_mode(mode)


class StaticFrameSource:
    """Engine-free source over a fixed frame or a scripted generator."""

    def __init__(self, frame: ViewFrame, advance: Callable[[], ViewFrame] | None = None):
        self._frame = frame
        self._advance = advance

    def frame(self) -> ViewFrame:
        return self._frame

    def step_once(self) -> None:
        if self._advance is not None:
            self._frame = self._advance()

    def reset(self) -> None:
        pass

    def set_mode(self, mode: str) -> None:
        pass