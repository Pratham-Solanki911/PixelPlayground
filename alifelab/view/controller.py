"""Playback controller: pure, GUI-free play/pause/step/reset + speed logic.

The controller owns playback state only (playing, speed, mode). The
visualizer itself decides what to do with the frames it produces.
"""
from __future__ import annotations

from .sources import FrameSource

PLAY_SPEEDS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
_NOMINAL_STEPS_PER_SECOND = 10.0


class PlaybackController:
    """State machine that drives a :class:`FrameSource`."""

    def __init__(
        self,
        source: FrameSource,
        *,
        speed: float = 1.0,
        mode: str = "train",
        auto_retry_episode: bool = True,
    ):
        self.source = source
        self.playing = False
        self.speed = _clamp_speed(speed)
        self.mode = str(mode)
        self.auto_retry_episode = bool(auto_retry_episode)
        self._accumulator = 0.0

    # --------------------------------------------------------------- playback

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_play(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def is_playing(self) -> bool:
        return self.playing

    def set_speed(self, speed: float) -> float:
        self.speed = _clamp_speed(speed)
        return self.speed

    def cycle_speed(self, direction: int = 1) -> float:
        """Cycle to the next preset speed (positive = faster, negative = slower)."""
        speeds = PLAY_SPEEDS
        try:
            index = speeds.index(self.speed)
        except ValueError:
            index = 0
        self.speed = speeds[(index + direction) % len(speeds)]
        return self.speed

    def fast_forward(self) -> None:
        """Jump straight to max speed and keep playing."""
        self.speed = PLAY_SPEEDS[-1]
        self.playing = True

    def is_fast(self) -> bool:
        return self.speed >= PLAY_SPEEDS[-1] and self.playing

    def set_mode(self, mode: str) -> None:
        self.mode = str(mode)
        self.source.set_mode(self.mode)

    # ------------------------------------------------------------------- step

    def step_once(self) -> None:
        self.source.step_once()

    def step_n(self, n: int) -> None:
        for _ in range(max(0, int(n))):
            self.step_once()

    def steps_per_tick(self, interval_seconds: float) -> int:
        """How many source steps to run in one GUI tick at the current speed.

        Uses an accumulator so slow speeds still fire a step once enough time
        has passed, and fast speeds don't starve the accumulator.
        """
        self._accumulator += self.speed * max(0.0, float(interval_seconds)) * _NOMINAL_STEPS_PER_SECOND
        steps = int(self._accumulator)
        if steps >= 1:
            self._accumulator -= float(steps)
        return steps

    ramping = steps_per_tick

    def reset_episode(self) -> None:
        self.source.reset()


def _clamp_speed(speed: float) -> float:
    if speed in PLAY_SPEEDS:
        return float(speed)
    return float(min(PLAY_SPEEDS, key=lambda p: abs(p - float(speed))))