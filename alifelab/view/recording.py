"""Frame recording that re-uses the existing renderers.

The visualizer itself never renders RGB/ASCII frames for recording -- it
delegates to *providers* supplied by the driver (a live session hands over
the existing ``RGBRenderer`` and ``AsciiRenderer``). This guarantees the
recording never duplicates rendering logic and works identically headless.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

RGBProvider = Callable[[], np.ndarray]
ASCIIProvider = Callable[[], str]


class Recorder:
    """Two independent channels (RGB frames, ASCII transcript)."""

    def __init__(self) -> None:
        self.rgb_provider: RGBProvider | None = None
        self.ascii_provider: ASCIIProvider | None = None
        self._rgb: List[np.ndarray] = []
        self._ascii: List[str] = []
        self._rgb_enabled = False
        self._ascii_enabled = False

    # ------------------------------------------------------------------- setup

    def attach(self, *, rgb: RGBProvider | None = None, ascii: ASCIIProvider | None = None) -> None:
        if rgb is not None:
            self.rgb_provider = rgb
        if ascii is not None:
            self.ascii_provider = ascii

    def set_rgb_recording(self, enabled: bool) -> None:
        self._rgb_enabled = bool(enabled)
        if not self._rgb_enabled:
            self.flush_rgb()

    def set_ascii_recording(self, enabled: bool) -> None:
        self._ascii_enabled = bool(enabled)
        if not self._ascii_enabled:
            self.flush_ascii()

    @property
    def rgb_recording(self) -> bool:
        return self._rgb_enabled

    @property
    def ascii_recording(self) -> bool:
        return self._ascii_enabled

    # ------------------------------------------------------------------ capture

    def capture(self) -> None:
        """Capture one frame from each enabled provider that is attached."""
        if self._rgb_enabled and self.rgb_provider is not None:
            self._rgb.append(np.asarray(self.rgb_provider()).copy())
        if self._ascii_enabled and self.ascii_provider is not None:
            self._ascii.append(str(self.ascii_provider()))

    def frame_count(self) -> int:
        return len(self._rgb)

    def line_count(self) -> int:
        return len(self._ascii)

    # ------------------------------------------------------------------- saving

    def save_rgb(self, path: str) -> None:
        """Save captured RGB frames to ``npz`` (key ``'frames'``: (N, H, W, 3))."""
        if not self._rgb:
            raise ValueError("no RGB frames captured; start recording first")
        np.savez(path, frames=np.stack(self._rgb))

    def save_ascii(self, path: str) -> None:
        if not self._ascii:
            raise ValueError("no ASCII lines captured; start recording first")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(self._ascii))
            handle.write("\n")

    def clear(self) -> None:
        self._rgb.clear()
        self._ascii.clear()

    def flush_rgb(self) -> None:
        self._rgb_enabled = False
        self._rgb.clear()

    def flush_ascii(self) -> None:
        self._ascii_enabled = False
        self._ascii.clear()