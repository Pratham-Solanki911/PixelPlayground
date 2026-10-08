"""Engine-free HUD panels for the interactive visualizer.

Panels draw text into NumPy arrays using the built-in bitmap font, so the
visualizer composes header + world + status strip into one presentation frame
with zero GUI-library dependency -- identical headless and live.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import palette as pal
from .events import Genealogy
from .font import draw_text, text_width
from .inspector import InspectorReport
from .population import PopulationTracker

_BG = (12, 14, 18)
_FG = (205, 210, 215)
_FG_DIM = (150, 156, 164)
_TITLE = (225, 228, 232)

_HEADER_BARS = {
    "dashboard": (194, 156, 14),
    "events": (120, 130, 150),
    "inspector": (64, 180, 206),
    "lineage": (150, 120, 200),
}


class Panel:
    """Base class: a titled strip of text drawn into an RGB canvas."""

    name = "panel"
    title = ""

    def __init__(self, font_scale: int = 2):
        self.font_scale = max(1, int(font_scale))
        self.header_bar = _HEADER_BARS.get(self.name, _TITLE)

    def lines(self) -> list[str]:
        return []

    def render(self, width: int, height: int, font_scale: int | None = None) -> np.ndarray:
        fs = self.font_scale if font_scale is None else max(1, int(font_scale))
        image = np.full((max(1, int(height)), max(1, int(width)), 3), _BG, dtype=np.uint8)
        cell_h = 7 * fs + fs
        y = 0
        if self.title:
            image[0:cell_h, :, :] = pal.dim(self.header_bar, 0.35)
            draw_text(image, fs // 2, fs // 2, self.title.upper(), _TITLE, scale=fs)
            y += cell_h
            image[y - fs // 2 : y, :, :] = pal.dim(self.header_bar, 0.15)
            y += fs // 2
        for line in self.lines():
            if y + cell_h > image.shape[0]:
                break
            draw_text(image, fs // 2, y + fs // 2, _clip(line, image.shape[1], fs), _FG, scale=fs)
            y += cell_h
        return image


def _clip(line: str, width: int, fs: int) -> str:
    if text_width(line, fs) <= width:
        return line
    lo, hi = 1, len(line)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(line[:mid] + "..", fs) <= width:
            lo = mid
        else:
            hi = mid - 1
    return line[:lo] + ".."


class TextPanel(Panel):
    """Displays whatever text lines it is given."""

    name = "events"

    def __init__(self, title: str = "Events", font_scale: int = 2):
        super().__init__(font_scale=font_scale)
        self.title = title
        self._lines: list[str] = []

    def set_lines(self, lines: Sequence[str]) -> "TextPanel":
        self._lines = list(lines)
        return self

    def lines(self) -> list[str]:
        return self._lines


EventPanel = TextPanel


class DashboardPanel(Panel):
    """Live population + cumulative counters from a PopulationTracker."""

    name = "dashboard"

    def __init__(self, tracker: PopulationTracker, title: str = "Population", font_scale: int = 2):
        super().__init__(font_scale=font_scale)
        self.title = title
        self.tracker = tracker

    def lines(self) -> list[str]:
        summary = self.tracker.summary()
        out = [f"Total   {summary['total']}"]
        species = summary["by_species"]
        if species:
            parts = "  ".join(f"{k.title()}:{v}" for k, v in species.items())
            out.append(_chunks(parts))
        counts = {
            "Births": "births",
            "Deaths": "deaths",
            "Predation": "predation",
            "Repro": "reproduction",
            "Food": "food_consumed",
            "Spawns": "spawns",
            "Mut": "mutations",
            "Attacks": "attack_attempts",
        }
        for label, key in counts.items():
            out.append(f"{label:<10} {summary[key]}")
        if self.tracker.last_deltas:
            deltas = self.tracker.last_deltas
            deltas = " ".join(f"{k[:3].upper()}+{v}" for k, v in deltas.items() if v)
            if deltas:
                out.append("delta " + deltas)
        return out

    def summary(self) -> dict[str, object]:
        return self.tracker.summary()


def _chunks(text: str, width: int = 48) -> str:
    if len(text) <= width:
        return text
    return text[: width - 3] + ".."


class InspectorPanel(Panel):
    """Two-column label: value listing from an InspectorReport."""

    name = "inspector"

    def __init__(self, font_scale: int = 2):
        super().__init__(font_scale=font_scale)
        self.title = "Inspector"
        self.report: InspectorReport | None = None

    def set_report(self, report: InspectorReport | None) -> "InspectorPanel":
        self.report = report
        return self

    def render(self, width: int, height: int, font_scale: int | None = None) -> np.ndarray:
        self._label_width = max(1, int(width * 0.38))
        return super().render(width, height, font_scale=font_scale)

    def lines(self) -> list[str]:
        if self.report is None:
            return ["click an entity to inspect"]
        status = "alive" if self.report.alive else "dead"
        out = [f"{self.report.name}  ({status})"]
        lw = max(self._label_width, 14)
        for field in self.report.fields:
            out.append(f"{field.label:<{lw}} {field.value}")
        return out


class LineagePanel(Panel):
    """Family tree for the selected entity, from cumulative genealogy."""

    name = "lineage"

    def __init__(self, genealogy: Genealogy, font_scale: int = 2):
        super().__init__(font_scale=font_scale)
        self.title = "Lineage"
        self.genealogy = genealogy
        self.entity_id: int | None = None
        self.entity_name: str = ""

    def set_target(self, entity_id: int | None, entity_name: str = "") -> "LineagePanel":
        self.entity_id = entity_id
        self.entity_name = entity_name
        return self

    def lines(self) -> list[str]:
        if self.entity_id is None:
            return ["no selection"]
        parents = self.genealogy.parents.get(self.entity_id, ())
        children = sorted(c for c, ps in self.genealogy.parents.items() if self.entity_id in ps)
        siblings = sorted(
            c
            for pid in parents
            for c in self.genealogy.parents
            if self.entity_id in self.genealogy.parents[c] and c != self.entity_id
        )
        name = self.entity_name or f"Entity {self.entity_id}"
        out = [f"* {name}  (id {self.entity_id})"]
        out.append(f"  parents:   {_ids(parents)}")
        out.append(f"  siblings:  {_ids(siblings)}")
        out.append(f"  children:  {_ids(children)}")
        out.append(f"  offspring: {self.genealogy.offspring.get(self.entity_id, 0)}")
        return out


def _ids(ids: Sequence[int]) -> str:
    return ", ".join(f"#{i}" for i in ids) if ids else "-"


def stack_vertical(panels: Sequence[Panel], width: int, gap: int = 2) -> np.ndarray:
    """Stack panels vertically into one canvas at fixed width."""
    renders = [panel.render(width, _guess_height(panel)) for panel in panels]
    total = sum(img.shape[0] for img in renders) + gap * (len(renders) - 1)
    canvas = np.full((max(1, total), max(1, width), 3), _BG, dtype=np.uint8)
    y = 0
    for img in renders:
        canvas[y : y + img.shape[0], :, :] = img
        y += img.shape[0] + gap
    return canvas


def stack_horizontal(panels: Sequence[Panel], total_width: int, height: int, gap: int = 2) -> np.ndarray:
    """Tile panels side by side at a fixed height into one canvas."""
    count = max(1, len(panels))
    width = max(1, (int(total_width) - gap * (count - 1)) // count)
    renders = [panel.render(width, height) for panel in panels]
    canvas = np.full((max(1, int(height)), max(1, int(total_width)), 3), _BG, dtype=np.uint8)
    x = 0
    for img in renders:
        canvas[:, x : x + img.shape[1], :] = img
        x += img.shape[1] + gap
    return canvas


def _guess_height(panel: Panel) -> int:
    fs = panel.font_scale
    cell_h = 7 * fs + fs
    rows = len(panel.lines())
    return (rows + 2) * cell_h if _title_used(panel) else (rows + 1) * cell_h


def _title_used(panel: Panel) -> bool:
    return bool(panel.title)