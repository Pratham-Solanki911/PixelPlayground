"""Data-driven color palettes for the interactive visualizer.

Colors are keyed by *species name* (falling back to entity kind), so new
species render automatically. Nothing here knows about any specific game.
"""
from __future__ import annotations

import colorsys
from typing import Tuple

SpeciesColor = Tuple[int, int, int]

SPECIES_COLORS: dict[str, SpeciesColor] = {
    "snake": (194, 156, 14),
    "predator": (212, 44, 44),
    "prey": (64, 180, 206),
    "food": (66, 192, 80),
    "plant": (150, 230, 90),
    "rock": (140, 130, 120),
    "stone": (140, 130, 120),
    "wall": (104, 104, 112),
}

TERRAIN_COLORS: dict[int, SpeciesColor] = {
    0: (25, 28, 32),  # empty floor
    1: (88, 88, 96),  # wall / obstacle
    2: (46, 92, 168),  # water
}

RESOURCE_COLOR: SpeciesColor = (206, 168, 44)
RESOURCE_CORE: SpeciesColor = (255, 222, 120)

AGENT_HEAD: SpeciesColor = (245, 245, 245)
SELECTION: SpeciesColor = (255, 255, 255)
HEADER_BG: SpeciesColor = (16, 18, 22)


def _hash_hue(name: str) -> float:
    seed = 2166136261
    for ch in name:
        seed = ((seed ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return (seed & 0xFF) / 255.0


def species_color(species: str | None, kind: str | None = None) -> SpeciesColor:
    key = (species or kind or "entity").lower()
    if key in SPECIES_COLORS:
        return SPECIES_COLORS[key]
    hue = _hash_hue(key)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return _rgb255(r), _rgb255(g), _rgb255(b)


def _rgb255(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 255))


def dim(color: SpeciesColor, factor: float = 0.55) -> SpeciesColor:
    return tuple(int(round(c * factor)) for c in color)


def brighten(color: SpeciesColor, factor: float = 1.25) -> SpeciesColor:
    return tuple(min(255, int(round(c * factor))) for c in color)


def blend(c1: SpeciesColor, c2: SpeciesColor, t: float) -> SpeciesColor:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))