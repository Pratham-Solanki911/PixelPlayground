"""A tiny 5x7 bitmap font for baking labels into RGB frames.

Self-contained (no PIL, no system fonts) so overlays and recorded frames look
the same everywhere, including headless machines.
"""
from __future__ import annotations

import numpy as np

_FONT5X7: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "0": (15, 17, 19, 21, 25, 17, 15),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 1, 14),
    "A": (14, 17, 17, 31, 17, 17, 17),
    "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (28, 18, 17, 17, 17, 18, 28),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15),
    "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14),
    "J": (7, 2, 2, 2, 2, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 21, 10),
    "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4),
    "Z": (31, 1, 2, 4, 8, 16, 31),
    "-": (0, 0, 0, 15, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 12, 12),
    ":": (0, 12, 12, 0, 12, 12, 0),
    "%": (25, 26, 2, 4, 8, 11, 19),
    "#": (10, 10, 31, 10, 31, 10, 10),
    "+": (0, 4, 4, 31, 4, 4, 0),
    "/": (1, 2, 2, 4, 8, 8, 16),
    "(": (3, 4, 8, 8, 8, 4, 3),
    ")": (24, 4, 2, 2, 2, 4, 24),
    "_": (0, 0, 0, 0, 0, 0, 31),
}

GLYPH_W = 5
GLYPH_H = 7


def text_width(text: str, scale: int = 1) -> int:
    return (GLYPH_W + 1) * len(text) * scale - scale


def draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color,
    *,
    scale: int = 1,
    upper: bool = True,
) -> int:
    """Draw bitmap text at (x, y). Returns the right edge x coordinate.

    Only ASCII characters present in the font are drawn; unknown characters
    become spaces. ``image`` is a mutable uint8 (H, W, 3) array.
    """
    image = image
    height, width = image.shape[:2]
    scale = max(1, int(scale))
    color = _to_rgb(color)
    cursor = int(x)
    for ch in str(text):
        if upper:
            ch = ch.upper()
        glyph = _FONT5X7.get(ch)
        pad = scale
        if glyph is None:
            cursor += pad
            continue
        for row in range(GLYPH_H):
            bits = glyph[row]
            py = y + row * scale
            if py < 0 or py >= height:
                continue
            for col in range(GLYPH_W):
                if not (bits >> col) & 1:
                    continue
                px = cursor + col * scale
                if px < 0 or px + scale > width:
                    continue
                image[py : py + scale, px : px + scale] = color
        cursor += (GLYPH_W + 1) * scale - 1
    return cursor


def _to_rgb(color) -> tuple[int, int, int]:
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if isinstance(color, int):
        return (color, color, color)
    values = [int(round(float(c))) for c in color[:3]]
    while len(values) < 3:
        values.append(0)
    return tuple(values[:3])