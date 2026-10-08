"""Procedural pixel-art for the interactive visualizer.

Sprites are engine-free 16x16 pixel grids described with ASCII templates.
Each row must contain exactly 16 columns. Characters are *roles* (M = main
body, D = shade, L = light, K = outline, W = white, P = pink/red accent,
A = apple red, etc.) that are resolved to colors at render time, so animal
sprites are tinted by species color while fixed items (apple, brick, water,
grass) keep their own palette. No image assets, fonts, or GPU needed.
"""
from __future__ import annotations

import functools
from typing import Callable

import numpy as np

from . import palette as pal

SIZE = 16

# Animal roles -> coloring via the species base color.
# M main, D shade, L light, K outline, W fixed white, P fixed pink/red accent.
ROLE_M = ord("M")
ROLE_D = ord("D")
ROLE_L = ord("L")
ROLE_K = ord("K")
ROLE_N = ord("N")  # outline alias (K and N are interchangeable)
ROLE_W = ord("W")
ROLE_P = ord("P")

# Fixed-item roles.
ROLE_A = ord("A")  # apple red
ROLE_C = ord("C")  # apple highlight
ROLE_G = ord("G")  # leaf green
ROLE_B = ord("B")  # stem brown
ROLE_Y = ord("Y")  # gold (resource dot / corn)

_EMPTY = ord(".")


def _tpl(rows: list[str]) -> np.ndarray:
    if len(rows) != SIZE:
        raise ValueError(f"sprite template needs {SIZE} rows, got {len(rows)}")
    for i, row in enumerate(rows):
        if len(row) != SIZE:
            raise ValueError(f"sprite row {i} needs {SIZE} cols, got {len(row)!r}")
    return np.asarray([[ord(ch) for ch in row] for row in rows], dtype=np.uint8)


# --------------------------------------------------------------------------
# animal sprites (tinted by species color)

SNAKE_HEAD: np.ndarray = _tpl(
    [
        "................",
        "................",
        ".....NNNNNNN....",
        "....NMMMMMMMN...",
        "NN..NWWMMWWMN...",
        "NN..NKWWMWWKM...",
        "NN..NMWWMMWMMN..",
        "......NMMMMMMN..",
        "......NNMMMMN...",
        "NNN..NNNKMN.....",
        "NNN..NMMMMN....R",
        ".......NMMN..RRR",
        "........NNN..RRR",
        "................",
        "................",
        "................",
    ]
)

SNAKE_BODY: np.ndarray = _tpl(
    [
        "................",
        "KMMMMMMMKMMMMMMK",
        "KMMMMMMMKMMMMMMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMLLLLLLKLLLLMK",
        "KMMMMMMMKMMMMMMK",
        "KMMMMMMMKMMMMMMK",
        "KKKKKKKKKKKKKKKK",
        "................",
        "................",
    ]
)

PREY_HEAD: np.ndarray = _tpl(
    [
        "...WW.....WW....",
        "...WW.PP.WW.....",
        "...WW.PP.WW.....",
        "...WW.....WW....",
        "..MMMMMMMMMMMM..",
        ".MMMMMMMMMMMMMM.",
        "MMMWKMMMMWKWMMM.",
        "MMMWKMMMMWKWMMM.",
        ".MMMMMMMMMMMMMM.",
        ".MMMMMMMMMMMMMM.",
        "..MMMMMMPMMMMM..",
        "....MMMPPPM.....",
        "......MMMM......",
        "................",
        "................",
        "................",
    ]
)

PRED_HEAD: np.ndarray = _tpl(
    [
        ".KK.........KK..",
        ".KKK.......KKK..",
        "..KKK.....KKK...",
        "...KKK...KKK....",
        "....KKKKKKK.....",
        "...KKKKKKKKK....",
        "..KKKKKKKKKKK...",
        ".KKWKMMMMMWKKK..",
        ".KKWKMMMMMWKKK..",
        "..KKKKKKKKKKK...",
        "..KKKKKKKPKKKK..",
        "..KKKWWKKWWKKK..",
        "..KKKWKKKKWKKK..",
        "...KKKKKKKKK....",
        "....KKKKKKK.....",
        "................",
    ]
)

MOUSE_HEAD: np.ndarray = _tpl(
    [
        "..WW........WW..",
        "..WW..PP..PP.WW.",
        "..WW........WW..",
        "..MM...WM...MM..",
        "..MM..MMWKMM..M.",
        "..MMMMMMMMMMMM..",
        ".MMMMMMMMMMMMMM.",
        ".MMMMMMMMMMMMMM.",
        "..MMMMMPPMMMMM..",
        "..MMM.MMMM.MMM..",
        "...MMMMMMMMMM...",
        "....MMMMMMMM....",
        "......MMMM......",
        ".......MM.......",
        "................",
        "................",
    ]
)

BLOB_BODY: np.ndarray = SNAKE_BODY  # generic bead used for every animal body

ANIMAL_HEADS: dict[str, np.ndarray] = {
    "snake": SNAKE_HEAD,
    "prey": PREY_HEAD,
    "mouse": MOUSE_HEAD,
    "predator": PRED_HEAD,
    "wolf": PRED_HEAD,
    "cat": PRED_HEAD,
}

# --------------------------------------------------------------------------
# fixed-item sprites

APPLE: np.ndarray = _tpl(
    [
        "................",
        "................",
        ".....GG..GG.....",
        "....GGGGGGGG....",
        ".....GGGGG......",
        "......BBB.......",
        ".....BBBBBB.....",
        "..DDDDBBBBB.....",
        "..DAAAAABAAA....",
        ".DAAAAACAAAAA...",
        ".DAAAAACAAAAAA..",
        ".DDAAACAAAAAAD..",
        "..DDAAAAAAAAAD..",
        "...DDDDDDDDDD...",
        "................",
        "................",
    ]
)

BUSH: np.ndarray = _tpl(
    [
        "................",
        "......KKKK......",
        ".....KMMMMK.....",
        "....KMMMMMMK....",
        "...KMMMMMMMMK...",
        "..KMMMMMMMMMMK..",
        "..KMMMMMMMMDMMK.",
        ".KMMMMMMMMMDDMK.",
        ".KMMMMDMMMDDMK..",
        ".KMMMDDMMDDDMK..",
        ".KMMMMMMMMMMMK..",
        "..KMMMMMMMMMMK..",
        "...KMMMMMMMMK...",
        "....KKKKKKKK....",
        "................",
        "................",
    ]
)

ROCK: np.ndarray = _tpl(
    [
        "................",
        "....KKKKKKK.....",
        "...KMMMMMMMK....",
        "..KMMLLMMMMMK...",
        ".KMMLLLMMMMMMK..",
        ".KMMLLLLDMMMMMK.",
        ".KMMMMLLDDMMMMMK",
        ".KMMMMMMDDMMMMMK",
        ".KMMMMMDDMMMMMMK",
        ".KMMDDDMMDDMMMK.",
        "..KDDDMDDDDMMK..",
        "...KDDDDDDDDK...",
        "....KKKKKKKK....",
        "................",
        "................",
        "................",
    ]
)

BRICK: np.ndarray = _tpl(
    [
        "KKKKKKKKKKKKKKKK",
        "KLMLLLLLLKLMLLLL",
        "KMMMMDDDDKMMMMDD",
        "KMMMMDDDDKMMMMDD",
        "KMMMMDDDDKMMMMDD",
        "KMMMMDDDDKMMMMDD",
        "KMMMMMMMMKMMMMMM",
        "KMMMMMMMMKMMMMMM",
        "KKKKKKKKKKKKKKKK",
        "KKLMLLLLLLKLMLLL",
        "KKMMMMDDDDKMMMMD",
        "KKMMMMDDDDKMMMMD",
        "KKMMMMDDDDKMMMMD",
        "KKMMMMDDDDKMMMMD",
        "KKMMMMMMMMKMMMMM",
        "KKKKKKKKKKKKKKKK",
    ]
)

GRASS: np.ndarray = _tpl(
    [
        "MDMDMDMDMDMDMDMD",
        "DMDMDMDMDMDMDMDM",
        "MLMDMDLMDMDMDLMD",
        "DMDMDMDMDMDMDMDM",
        "MDLMDMDMDMDMDMDM",
        "DMDMDMDMDMDMDLDM",
        "MLLDMDMDLMDMDMDM",
        "DMDMDMDMDMDMDMDM",
        "MDMDMDLMDMDLMDMD",
        "DMDMDLMDMDMDMDDM",
        "MDMDMDMDMDMDMDMD",
        "DMDLMDMDLMDMDMDM",
        "MLMDMDMDMDMDLDM.",
        "DMDMDMDMDMDMDMDM",
        "MDMDMDLMDMDMDMDM",
        "DMDDMDMDMDMDMDMM",
    ]
)

WATER: np.ndarray = _tpl(
    [
        "LLLLLLDDDLLLLDDD",
        "LDDDDDDDLLDDDDDD",
        "LDDLLDDDDDDDDLLD",
        "LDDLDDDDDDDDDDDD",
        "LDLDDDLLDDDLLLDD",
        "DDDDDDDDDDDDDLLD",
        "DDLLDDDDDDDDDDDD",
        "DDDDDLLDDDDLLDDD",
        "DDDDDDDDDLLLLLLD",
        "DDDLLLLDDDDDDDDD",
        "DDDDDDDDDDDDLLLD",
        "DLLDDDDDDLDDDDDD",
        "DDDDDLLDDDDDDDDD",
        "DDDDDDDDDDDDDDDD",
        "DDDDDDDDLLDDDDDD",
        "DDDDDDDDDDDDDDDD",
    ]
)

CORB: np.ndarray = _tpl(
    [
        "................",
        "................",
        "......YYYY......",
        "....YYYYYYYY....",
        "...YYYYYYYYYY...",
        "...YDDYYYYDDY...",
        "...YDDYYYYDDY...",
        "...YYYYYYYYYY...",
        "....YYYYYYYY....",
        "......YYYY......",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
    ]
)

# --------------------------------------------------------------------------
# color resolution

_WHITE: tuple[int, int, int] = (250, 250, 250)
_PINK: tuple[int, int, int] = (250, 130, 150)
_APPLE_RED: tuple[int, int, int] = (224, 46, 40)
_APPLE_LIGHT: tuple[int, int, int] = (252, 150, 130)
_APPLE_DARK: tuple[int, int, int] = (160, 26, 22)
_STEM: tuple[int, int, int] = (120, 74, 42)
_LEAF: tuple[int, int, int] = (92, 178, 60)
_MORTAR: tuple[int, int, int] = (56, 56, 62)
_BRICK_MAIN: tuple[int, int, int] = (118, 118, 128)
_BRICK_LIGHT: tuple[int, int, int] = (152, 152, 162)
_BRICK_DARK: tuple[int, int, int] = (78, 78, 86)
_WATER_MAIN: tuple[int, int, int] = (46, 92, 168)
_WATER_LIGHT: tuple[int, int, int] = (116, 176, 232)
_WATER_DARK: tuple[int, int, int] = (28, 58, 120)
_GOLD: tuple[int, int, int] = (206, 168, 44)
_GOLD_LIGHT: tuple[int, int, int] = (255, 222, 120)

FLOOR_BASE: tuple[int, int, int] = (28, 36, 28)
FLOOR_SPECK: tuple[int, int, int] = (44, 54, 42)
FLOOR_DARK: tuple[int, int, int] = (21, 27, 21)


def _to_rgba(rgb: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Attach an alpha channel (255 where a role pixel exists, else 0)."""
    alpha = np.where(grid != _EMPTY, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def _scale_rgb(sprite: np.ndarray, factor: float) -> np.ndarray:
    out = sprite.astype(np.float32)
    out[..., :3] *= factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _tint_grid(grid: np.ndarray, base: tuple[int, int, int]) -> np.ndarray:
    """Resolve a role grid to an RGBA sprite using a species base color."""
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    mapping = {
        ROLE_M: (base, True),
        ROLE_D: (pal.dim(base, 0.55), True),
        ROLE_L: (pal.blend(base, _WHITE, 0.45), True),
        ROLE_K: (pal.blend(base, (12, 12, 14), 0.78), True),
        ROLE_N: (pal.blend(base, (12, 12, 14), 0.78), True),
        ROLE_W: (_WHITE, False),
        ROLE_P: (_PINK, False),
    }
    for code, (color, _) in mapping.items():
        mask = grid == code
        if mask.any():
            out[mask] = color
    return _to_rgba(out, grid)


def _fixed_grid(grid: np.ndarray, mapping: dict[int, tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for code, color in mapping.items():
        mask = grid == code
        if mask.any():
            out[mask] = color
    return _to_rgba(out, grid)


@functools.lru_cache(maxsize=512)
def _tinted(kind: str, base: tuple[int, int, int], dead: bool) -> np.ndarray:
    if kind == "snake":
        head = SNAKE_HEAD
    elif kind in ANIMAL_HEADS:
        head = ANIMAL_HEADS[kind]
    else:
        head = BLOB_BODY
    sprite = _tint_grid(head, base)
    if dead:
        sprite = _scale_rgb(sprite, 0.22)
    return sprite


@functools.lru_cache(maxsize=256)
def _body(kind: str, base: tuple[int, int, int], dead: bool) -> np.ndarray:
    sprite = _tint_grid(SNAKE_BODY, base)
    if dead:
        sprite = _scale_rgb(sprite, 0.22)
    return sprite


@functools.lru_cache(maxsize=64)
def _item(kind: str) -> np.ndarray:
    if kind == "food":
        return _fixed_grid(
            APPLE,
            {ROLE_A: _APPLE_RED, ROLE_C: _APPLE_LIGHT, ROLE_D: _APPLE_DARK, ROLE_B: _STEM, ROLE_G: _LEAF, ROLE_K: pal.dim(_STEM, 0.6)},
        )
    if kind in ("wall", "terrain"):
        return _fixed_grid(BRICK, {ROLE_M: _BRICK_MAIN, ROLE_L: _BRICK_LIGHT, ROLE_D: _BRICK_DARK, ROLE_K: _MORTAR})
    if kind == "water":
        return _fixed_grid(WATER, {ROLE_M: _WATER_MAIN, ROLE_L: _WATER_LIGHT, ROLE_D: _WATER_DARK, ROLE_K: _WATER_DARK})
    if kind == "grass":
        return _fixed_grid(GRASS, {ROLE_M: FLOOR_BASE, ROLE_L: FLOOR_SPECK, ROLE_D: FLOOR_DARK, ROLE_K: FLOOR_DARK})
    if kind == "resource":
        return _fixed_grid(CORB, {ROLE_Y: _GOLD, ROLE_D: _GOLD_LIGHT, ROLE_M: _GOLD, ROLE_L: _GOLD_LIGHT, ROLE_K: pal.dim(_GOLD, 0.5)})
    # plant / bush / rock use species tint
    return _tint_grid(BUSH if kind in ("plant", "tree", "flower", "bush") else ROCK, pal.species_color(kind))


def _rotate(sprite: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """Rotate a RIGHT-facing head sprite for the entity's direction."""
    dx, dy = direction
    if (dx, dy) == (-1, 0):  # LEFT
        return np.rot90(sprite, 2)
    if (dx, dy) == (0, -1):  # UP
        return np.rot90(sprite, -1)
    if (dx, dy) == (0, 1):  # DOWN
        return np.rot90(sprite, 1)
    return sprite


# --------------------------------------------------------------------------
# painting

def _resize(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    iy = np.minimum((np.arange(th) * (arr.shape[0] - 1)) // max(1, th - 1), arr.shape[0] - 1)
    ix = np.minimum((np.arange(tw) * (arr.shape[1] - 1)) // max(1, tw - 1), arr.shape[1] - 1)
    return arr[np.ix_(iy, ix)]


def stamp_sprite(image: np.ndarray, cx: int, cy: int, scale: int, sprite: np.ndarray, *, rotate: tuple[int, int] | None = None) -> None:
    """Nearest-upscale ``sprite`` into cell (cx, cy), compositing over the cell.

    Sprites are RGBA: transparent role pixels let the terrain underneath show
    through instead of punching black holes in the map.
    """
    if sprite.ndim == 2 or sprite.shape[-1] == 3:
        alpha = np.full(sprite.shape[:2], 255, dtype=np.uint8)
        sprite = np.dstack([sprite[..., :3], alpha])
    cell = _resize(sprite, scale, scale)
    if rotate is not None:
        cell = _resize(_rotate(sprite, rotate), scale, scale)
    y0, y1 = cy * scale, (cy + 1) * scale
    x0, x1 = cx * scale, (cx + 1) * scale
    region = image[y0:y1, x0:x1]
    a = (cell[..., 3:] / 255.0).astype(np.float32)
    blended = cell[..., :3].astype(np.float32) * a + region.astype(np.float32) * (1.0 - a)
    region[...] = np.clip(blended, 0, 255).astype(np.uint8)