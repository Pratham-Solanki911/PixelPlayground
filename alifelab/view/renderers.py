"""Generic rendering pipeline for the interactive visualizer.

The API is deliberately engine-free: renderers only consume
:class:`~alifelab.view.model.ViewFrame` / ``ViewEntity`` objects and produce
NumPy RGB arrays (or structured overlays). No simulation, RL, trainer, or
agent objects are touched here.

The interactive RGB renderer here is a *display* renderer (zoom, selection,
direction markers, sizing). Frame recording that must re-use the existing
``AsciiRenderer`` / ``RGBRenderer`` is handled by :mod:`alifelab.view.recording`,
which consumes arbitrary providers -- so RGB logic is never duplicated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from . import palette as pal
from . import sprites
from .font import draw_text, text_width
from .model import TERRAIN_EMPTY, ViewEntity, ViewFrame

SPRITE_MIN_SCALE = 3


class Renderer(ABC):
    """Base class for everything that turns a frame into pixels/labels."""

    @abstractmethod
    def render(self, frame: ViewFrame, **kwargs):  # pragma: no cover - ABC
        ...


class EntityRenderer(Renderer):
    """Draws a single generic entity (head, body, selection ring, size)."""

    def __init__(self, *, ignore: str | None = None):
        self.ignore = ignore

    def render(self, frame: ViewFrame, **kwargs):
        raise NotImplementedError("EntityRenderer.render_onto() draws onto an image")

    def cell_fills(self, entity: ViewEntity) -> dict[tuple[int, int], tuple[int, int, int]]:
        base = pal.species_color(entity.species, entity.kind)
        fills: dict[tuple[int, int], tuple[int, int, int]] = {}
        for cell in entity.body:
            fills[cell] = pal.dim(base, 0.45 if entity.alive else 0.18)
        if entity.is_agent:
            fills[entity.head] = pal.blend(pal.AGENT_HEAD, base, 0.35)
        else:
            fills[entity.head] = pal.brighten(base, 1.15) if entity.alive else pal.dim(base, 0.25)
        return fills

    def draw_onto(
        self,
        image: np.ndarray,
        entity: ViewEntity,
        *,
        scale: int = 1,
        selected: bool = False,
        show_direction: bool = True,
    ) -> None:
        h, w = image.shape[:2]
        scale = max(1, int(scale))
        if self.ignore and entity.species == self.ignore:
            return
        if scale >= SPRITE_MIN_SCALE:
            self._draw_sprite(image, h, w, entity, scale=scale, selected=selected)
            return
        fills = self.cell_fills(entity)
        for (cx, cy), color in fills.items():
            if 0 <= cx < w // scale and 0 <= cy < h // scale:
                _stamp(image, cx, cy, scale, color)
        if selected:
            hx, hy = entity.head
            if 0 <= hx < w // scale and 0 <= hy < h // scale:
                _ring(image, hx, hy, scale, pal.SELECTION)
        if show_direction and entity.direction and entity.direction != (0, 0):
            hx, hy = entity.head
            tx, ty = hx + entity.direction[0], hy + entity.direction[1]
            if entity.state == "static":
                return
            if 0 <= tx < w // scale and 0 <= ty < h // scale:
                _stamp(image, tx, ty, scale, pal.blend(pal.species_color(entity.species, entity.kind), (255, 255, 255), 0.55))

    def _draw_sprite(self, image: np.ndarray, h: int, w: int, entity: ViewEntity, *, scale: int, selected: bool) -> None:
        kind = (entity.kind or entity.species or "").lower()
        base = pal.species_color(entity.species, entity.kind) if kind not in ("food", "wall", "water", "grass", "resource") else (0, 0, 0)
        cols = w // scale
        rows = h // scale

        def in_bounds(cx: int, cy: int) -> bool:
            return 0 <= cx < cols and 0 <= cy < rows

        if entity.alive:
            for (cx, cy) in entity.body:
                if in_bounds(cx, cy):
                    sprite = sprites._body("snake", base, False)
                    sprites.stamp_sprite(image, cx, cy, scale, sprite)
            hx, hy = entity.head
            if in_bounds(hx, hy):
                if kind in ("plant", "bush", "tree", "flower", "rock", "stone"):
                    sprite = sprites._item(kind)
                elif kind == "food":
                    sprite = sprites._item("food")
                elif kind in ("wall", "water", "grass", "resource"):
                    sprite = sprites._item(kind)
                else:
                    sprite = sprites._tinted(kind, base, False)
                    if entity.is_agent:
                        sprite = _agent_tint(sprite)
                    dr = entity.direction if entity.direction else (0, 0)
                    if dr != (0, 0) and entity.state != "static":
                        sprites.stamp_sprite(image, hx, hy, scale, sprite, rotate=dr)
                        if scale >= 8 and entity.is_agent:
                            _badge(image, hx, hy, scale)
                        if selected:
                            _ring(image, hx, hy, scale, pal.SELECTION)
                        return
                sprites.stamp_sprite(image, hx, hy, scale, sprite)
        else:
            body_sprite = sprites._body("snake", base, True)
            for (cx, cy) in entity.body:
                if in_bounds(cx, cy):
                    sprites.stamp_sprite(image, cx, cy, scale, body_sprite)
            hx, hy = entity.head
            if in_bounds(hx, hy):
                if kind in ("plant", "bush", "tree", "flower", "rock", "stone", "food", "wall", "water", "grass", "resource"):
                    sprite = sprites._item(kind)
                else:
                    sprite = sprites._tinted(kind, base, True)
                sprite = _pepper(sprite)
                sprites.stamp_sprite(image, hx, hy, scale, sprite)
        if selected:
            hx, hy = entity.head
            if 0 <= hx < cols and 0 <= hy < rows:
                _ring(image, hx, hy, scale, pal.SELECTION)


class WorldRenderer(Renderer):
    """Renders the whole grid: terrain, resources, then all entities."""

    def __init__(self, entity_renderer: EntityRenderer | None = None):
        self.entity_renderer = entity_renderer or EntityRenderer()

    def render(self, frame: ViewFrame, *, scale: int = 1, selected_id: int | None = None) -> np.ndarray:
        scale = max(1, int(scale))
        image = np.empty((frame.world.height * scale, frame.world.width * scale, 3), dtype=np.uint8)
        _paint_terrain(image, frame.world.terrain, frame.world.resources, scale)
        selected = frame.world.entity_by_id(selected_id) if selected_id is not None else None
        for entity in frame.world.entities:
            if entity.id == getattr(selected, "id", None):
                continue
            self.entity_renderer.draw_onto(image, entity, scale=scale, selected=False)
        if selected is not None:
            self.entity_renderer.draw_onto(image, selected, scale=scale, selected=True)
        return image


# ---------------------------------------------------------------------------


def _paint_terrain(image: np.ndarray, terrain: np.ndarray, resources: np.ndarray, scale: int) -> None:
    height, width = terrain.shape
    scale = max(1, int(scale))
    if scale >= SPRITE_MIN_SCALE:
        _paint_terrain_sprites(image, terrain, resources, scale)
        return
    maximum = float(resources.max()) if resources.size else 0.0
    for y in range(height):
        for x in range(width):
            base = pal.TERRAIN_COLORS.get(int(terrain[y, x]), pal.TERRAIN_COLORS[TERRAIN_EMPTY])
            amount = float(resources[y, x]) if (resources.size and maximum > 0) else 0.0
            if amount > 0:
                t = float(np.clip(amount / maximum, 0.0, 1.0))
                color = pal.blend(pal.RESOURCE_COLOR, pal.RESOURCE_CORE, t)
                color = pal.blend(base, color, 0.75)
            else:
                color = base
            _stamp(image, x, y, scale, color)


def _paint_terrain_sprites(image: np.ndarray, terrain: np.ndarray, resources: np.ndarray, scale: int) -> None:
    height, width = terrain.shape
    maximum = float(resources.max()) if resources.size else 0.0
    grass = sprites._item("grass")
    brick = sprites._item("wall")
    water = sprites._item("water")
    resource = sprites._item("resource")
    for y in range(height):
        for x in range(width):
            tile = grass if int(terrain[y, x]) == TERRAIN_EMPTY else brick if int(terrain[y, x]) == 1 else water
            amount = float(resources[y, x]) if (resources.size and maximum > 0) else 0.0
            if amount > 0:
                t = float(np.clip(amount / maximum, 0.0, 1.0))
                tint = t * 0.6
                if int(terrain[y, x]) == TERRAIN_EMPTY:
                    colored = _mix_sprites(grass, resource, tint)
                    sprites.stamp_sprite(image, x, y, scale, colored)
                    if scale >= 8:
                        sprites.stamp_sprite(image, x, y, scale, resource)
                else:
                    sprites.stamp_sprite(image, x, y, scale, tile)
            else:
                sprites.stamp_sprite(image, x, y, scale, tile)


def _mix_sprites(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t).astype(np.uint8)


def _agent_tint(sprite: np.ndarray) -> np.ndarray:
    rgb = sprite[..., :3]
    alpha = sprite[..., 3:]
    blended = _mix_sprites(rgb, np.full_like(rgb, 255, dtype=np.uint8), 0.22)
    return np.dstack([blended, alpha])


def _pepper(sprite: np.ndarray) -> np.ndarray:
    """Dead entities get a faint dark cross so they read as remains."""
    out = sprite.copy().astype(np.float32)
    out *= 0.8
    for i in range(0, sprites.SIZE, 2):
        out[i, i] = out[i, sprites.SIZE - 1 - i] = 30
        out[i + 1, i] = out[i + 1, sprites.SIZE - 1 - i] = 30
    return out.astype(np.uint8)


def _badge(image: np.ndarray, cx: int, cy: int, scale: int) -> None:
    """Small white corner chip marking the RL agent's head cell."""
    size = max(2, scale // 6)
    y0, x0 = cy * scale + 1, cx * scale + 1
    image[y0 : y0 + size, x0 : x0 + size] = pal.SELECTION


def _stamp(image: np.ndarray, cx: int, cy: int, scale: int, color) -> None:
    image[cy * scale : (cy + 1) * scale, cx * scale : (cx + 1) * scale] = color


def _ring(image: np.ndarray, cx: int, cy: int, scale: int, color) -> None:
    """Outline a single cell (only visible when scale >= 2)."""
    if scale < 2:
        _stamp(image, cx, cy, scale, color)
        return
    y0, y1 = cy * scale, (cy + 1) * scale
    x0, x1 = cx * scale, (cx + 1) * scale
    thickness = max(1, scale // 5)
    image[y0 : y0 + thickness, x0:x1] = color
    image[y1 - thickness : y1, x0:x1] = color
    image[y0:y1, x0 : x0 + thickness] = color
    image[y0:y1, x1 - thickness : x1] = color


class OverlayRenderer(Renderer):
    """Adds a readable label strip above the world image.

    Uses the built-in 5x7 bitmap font so recorded RGB frames carry readable
    headers even on machines without GUI fonts.
    """

    def __init__(self, font_scale: int = 2):
        self.font_scale = max(1, int(font_scale))

    def render(
        self,
        frame: ViewFrame,
        world_image: np.ndarray,
        *,
        extra: dict[str, float | str | None] | None = None,
        max_lines: int = 4,
    ) -> np.ndarray:
        lines = self.lines(frame, extra=extra)[:max_lines]
        fs = self.font_scale
        width = world_image.shape[1]
        header_h = max(1, len(lines)) * (7 * fs + fs)
        header = np.full((header_h, width, 3), pal.HEADER_BG, dtype=np.uint8)

        def clamp(line: str) -> str:
            cell = (5 + 1) * fs
            max_chars = max(1, width // cell)
            if text_width(line, fs) <= width:
                return line
            return line[: max(1, max_chars - 2)] + ".."

        palette_of_line = (pal.brighten(pal.AGENT_HEAD, 0.85), (240, 190, 90), pal.species_color("prey"))
        for i, line in enumerate(lines):
            color = palette_of_line[i % len(palette_of_line)]
            draw_text(header, fs, i * (7 * fs + fs) + fs // 2, clamp(line), color, scale=fs)
        return np.vstack([header, world_image])

    def lines(self, frame: ViewFrame, extra: dict[str, float | str | None] | None = None) -> list[str]:
        world = frame.world
        ret = "-" if frame.episode_return is None else f"{frame.episode_return:.2f}"
        lines = [f"{frame.mode.upper():<5} EP {frame.episode:03d} STEP {frame.step:04d} RETURN {ret} LEN {frame.episode_length or 0}"]
        agent = frame.agent
        if agent and agent.active:
            loss = "-" if agent.loss is None else f"{agent.loss:.4f}"
            eps = "-" if agent.epsilon is None else f"{agent.epsilon:.3f}"
            value = "-" if agent.value is None else f"{agent.value:.3f}"
            rec = "R-ON" if agent.recurrent_active else ("R-OFF" if agent.recurrent else "-")
            lines.append(f"AGENT {agent.agent_id or '-'} ACTION {agent.current_action_label} REW {agent.reward if agent.reward is not None else '-'} LOSS {loss} EPS {eps} VALUE {value} {rec}")
        pop = [f"{k.upper()} {v}" for k, v in sorted(frame.population.items())]
        lines.append(f"POP {world.total_population} " + " ".join(pop))
        if extra:
            tail = " ".join(f"{k.upper()} {v}" for k, v in extra.items() if v is not None)
            if tail:
                lines.append(tail)
        return lines


class ObservationRenderer(Renderer):
    """Renders the agent's observation array as a small RGB thumbnail.

    Handles 2D single-channel, 3D multi-channel (first 3 channels), and
    flat vectors (drawn as a horizontal energy bar). Pure normalization --
    no knowledge of what the channels mean.
    """

    def render(self, frame: ViewFrame, observation=None, *, scale: int = 4) -> np.ndarray:
        obs = frame.agent.observation if observation is None else observation
        return self.render_array(obs, scale=scale)

    def render_array(self, obs, *, scale: int = 4) -> np.ndarray:
        if obs is None:
            return np.zeros((2, 2, 3), dtype=np.uint8)
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            channels = arr[..., :3]
            out = self._normalize(channels)
        elif arr.ndim == 2:
            span = arr.max() - arr.min() if arr.size else 1.0
            gray = ((arr - arr.min()) / span * 255.0).astype(np.uint8) if span else arr.astype(np.uint8)
            out = np.repeat(gray[..., None], 3, axis=-1)
        elif arr.ndim == 1:
            length = arr.size
            row = np.full((1, max(1, length), 3), 20, dtype=np.uint8)
            if length:
                span = arr.max() - arr.min() if arr.max() != arr.min() else 1.0
                levels = ((arr - arr.min()) / span * 255.0).astype(np.uint8)
                row[0, :, 0] = levels
                row[0, :, 1] = np.clip(255 - levels, 0, 255) // 2
            out = row
        else:
            flat = arr.reshape(-1)
            out = self.render_array(flat, scale=1)
        if scale > 1:
            out = np.repeat(out, int(scale), axis=0)
            out = np.repeat(out, int(scale), axis=1)
        return out

    @staticmethod
    def _normalize(channels) -> np.ndarray:
        arr = channels.astype(np.float32)
        out = np.zeros_like(arr)
        for c in range(arr.shape[-1]):
            channel = arr[..., c]
            if channel.size and float(np.max(channel)) - float(np.min(channel)) > 1e-9:
                low, high = np.min(channel), np.max(channel)
                out[..., c] = (channel - low) / (high - low) * 255.0
        return out.astype(np.uint8)