"""Monospace ASCII renderer of a WorldSnapshot with optional ANSI colors."""
from __future__ import annotations

from ..obs.common import entity_cells


class AsciiRenderer:
    def __init__(self, color: bool = True, agent_marker: str = "@", body_marker: str = "o"):
        self.color = bool(color)
        self.agent_marker = agent_marker
        self.body_marker = body_marker
        self._symbols = {
            "snake": "S",
            "predator": "D",
            "prey": "p",
            "food": "f",
            "plant": "P",
            "rock": "o",
        }
        self._colors = {
            "snake": "33",
            "predator": "31",
            "prey": "36",
            "food": "32",
            "plant": "92",
            "rock": "90",
        }
        self._terrain_symbols = {0: ".", 1: "#", 2: "~"}

    def render(self, snapshot) -> str:
        height, width = snapshot.layers.terrain.shape
        grid: list[list[str]] = []
        for y in range(height):
            row: list[str] = []
            for x in range(width):
                code = int(snapshot.layers.terrain[y, x])
                row.append(self._terrain_symbols.get(code, "?"))
            grid.append(row)
        if snapshot.layers.resources is not None:
            resources = snapshot.layers.resources
            maximum = float(resources.max()) if resources.size else 0.0
            if maximum > 0:
                for y in range(height):
                    for x in range(width):
                        if grid[y][x] == "." and resources[y, x] > 0:
                            grid[y][x] = "*" if resources[y, x] < maximum else "+"

        markers: list[tuple[int, int, str, str | None]] = []
        for view in snapshot.entities:
            symbol = self._symbols.get(view.kind, "?")
            color = self._colors.get(view.kind, None)
            cells = entity_cells(view)
            if not cells:
                cells = ((view.x, view.y),)
            head, *rest = cells
            markers.append((head[0], head[1], self.agent_marker if view.controller else symbol, color))
            for x, y in rest:
                markers.append((x, y, self.body_marker, color))

        for x, y, mark, color in markers:
            if 0 <= x < width and 0 <= y < height:
                if self.color and color:
                    grid[y][x] = f"\x1b[{color}m{mark}\x1b[0m"
                else:
                    grid[y][x] = mark

        return "\n".join("".join(row) for row in grid)