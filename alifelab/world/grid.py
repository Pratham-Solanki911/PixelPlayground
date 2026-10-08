from __future__ import annotations


class Grid:
    def __init__(self, width: int, height: int, wrap: bool = True):
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")
        self.width = int(width)
        self.height = int(height)
        self.wrap = bool(wrap)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def size(self) -> int:
        return self.width * self.height

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def normalize(self, x: int, y: int) -> tuple[int, int]:
        if self.wrap:
            return (x % self.width, y % self.height)
        return (max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y)))

    def delta(self, ax: int, ay: int, bx: int, by: int) -> tuple[int, int]:
        dx = bx - ax
        dy = by - ay
        if self.wrap:
            if abs(dx) > self.width // 2:
                dx -= self.width if dx > 0 else -self.width
            if abs(dy) > self.height // 2:
                dy -= self.height if dy > 0 else -self.height
        return (dx, dy)

    def chebyshev(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.wrap:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return max(dx, dy)

    def manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.wrap:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return dx + dy

    def cells(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width)]
