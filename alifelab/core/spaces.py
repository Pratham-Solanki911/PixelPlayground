from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Space(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any:
        ...

    @abstractmethod
    def contains(self, value: Any) -> bool:
        ...


class DiscreteSpace(Space):
    def __init__(self, n: int, start: int = 0):
        if n <= 0:
            raise ValueError("DiscreteSpace requires n > 0")
        self.n = int(n)
        self.start = int(start)

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def size(self) -> int:
        return self.n

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.start, self.start + self.n))

    def contains(self, value: Any) -> bool:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False
        return self.start <= v < self.start + self.n

    def __repr__(self) -> str:
        return f"DiscreteSpace(n={self.n})"


class BoxSpace(Space):
    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | type = np.float32,
    ):
        if shape is None:
            shape = np.asarray(low).shape if np.asarray(low).ndim else ()
        self._shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self._shape).copy() if self._shape else np.asarray(low, dtype=self.dtype)
        self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self._shape).copy() if self._shape else np.asarray(high, dtype=self.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 1

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self.dtype.kind == "i" or self.dtype.kind == "u":
            low = np.asarray(self.low, dtype=np.int64)
            high = np.asarray(self.high, dtype=np.int64) + 1
            return rng.integers(low, high, size=self._shape, dtype=self.dtype)
        out = rng.uniform(0.0, 1.0, size=self._shape).astype(self.dtype)
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        return (low + out * (high - low)).astype(self.dtype)

    def contains(self, value: Any) -> bool:
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape != self._shape:
            return False
        return bool(np.all(arr >= self.low) and np.all(arr <= self.high))

    def __repr__(self) -> str:
        return f"BoxSpace(shape={self._shape}, dtype={self.dtype})"
