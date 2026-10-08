from __future__ import annotations

import copy
from typing import Iterator

from .components import Component


class Entity:
    __slots__ = ("id", "kind", "_components")

    def __init__(self, entity_id: int, kind: str):
        self.id = int(entity_id)
        self.kind = str(kind)
        self._components: dict[type, Component] = {}

    @property
    def components(self) -> dict[type, Component]:
        return self._components

    def add(self, component: Component) -> Component:
        self._components[type(component)] = component
        return component

    def get(self, cls: type) -> Component | None:
        return self._components.get(cls)

    def require(self, cls: type) -> Component:
        comp = self._components.get(cls)
        if comp is None:
            raise KeyError(f"Entity {self.id} ({self.kind}) has no {cls.__name__} component")
        return comp

    def has(self, cls: type) -> bool:
        return cls in self._components

    def remove(self, cls: type) -> bool:
        return self._components.pop(cls, None) is not None

    def clone(self) -> "Entity":
        clone = Entity(self.id, self.kind)
        for cls, comp in self._components.items():
            clone._components[cls] = copy.deepcopy(comp)
        return clone

    def __contains__(self, cls: type) -> bool:
        return cls in self._components

    def __iter__(self) -> Iterator[Component]:
        return iter(self._components.values())

    def __repr__(self) -> str:
        names = ",".join(c.__class__.__name__ for c in self._components.values())
        return f"Entity(id={self.id}, kind={self.kind}, components=[{names}])"
