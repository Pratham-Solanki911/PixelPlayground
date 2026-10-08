from __future__ import annotations

from typing import Any, Callable, Iterable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any = None) -> Callable[[Any], Any]:
        if obj is not None:
            self._add(name, obj)
            return obj

        def decorator(target: Any) -> Any:
            self._add(name, target)
            return target

        return decorator

    def _add(self, name: str, obj: Any) -> None:
        key = name.lower()
        if key in self._items and self._items[key] is not obj:
            raise KeyError(f"{self.kind} '{name}' is already registered")
        self._items[key] = obj

    def get(self, name: str) -> Any:
        key = str(name).lower()
        if key not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}")
        return self._items[key]

    def create(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._items

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._items))


RULES = Registry("rule")
OBSERVATIONS = Registry("observation")
ACTIONS = Registry("action space")
REWARDS = Registry("reward function")
AGENTS = Registry("agent")
ARCHITECTURES = Registry("architecture")
FEATURE_EXTRACTORS = Registry("feature extractor")
COMPONENTS = Registry("component")
PRESETS = Registry("entity preset")


def register_rule(name: str) -> Callable[[Any], Any]:
    return RULES.register(name)


def register_observation(name: str) -> Callable[[Any], Any]:
    return OBSERVATIONS.register(name)


def register_action(name: str) -> Callable[[Any], Any]:
    return ACTIONS.register(name)


def register_reward(name: str) -> Callable[[Any], Any]:
    return REWARDS.register(name)


def register_agent(name: str) -> Callable[[Any], Any]:
    return AGENTS.register(name)


def register_extractor(name: str) -> Callable[[Any], Any]:
    return FEATURE_EXTRACTORS.register(name)


def register_architecture(name: str) -> Callable[[Any], Any]:
    return ARCHITECTURES.register(name)


def register_component(name: str) -> Callable[[Any], Any]:
    return COMPONENTS.register(name)


def register_preset(name: str) -> Callable[[Any], Any]:
    return PRESETS.register(name)
