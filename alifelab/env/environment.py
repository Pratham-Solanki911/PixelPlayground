from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.interfaces import ActionSpace, ObservationBuilder, RewardContext, RewardFunction
from ..core.snapshot import EntityView, WorldSnapshot
from ..world.components import Controller
from ..world.engine import WorldEngine


class LifeEnv:
    def __init__(
        self,
        world: WorldEngine,
        action_interface: ActionSpace,
        observation_builder: ObservationBuilder,
        reward_function: RewardFunction,
        max_steps: int = 200,
        controller: str = "agent",
        control_switch: str = "none",
        seed: int | None = None,
    ):
        self.world = world
        self.action_interface = action_interface
        self.observation_builder = observation_builder
        self.reward_function = reward_function
        self.max_steps = int(max_steps)
        self.controller_name = controller
        self.control_switch = control_switch
        if seed is not None:
            self.world.seed = int(seed)
        self.episode = 0
        self.agent_id: int | None = None
        self._controlled_ids: set[int] = set()
        self._steps = 0
        self._done = False
        self._episode_return = 0.0
        self._prev_energy: float | None = None
        self._prev_health: float | None = None
        self._prev_size: float | None = None
        self._metrics: dict[str, Any] = {}
        self._renderer = None

    @property
    def observation_space(self):
        return self.observation_builder.space

    @property
    def action_space(self):
        return self.action_interface.space

    @property
    def grid_size(self) -> tuple[int, int]:
        return (self.world.height, self.world.width)

    def set_renderer(self, renderer: Any) -> None:
        self._renderer = renderer

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.episode += 1
        if seed is not None:
            self.world.seed = int(seed)
        self.world.reset()
        self.world.episode = self.episode
        self.observation_builder.reset()
        self.reward_function.reset()
        self._steps = 0
        self._done = False
        self._episode_return = 0.0
        self._controlled_ids = set()
        self._select_agent(initial=True)
        self._metrics = {
            "food": 0,
            "predation": 0,
            "predation_attempts": 0,
            "births": 0,
            "deaths": 0,
            "mutated": 0,
            "energy_gain": 0.0,
            "died": False,
            "steps": 0,
        }
        view = self._agent_view()
        self._prev_energy = view.energy if view else None
        self._prev_health = view.health if view else None
        self._prev_size = view.size if view else None
        snapshot = self.world.snapshot()
        observation = self.observation_builder.build(snapshot, self.agent_id)
        return observation, self._info(snapshot, alive=view is not None)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode has ended; call reset() before step()")
        intent = self.action_interface.decode(action)
        if self.agent_id is not None and self.agent_id in self.world.entities:
            self.world.set_intent(self.agent_id, intent)
        events = self.world.step()
        self._steps += 1
        self._accumulate(events)

        snapshot = self.world.snapshot()
        view = self._view_from(snapshot)
        if view is None and self.agent_id is not None:
            if self.control_switch == "newest":
                self._select_agent(initial=False)
                view = self._view_from(snapshot)
        alive = view is not None
        terminated = not alive
        truncated = (not terminated) and self.max_steps > 0 and self._steps >= self.max_steps
        done = terminated or truncated

        observation = self.observation_builder.build(snapshot, self.agent_id)

        ctx = RewardContext(
            tick=self.world.tick,
            agent_id=self.agent_id,
            events=events,
            energy=view.energy if view else None,
            prev_energy=self._prev_energy,
            health=view.health if view else None,
            prev_health=self._prev_health,
            size=view.size if view else 1.0,
            prev_size=self._prev_size if self._prev_size is not None else (view.size if view else 1.0),
            age=view.age if view else 0.0,
            population=len(self.world.entities),
            died=not alive,
            done=done,
        )
        reward = float(self.reward_function.evaluate(ctx))
        self._episode_return += reward
        if view is not None:
            self._prev_energy = view.energy
            self._prev_health = view.health
            self._prev_size = view.size
        else:
            self._prev_energy = None
            self._prev_health = None
            self._prev_size = None

        self._done = done
        info = self._info(snapshot, alive=alive)
        info["events"] = [e.kind for e in events]
        if done:
            info["metrics"] = self.final_metrics()
            info["episode_return"] = self._episode_return
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("No renderer attached; call env.set_renderer(renderer)")
        snapshot = self.world.snapshot()
        if mode == "rgb_array":
            return self._renderer.render(snapshot)
        raise ValueError(f"Unsupported render mode '{mode}'")

    def close(self) -> None:
        self._renderer = None

    def get_state(self) -> dict[str, Any]:
        state = self.world.get_state()
        state["episode"] = self.episode
        state["steps"] = self._steps
        state["return"] = self._episode_return
        return state

    @property
    def episode_return(self) -> float:
        return self._episode_return

    def _select_agent(self, initial: bool) -> None:
        candidates = []
        for entity in self.world.get_entities():
            controller = entity.get(Controller)
            if controller is not None and controller.name == self.controller_name:
                candidates.append(entity.id)
        if not candidates:
            self.agent_id = None
            return
        candidates.sort()
        if initial or self.control_switch != "newest":
            chosen = candidates[0]
        else:
            chosen = candidates[-1]
        self.agent_id = chosen
        self._controlled_ids.add(chosen)

    def _agent_view(self) -> EntityView | None:
        if self.agent_id is None:
            return None
        return self.world.snapshot().by_id.get(self.agent_id)

    def _view_from(self, snapshot: WorldSnapshot) -> EntityView | None:
        if self.agent_id is None:
            return None
        return snapshot.by_id.get(self.agent_id)

    def _accumulate(self, events) -> None:
        for event in events:
            if event.kind == "consume" and event.actor_id in self._controlled_ids:
                self._metrics["food"] += 1
                self._metrics["energy_gain"] += float(event.data.get("energy", 0.0))
            elif event.kind == "predation" and event.actor_id in self._controlled_ids:
                self._metrics["predation"] += 1
                self._metrics["energy_gain"] += float(event.data.get("energy_gain", 0.0))
            elif event.kind == "predation_attempt" and event.actor_id in self._controlled_ids:
                self._metrics["predation_attempts"] += 1
            elif event.kind == "reproduce" and event.actor_id in self._controlled_ids:
                self._metrics["births"] += 1
            elif event.kind == "mutate" and event.actor_id in self._controlled_ids:
                self._metrics["mutated"] += 1
            elif event.kind == "death":
                self._metrics["deaths"] += 1
                if event.actor_id in self._controlled_ids:
                    self._metrics["died"] = True
        self._metrics["steps"] = self._steps

    def final_metrics(self) -> dict[str, Any]:
        view = self._agent_view()
        metrics = dict(self._metrics)
        metrics["survived"] = not self._metrics["died"]
        metrics["energy_final"] = view.energy if view and view.energy is not None else 0.0
        metrics["health_final"] = view.health if view and view.health is not None else 0.0
        metrics["size_final"] = view.size if view else 0.0
        metrics["age_final"] = view.age if view else 0.0
        metrics["population"] = dict(self.world.population_counts())
        metrics["population_total"] = len(self.world.entities)
        metrics["return"] = self._episode_return
        metrics["length"] = self._steps
        return metrics

    def _info(self, snapshot: WorldSnapshot, alive: bool) -> dict[str, Any]:
        return {
            "tick": self.world.tick,
            "steps": self._steps,
            "episode": self.episode,
            "agent_id": self.agent_id,
            "alive": alive,
            "population": dict(snapshot.population),
            "population_total": len(self.world.entities),
            "metrics": dict(self._metrics),
            "return": self._episode_return,
        }
