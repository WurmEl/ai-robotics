"""Shared static grid-world environment for all navigation agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Position: TypeAlias = tuple[int, int]
Grid: TypeAlias = NDArray[np.integer]


class Action(IntEnum):
    """Available four-directional robot actions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class GridWorldState:
    """Current observable state of the static grid world."""

    grid: Grid
    start: Position
    position: Position
    goal: Position
    steps: int
    collisions: int
    done: bool


class GridWorldEnv:
    """A small static grid environment with shared movement and reward logic."""

    def __init__(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        *,
        step_reward: float = -0.01,
        collision_reward: float = -1.0,
        goal_reward: float = 1.0,
        max_steps: int | None = None,
    ) -> None:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            raise ValueError("grid must be a non-empty two-dimensional NumPy array.")
        if not np.isin(grid, (0, 1)).all():
            raise ValueError("grid cells must be 0 (free) or 1 (obstacle).")
        self.grid = grid.astype(np.int8, copy=True)
        self.start = self._validate_position(start, "start")
        self.goal = self._validate_position(goal, "goal")
        if self.start == self.goal:
            raise ValueError("start and goal must be different cells.")
        if self.grid[self.start] or self.grid[self.goal]:
            raise ValueError("start and goal must be on free cells.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")

        self.step_reward = float(step_reward)
        self.collision_reward = float(collision_reward)
        self.goal_reward = float(goal_reward)
        self.max_steps = max_steps
        self.reset()

    def _validate_position(self, position: Position, name: str) -> Position:
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or not all(isinstance(value, (int, np.integer)) for value in position)
        ):
            raise ValueError(f"{name} must be a (row, column) pair.")
        row, col = (int(value) for value in position)
        if not (0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]):
            raise ValueError(f"{name} must be inside the grid.")
        return row, col

    def get_state(self) -> GridWorldState:
        """Return the current state, including the global grid."""
        return GridWorldState(
            grid=self.grid,
            start=self.start,
            position=self.position,
            goal=self.goal,
            steps=self.steps,
            collisions=self.collisions,
            done=self.done,
        )

    def reset(self) -> GridWorldState:
        """Return the robot to the start and reset episode counters."""
        self.position = self.start
        self.steps = 0
        self.collisions = 0
        self.done = False
        return self.get_state()

    def update_dynamic_obstacles(self) -> None:
        """Extension point for later obstacle updates; static maps do nothing."""

    def step(self, action: int | Action) -> tuple[GridWorldState, float, bool, dict[str, object]]:
        """Apply one action and return ``(state, reward, done, info)``."""
        if self.done:
            raise RuntimeError("Episode is finished; call reset() before stepping again.")
        try:
            selected_action = Action(action)
        except ValueError as error:
            raise ValueError(
                f"action must be an Action value from 0 to 3; got {action!r}."
            ) from error

        delta_row, delta_col = ACTION_DELTAS[selected_action]
        next_position = (
            self.position[0] + delta_row,
            self.position[1] + delta_col,
        )
        collision = (
            not (0 <= next_position[0] < self.grid.shape[0])
            or not (0 <= next_position[1] < self.grid.shape[1])
            or bool(self.grid[next_position])
        )
        reward = self.collision_reward if collision else self.step_reward
        if not collision:
            self.position = next_position

        self.steps += 1
        self.update_dynamic_obstacles()
        success = self.position == self.goal
        if success:
            reward += self.goal_reward
            self.done = True
        elif self.max_steps is not None and self.steps >= self.max_steps:
            self.done = True

        info: dict[str, object] = {
            "collision": collision,
            "success": success,
            "position": self.position,
            "steps": self.steps,
            "collisions": self.collisions + int(collision),
            "termination": "goal" if success else "max_steps" if self.done else None,
        }
        if collision:
            self.collisions += 1

        return self.get_state(), reward, self.done, info
