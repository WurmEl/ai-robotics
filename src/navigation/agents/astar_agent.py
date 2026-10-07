"""Readable A* path planner for four-directional grid worlds."""

from __future__ import annotations

import heapq
from itertools import count
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from navigation.env.grid_world import ACTION_DELTAS, Action

Position: TypeAlias = tuple[int, int]
Grid = NDArray[np.integer]


def manhattan_distance(first: Position, second: Position) -> int:
    """Return the Manhattan distance between two grid positions."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


class AStarAgent:
    """Plan with global map access; call ``select_action`` again after each step."""

    def plan_path(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
    ) -> list[Position] | None:
        """Return a shortest path including start and goal, or ``None`` if absent."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            raise ValueError("grid must be a non-empty two-dimensional NumPy array.")
        if not np.isin(grid, (0, 1)).all():
            raise ValueError("grid cells must be 0 (free) or 1 (obstacle).")
        start = self._validate_position(grid, start, "start")
        goal = self._validate_position(grid, goal, "goal")
        if grid[start] or grid[goal]:
            return None
        if start == goal:
            return [start]

        frontier: list[tuple[int, int, Position]] = []
        order = count()
        heapq.heappush(frontier, (manhattan_distance(start, goal), next(order), start))
        came_from: dict[Position, Position | None] = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct_path(came_from, goal)

            for _action, (delta_row, delta_col) in ACTION_DELTAS.items():
                neighbor = (current[0] + delta_row, current[1] + delta_col)
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue
                if grid[neighbor]:
                    continue

                new_cost = cost_so_far[current] + 1
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + manhattan_distance(neighbor, goal)
                heapq.heappush(frontier, (priority, next(order), neighbor))

        return None

    def select_action(
        self,
        grid: Grid,
        position: Position,
        goal: Position,
    ) -> Action | None:
        """Return the first action of a freshly computed path, or ``None``."""
        path = self.plan_path(grid, position, goal)
        if path is None or len(path) < 2:
            return None

        delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        for action, action_delta in ACTION_DELTAS.items():
            if delta == action_delta:
                return action
        raise RuntimeError(f"A* produced a non-adjacent path step: {path[0]} -> {path[1]}.")

    @staticmethod
    def _validate_position(grid: Grid, position: Position, name: str) -> Position:
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or not all(isinstance(value, (int, np.integer)) for value in position)
        ):
            raise ValueError(f"{name} must be a (row, column) pair.")
        result = (int(position[0]), int(position[1]))
        if not (0 <= result[0] < grid.shape[0] and 0 <= result[1] < grid.shape[1]):
            raise ValueError(f"{name} must be inside the grid.")
        return result

    @staticmethod
    def _reconstruct_path(
        came_from: dict[Position, Position | None],
        goal: Position,
    ) -> list[Position]:
        path = [goal]
        while (parent := came_from[path[-1]]) is not None:
            path.append(parent)
        path.reverse()
        return path
