"""Generate NumPy grids and start/goal pairs with reachable paths."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Grid: TypeAlias = NDArray[np.int8]
Position: TypeAlias = tuple[int, int]
MapData: TypeAlias = tuple[Grid, Position, Position]


def _get_rng(
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        return rng
    return np.random.default_rng(seed)


def _validate_grid(grid: NDArray[np.integer]) -> None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("grid must be a two-dimensional NumPy array.")
    if grid.size == 0:
        raise ValueError("grid must not be empty.")
    if not np.isin(grid, (0, 1)).all():
        raise ValueError("grid cells must be 0 (free) or 1 (obstacle).")


def _validate_path_lengths(
    min_path_length: int,
    max_path_length: int | None,
    max_possible: int,
) -> None:
    if not isinstance(min_path_length, (int, np.integer)) or min_path_length < 0:
        raise ValueError("min_path_length must be a non-negative integer.")
    if max_path_length is not None:
        if not isinstance(max_path_length, (int, np.integer)):
            raise ValueError("max_path_length must be an integer or None.")
        if max_path_length < min_path_length:
            raise ValueError("max_path_length must be at least min_path_length.")
    if min_path_length > max_possible:
        raise ValueError("min_path_length is too large for this grid.")


def _validate_random_map_parameters(
    min_size: int,
    max_size: int,
    min_obstacle_probability: float,
    max_obstacle_probability: float,
    min_path_length: int,
    max_path_length: int | None,
    max_attempts: int,
) -> None:
    if not isinstance(min_size, (int, np.integer)) or min_size <= 0:
        raise ValueError("min_size must be a positive integer.")
    if not isinstance(max_size, (int, np.integer)) or max_size < min_size:
        raise ValueError("max_size must be an integer at least min_size.")
    if not 0 <= min_obstacle_probability <= max_obstacle_probability <= 1:
        raise ValueError("Obstacle probabilities must satisfy 0 <= min <= max <= 1.")
    _validate_path_lengths(
        min_path_length,
        max_path_length,
        max_size * max_size - 1,
    )
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")


def _reachable_distances(grid: NDArray[np.integer], start: Position) -> dict[Position, int]:
    distances = {start: 0}
    queue = deque([start])
    height, width = grid.shape

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            next_position = (next_row, next_col)
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and grid[next_position] == 0
                and next_position not in distances
            ):
                distances[next_position] = distances[(row, col)] + 1
                queue.append(next_position)

    return distances


def bfs_distance(
    grid: NDArray[np.integer],
    start: Position,
    goal: Position,
) -> int | None:
    """Return the shortest four-directional path length, or ``None`` if blocked."""
    _validate_grid(grid)
    height, width = grid.shape
    for name, position in (("start", start), ("goal", goal)):
        if (
            len(position) != 2
            or not all(isinstance(value, (int, np.integer)) for value in position)
            or not (0 <= position[0] < height and 0 <= position[1] < width)
        ):
            raise ValueError(f"{name} must be a valid (row, column) grid position.")
        if grid[position] == 1:
            return None

    return _reachable_distances(grid, start).get(goal)


def random_start_goal(
    grid: NDArray[np.integer],
    min_path_length: int = 5,
    max_path_length: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[Position, Position]:
    """Choose distinct free cells with a path length inside the requested range.

    Candidate starts are checked in randomized order. A BFS from each start
    determines every reachable goal, so this terminates even if no valid pair
    exists.
    """
    _validate_grid(grid)
    free_cells = np.argwhere(grid == 0)
    _validate_path_lengths(min_path_length, max_path_length, grid.size - 1)
    if len(free_cells) < 2:
        raise ValueError("At least two free cells are required for start and goal.")

    generator = _get_rng(rng, seed)
    for index in generator.permutation(len(free_cells)):
        start = (int(free_cells[index, 0]), int(free_cells[index, 1]))
        distances = _reachable_distances(grid, start)
        goals = [
            position
            for position, distance in distances.items()
            if distance > 0
            and distance >= min_path_length
            and (max_path_length is None or distance <= max_path_length)
        ]
        if goals:
            goal = goals[int(generator.integers(len(goals)))]
            return start, goal

    raise ValueError("No free start/goal pair satisfies the path-length constraints.")


def generate_grid(
    height: int,
    width: int,
    obstacle_probability: float = 0.3,
    min_path_length: int = 5,
    max_path_length: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    max_attempts: int = 100,
) -> MapData:
    """Generate a random binary grid containing a valid start-to-goal route."""
    if not isinstance(height, (int, np.integer)) or height <= 0:
        raise ValueError("height must be a positive integer.")
    if not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError("width must be a positive integer.")
    if not 0.0 <= obstacle_probability <= 1.0:
        raise ValueError("obstacle_probability must be between 0 and 1.")
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")
    _validate_path_lengths(min_path_length, max_path_length, height * width - 1)
    generator = _get_rng(rng, seed)

    for _ in range(max_attempts):
        grid = (generator.random((height, width)) < obstacle_probability).astype(np.int8)
        try:
            start, goal = random_start_goal(
                grid,
                min_path_length,
                max_path_length,
                rng=generator,
            )
        except ValueError:
            continue
        return grid, start, goal

    raise ValueError(
        f"Could not generate a valid map in {max_attempts} attempts; "
        "try reducing obstacle probability or path-length constraints."
    )


def generate_random_map(
    min_size: int = 10,
    max_size: int = 30,
    min_obstacle_probability: float = 0.1,
    max_obstacle_probability: float = 0.35,
    min_path_length: int = 5,
    max_path_length: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    max_attempts: int = 100,
) -> MapData:
    """Generate a valid map with randomized dimensions and obstacle density."""
    _validate_random_map_parameters(
        min_size,
        max_size,
        min_obstacle_probability,
        max_obstacle_probability,
        min_path_length,
        max_path_length,
        max_attempts,
    )

    generator = _get_rng(rng, seed)
    height = int(generator.integers(min_size, max_size + 1))
    width = int(generator.integers(min_size, max_size + 1))
    obstacle_probability = float(
        generator.uniform(min_obstacle_probability, max_obstacle_probability)
    )
    return generate_grid(
        height,
        width,
        obstacle_probability,
        min_path_length,
        max_path_length,
        rng=generator,
        max_attempts=max_attempts,
    )


def generate_dataset(
    nb_maps: int,
    min_size: int = 10,
    max_size: int = 30,
    min_obstacle_probability: float = 0.1,
    max_obstacle_probability: float = 0.35,
    min_path_length: int = 5,
    max_path_length: int | None = None,
    save_path: str | Path | None = None,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    max_attempts: int = 100,
) -> list[MapData]:
    """Generate a list of maps and optionally pickle it once when complete."""
    if not isinstance(nb_maps, (int, np.integer)) or nb_maps < 0:
        raise ValueError("nb_maps must be a non-negative integer.")
    _validate_random_map_parameters(
        min_size,
        max_size,
        min_obstacle_probability,
        max_obstacle_probability,
        min_path_length,
        max_path_length,
        max_attempts,
    )
    generator = _get_rng(rng, seed)
    dataset = [
        generate_random_map(
            min_size=min_size,
            max_size=max_size,
            min_obstacle_probability=min_obstacle_probability,
            max_obstacle_probability=max_obstacle_probability,
            min_path_length=min_path_length,
            max_path_length=max_path_length,
            rng=generator,
            max_attempts=max_attempts,
        )
        for _ in range(nb_maps)
    ]

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output_file:
            pickle.dump(dataset, output_file)

    return dataset
