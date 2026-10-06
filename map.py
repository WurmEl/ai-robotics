# This file handles the generation and management of maps.

import numpy as np
import pickle
from collections import deque
from pathlib import Path


def bfs_distance(grid, start, goal):
    """
    Compute the shortest path length between start and goal using BFS.

    Returns:
        int: Shortest path length.
        None: If no path exists.
    """

    queue = deque([(start, 0)])
    visited = {start}

    directions = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]

    while queue:

        current, distance = queue.popleft()

        if current == goal:
            return distance

        x, y = current

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            # Check if the new position is inside the grid
            if not (0 <= nx < grid.shape[0] and
                    0 <= ny < grid.shape[1]):
                continue

            # Check if the new position is an obstacle
            if grid[nx, ny] == 1:
                continue

            next_pos = (nx, ny)

            # Skip already visited positions
            if next_pos in visited:
                continue

            visited.add(next_pos)
            queue.append((next_pos, distance + 1))

    # No path exists
    return None


def random_start_goal(grid, min_path_length=5, max_path_length=None):
    """
    Randomly select a start and goal position.

    The path between them must respect the specified length constraints.
    """

    free_cells = np.argwhere(grid == 0)

    while True:

        start_idx, goal_idx = np.random.choice(
            len(free_cells),
            size=2,
            replace=False
        )

        start = tuple(free_cells[start_idx])
        goal = tuple(free_cells[goal_idx])

        distance = bfs_distance(grid, start, goal)

        # Skip if start and goal are not connected
        if distance is None:
            continue

        # Check minimum distance
        if distance < min_path_length:
            continue

        # Check maximum distance if specified
        if max_path_length is not None and distance > max_path_length:
            continue

        return start, goal


def generate_grid(
    height,
    width,
    obstacle_probability=0.3,
    min_path_length=5,
    max_path_length=None
):
    """
    Generate a random valid grid with random start and goal positions.
    """

    while True:

        grid = (
            np.random.random((height, width))
            < obstacle_probability
        ).astype(np.int8)

        start, goal = random_start_goal(
            grid,
            min_path_length,
            max_path_length
        )

        return grid, start, goal


def generate_random_map(
    min_size=10,
    max_size=30,
    min_obstacle_probability=0.1,
    max_obstacle_probability=0.35,
    min_path_length=5,
    max_path_length=None
):
    """
    Generate a map with randomized dimensions and obstacle density.
    """

    height = np.random.randint(min_size, max_size + 1)
    width = np.random.randint(min_size, max_size + 1)

    obstacle_probability = np.random.uniform(
        min_obstacle_probability,
        max_obstacle_probability
    )

    return generate_grid(
        height,
        width,
        obstacle_probability,
        min_path_length,
        max_path_length
    )


def generate_dataset(
    nb_maps,
    min_size=10,
    max_size=30,
    min_obstacle_probability=0.1,
    max_obstacle_probability=0.35,
    min_path_length=5,
    max_path_length=None,
    save_path=None
):
    """
    Generate a dataset of randomized maps.
    """

    dataset = []

    for _ in range(nb_maps):

        grid, start, goal = generate_random_map(
            min_size=min_size,
            max_size=max_size,
            min_obstacle_probability=min_obstacle_probability,
            max_obstacle_probability=max_obstacle_probability,
            min_path_length=min_path_length,
            max_path_length=max_path_length
        )

        dataset.append((grid, start, goal))

        if save_path is not None:

            save_path = Path(save_path)
            
            # Create the parent directory if it does not exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)

    return dataset


if __name__ == "__main__":

    dataset = generate_dataset(
        nb_maps=10,
        min_size=10,
        max_size=20,
        min_obstacle_probability=0.1,
        max_obstacle_probability=0.3,
        min_path_length=10,
        max_path_length=50,
        save_path="data/train_maps.pkl"
    )

    for i, (grid, start, goal) in enumerate(dataset):

        print(f"\nMap {i}")
        print(f"Size: {grid.shape}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(grid)