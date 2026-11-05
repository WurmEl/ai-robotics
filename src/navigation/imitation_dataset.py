"""Utilities for generating A*-labeled behavior-cloning datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from navigation.agents.astar_agent import AStarAgent
from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import ACTION_DELTAS, Action, GridWorldState
from navigation.env.map_generator import load_layout_dataset, random_start_goal


def generate_imitation_dataset(
    layouts_path: str | Path,
    *,
    height: int,
    width: int,
    task_seed: int,
    tasks_per_layout: int,
    min_path_length: int = 2,
    max_path_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate full-map observations and deterministic A* action labels."""
    if tasks_per_layout <= 0:
        raise ValueError("tasks_per_layout must be positive.")
    layouts = load_layout_dataset(layouts_path)
    if not layouts:
        raise ValueError("The layout dataset must contain at least one layout.")
    if any(grid.shape != (height, width) for grid in layouts):
        raise ValueError("All layouts must match the requested height and width.")

    rng = np.random.default_rng(task_seed)
    encoder = FullMapObservationEncoder((height, width))
    planner = AStarAgent()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    task_count = 0

    for layout_index, grid in enumerate(layouts):
        for task_index in range(tasks_per_layout):
            try:
                start, goal = random_start_goal(
                    grid,
                    min_path_length=min_path_length,
                    max_path_length=max_path_length,
                    rng=rng,
                )
            except ValueError as error:
                raise ValueError(
                    f"Could not generate task {task_index + 1} for layout {layout_index}."
                ) from error
            path = planner.plan_path(grid, start, goal)
            if path is None or len(path) < 2:
                raise RuntimeError("A* failed to produce a path for a valid task.")
            task_count += 1
            for position, next_position in zip(path, path[1:], strict=False):
                delta = (next_position[0] - position[0], next_position[1] - position[1])
                action = next(
                    (
                        candidate
                        for candidate, candidate_delta in ACTION_DELTAS.items()
                        if candidate_delta == delta
                    ),
                    None,
                )
                if action is None:
                    raise RuntimeError(
                        f"Invalid A* path transition: {position} -> {next_position}."
                    )
                state = GridWorldState(
                    grid=grid,
                    start=start,
                    position=position,
                    goal=goal,
                    steps=0,
                    collisions=0,
                    done=False,
                )
                observations.append(encoder.encode(state))
                actions.append(int(Action(action)))

    return np.stack(observations), np.asarray(actions, dtype=np.int64), task_count


def save_imitation_dataset(
    path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    height: int,
    width: int,
    task_count: int,
    task_seed: int,
    source_layouts_path: str | Path,
) -> None:
    """Save an imitation dataset as a compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        height=np.int64(height),
        width=np.int64(width),
        task_count=np.int64(task_count),
        sample_count=np.int64(len(actions)),
        task_seed=np.int64(task_seed),
        source_layouts_path=np.asarray(str(source_layouts_path)),
    )
