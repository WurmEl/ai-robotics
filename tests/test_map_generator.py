import pickle

import numpy as np
import pytest

from navigation.env.map_generator import (
    bfs_distance,
    generate_dataset,
    generate_grid,
    random_start_goal,
)


def test_bfs_distance_returns_shortest_path_length():
    grid = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )

    assert bfs_distance(grid, (0, 0), (2, 2)) == 4


def test_generated_grid_has_a_valid_route_and_seed_is_reproducible():
    first = generate_grid(8, 9, obstacle_probability=0.2, seed=17)
    second = generate_grid(8, 9, obstacle_probability=0.2, seed=17)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert bfs_distance(first[0], first[1], first[2]) is not None


def test_start_goal_selection_fails_when_no_valid_pair_exists():
    grid = np.ones((4, 4), dtype=np.int8)
    grid[0, 0] = 0

    with pytest.raises(ValueError, match="At least two free cells"):
        random_start_goal(grid)


def test_dataset_is_saved_as_one_complete_list(tmp_path):
    path = tmp_path / "nested" / "maps.pkl"
    dataset = generate_dataset(
        3,
        min_size=5,
        max_size=5,
        min_obstacle_probability=0.0,
        max_obstacle_probability=0.0,
        min_path_length=2,
        save_path=path,
        seed=23,
    )

    with path.open("rb") as saved_file:
        saved_dataset = pickle.load(saved_file)
    assert len(dataset) == 3
    assert len(saved_dataset) == 3
    assert all(bfs_distance(grid, start, goal) is not None for grid, start, goal in dataset)


def test_empty_dataset_still_validates_generation_parameters():
    with pytest.raises(ValueError, match="min_size"):
        generate_dataset(0, min_size=0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 4), "height"),
        ((4, 4, 1.5), "obstacle_probability"),
    ],
)
def test_generate_grid_rejects_invalid_parameters(arguments, message):
    with pytest.raises(ValueError, match=message):
        generate_grid(*arguments)
