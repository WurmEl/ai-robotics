import numpy as np

from navigation.env.grid_world import Action, GridWorldEnv


def test_environment_moves_and_counts_steps():
    env = GridWorldEnv(np.zeros((3, 3), dtype=np.int8), (1, 1), (0, 2))

    state, reward, done, info = env.step(Action.UP)

    assert state.position == (0, 1)
    assert reward == env.step_reward
    assert done is False
    assert info["steps"] == 1


def test_obstacle_and_wall_moves_are_collisions():
    grid = np.array([[0, 1], [0, 0]], dtype=np.int8)
    env = GridWorldEnv(grid, (0, 0), (1, 1))

    state, reward, done, info = env.step(Action.RIGHT)
    assert state.position == (0, 0)
    assert reward == env.collision_reward
    assert done is False
    assert info["collision"] is True
    assert state.collisions == 1

    state, _, _, info = env.step(Action.UP)
    assert state.position == (0, 0)
    assert info["collision"] is True


def test_reaching_goal_terminates_episode():
    env = GridWorldEnv(np.zeros((1, 2), dtype=np.int8), (0, 0), (0, 1))

    state, reward, done, info = env.step(Action.RIGHT)

    assert state.position == (0, 1)
    assert reward == env.step_reward + env.goal_reward
    assert done is True
    assert info["success"] is True


def test_reset_clears_episode_counters():
    env = GridWorldEnv(np.zeros((2, 2), dtype=np.int8), (0, 0), (1, 1))
    env.step(Action.UP)

    state = env.reset()

    assert state.position == (0, 0)
    assert state.steps == 0
    assert state.collisions == 0
