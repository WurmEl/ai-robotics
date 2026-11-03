import numpy as np

from navigation.env.grid_world import GridWorldEnv
from navigation.rl.shaping import potential_shaping_reward


def test_normalized_shaping_at_same_distance_does_not_cancel_step_penalty():
    env = GridWorldEnv(np.zeros((10, 10), dtype=np.int8), (4, 4), (0, 0))
    state = env.get_state()
    shaping = potential_shaping_reward(state, state, alpha=0.1, gamma=0.99)

    assert abs(shaping) < 0.01
    assert -0.01 + shaping < 0
