import numpy as np
import torch

from navigation.agents.dqn_agent import DQNAgent
from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import GridWorldEnv


def test_full_map_encoder_returns_obstacle_robot_goal_channels():
    grid = np.array([[0, 1], [0, 0]], dtype=np.int8)
    env = GridWorldEnv(grid, (1, 0), (1, 1))
    observation = FullMapObservationEncoder((2, 2)).encode(env.get_state())

    assert observation.shape == (3, 2, 2)
    assert observation.dtype == np.float32
    assert observation[0, 0, 1] == 1.0
    assert observation[1, 1, 0] == 1.0
    assert observation[2, 1, 1] == 1.0


def test_dqn_selects_valid_action_and_learns_from_replay(tmp_path):
    env = GridWorldEnv(np.zeros((2, 2), dtype=np.int8), (0, 0), (1, 1))
    agent = DQNAgent(
        (2, 2),
        batch_size=2,
        replay_capacity=4,
        target_update_interval=1,
        seed=5,
    )
    initial_state = env.get_state()
    action = agent.select_action(initial_state, explore=False)
    assert int(action) in range(4)

    next_state, reward, done, _ = env.step(1)
    agent.observe(initial_state, action, reward, next_state, done)
    agent.observe(initial_state, action, reward, next_state, done)
    loss = agent.learn()

    assert loss is not None
    assert np.isfinite(loss)
    assert agent.q_network(
        torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    ).shape == (1, 4)

    model_path = tmp_path / "models" / "dqn.pt"
    agent.save(model_path)
    restored = DQNAgent((2, 2), batch_size=2, seed=6)
    restored.load(model_path)
    assert restored.updates == agent.updates
    for key, value in agent.q_network.state_dict().items():
        torch.testing.assert_close(restored.q_network.state_dict()[key], value)


def test_dqn_rejects_observations_with_different_map_size():
    env = GridWorldEnv(np.zeros((2, 3), dtype=np.int8), (0, 0), (1, 2))
    agent = DQNAgent((2, 2), seed=1)

    try:
        agent.select_action(env.get_state(), explore=False)
    except ValueError as error:
        assert "Expected map shape" in str(error)
    else:
        raise AssertionError("The encoder must reject a different map size.")
