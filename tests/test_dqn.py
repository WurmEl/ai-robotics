import numpy as np
import pytest
import torch

from navigation.agents.dqn_agent import DQNAgent
from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import GridWorldEnv


def _parameters_equal(first: DQNAgent, second: DQNAgent) -> bool:
    return all(
        torch.equal(value, second.q_network.state_dict()[name])
        for name, value in first.q_network.state_dict().items()
    )


def test_dqn_initialization_is_reproducible_for_equal_seeds():
    first = DQNAgent((2, 2), seed=19)
    second = DQNAgent((2, 2), seed=19)

    assert _parameters_equal(first, second)


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
    assert agent.q_network(torch.zeros((1, 3, 2, 2), dtype=torch.float32)).shape == (1, 4)

    model_path = tmp_path / "models" / "dqn.pt"
    agent.save(model_path)
    restored = DQNAgent((2, 2), batch_size=2, seed=6)
    restored.load(model_path)
    assert restored.updates == agent.updates
    assert restored.exploration_steps == agent.exploration_steps
    assert restored.epsilon == agent.epsilon
    for key, value in agent.q_network.state_dict().items():
        torch.testing.assert_close(restored.q_network.state_dict()[key], value)


def test_dqn_rejects_observations_with_different_map_size():
    env = GridWorldEnv(np.zeros((2, 3), dtype=np.int8), (0, 0), (1, 2))
    agent = DQNAgent((2, 2), seed=1)

    with pytest.raises(ValueError, match="Expected map shape"):
        agent.select_action(env.get_state(), explore=False)


def test_epsilon_decays_linearly_on_exploratory_selections_only():
    env = GridWorldEnv(np.zeros((2, 2), dtype=np.int8), (0, 0), (1, 1))
    agent = DQNAgent(
        (2, 2),
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=4,
        seed=3,
    )

    assert agent.epsilon == 1.0
    agent.select_action(env.get_state(), explore=False)
    assert agent.exploration_steps == 0
    assert agent.epsilon == 1.0

    for expected_step in range(1, 6):
        agent.select_action(env.get_state(), explore=True)
        assert agent.exploration_steps == expected_step
        assert agent.epsilon == pytest.approx(max(0.2, 1.0 - 0.2 * expected_step))


def test_epsilon_schedule_rejects_nonpositive_decay_steps():
    with pytest.raises(ValueError, match="epsilon_decay_steps must be positive"):
        DQNAgent((2, 2), epsilon_decay_steps=0)
