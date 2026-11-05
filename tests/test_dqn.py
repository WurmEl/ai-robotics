import numpy as np
import pytest
import torch

from navigation.agents.dqn_agent import DQNAgent, compute_dqn_targets, select_next_values
from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import GridWorldEnv
from navigation.rl.network import DuelingCNNQNetwork
from navigation.rl.replay_buffer import ReplayBuffer, Transition


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


def test_target_network_starts_equal_to_policy_network():
    agent = DQNAgent((2, 2), seed=12)

    for name, value in agent.q_network.state_dict().items():
        torch.testing.assert_close(agent.target_network.state_dict()[name], value)


def test_replay_buffer_copies_observations():
    observation = np.zeros((3, 2, 2), dtype=np.float32)
    next_observation = np.ones((3, 2, 2), dtype=np.float32)
    buffer = ReplayBuffer(capacity=2, seed=1)
    buffer.add(
        Transition(
            observation=observation,
            action=0,
            reward=1.0,
            next_observation=next_observation,
            done=False,
        )
    )

    observation[0, 0, 0] = 9.0
    next_observation[0, 0, 0] = 9.0
    stored = buffer.sample(1)[0]

    assert stored.observation[0, 0, 0] == 0.0
    assert stored.next_observation[0, 0, 0] == 1.0


def test_cnn_network_trains_and_round_trips_checkpoint(tmp_path):
    env = GridWorldEnv(np.zeros((5, 5), dtype=np.int8), (0, 0), (4, 4))
    agent = DQNAgent(
        (5, 5),
        network="cnn",
        batch_size=2,
        replay_capacity=4,
        target_update_interval=1,
        seed=8,
    )
    initial_state = env.get_state()
    action = agent.select_action(initial_state, explore=False)
    next_state, reward, done, _ = env.step(action)
    agent.observe(initial_state, action, reward, next_state, done)
    agent.observe(initial_state, action, reward, next_state, done)

    assert agent.learn() is not None
    assert agent.q_network(torch.zeros((2, 3, 5, 5), dtype=torch.float32)).shape == (2, 4)

    path = tmp_path / "cnn.pt"
    agent.save(path)
    restored = DQNAgent((5, 5), network="cnn", seed=9)
    restored.load(path)
    assert restored.network == "cnn"


def test_dqn_bootstraps_truncation_but_not_termination():
    rewards = torch.tensor([1.0, 1.0])
    next_values = torch.tensor([5.0, 5.0])
    terminated = torch.tensor([1.0, 0.0])

    targets = compute_dqn_targets(rewards, next_values, terminated, gamma=0.9)

    assert targets.tolist() == pytest.approx([1.0, 5.5])


class _FixedQNetwork(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values.expand(observations.shape[0], -1)


def test_vanilla_next_values_take_maximum_from_target_network():
    observations = torch.zeros((1, 3, 2, 2))
    target = _FixedQNetwork([2.0, 5.0, 1.0, 4.0])

    values = select_next_values(target, observations)

    assert values.tolist() == [5.0]


def test_double_dqn_uses_online_action_and_target_value():
    observations = torch.zeros((1, 3, 2, 2))
    online = _FixedQNetwork([1.0, 7.0, 5.0, 0.0])
    target = _FixedQNetwork([9.0, 2.0, 6.0, 3.0])

    values = select_next_values(
        target,
        observations,
        double_dqn=True,
        online_network=online,
    )

    assert values.tolist() == [2.0]


def test_dueling_cnn_returns_one_q_value_per_action_and_aggregates_heads():
    network = DuelingCNNQNetwork((3, 5, 5))
    observations = torch.zeros((2, 3, 5, 5))

    with torch.no_grad():
        hidden = network.shared(network.features(observations).flatten(start_dim=1))
        values = network.value_head(hidden)
        advantages = network.advantage_head(hidden)
        expected = values + advantages - advantages.mean(dim=1, keepdim=True)

    actual = network(observations)

    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected)


def test_dueling_agent_and_checkpoint_round_trip(tmp_path):
    agent = DQNAgent((5, 5), network="cnn", dueling=True, seed=4)
    path = tmp_path / "dueling.pt"
    agent.save(path)

    restored = DQNAgent((5, 5), network="cnn", dueling=True, seed=9)
    restored.load(path)

    assert restored.dueling is True
    for key, value in agent.q_network.state_dict().items():
        torch.testing.assert_close(restored.q_network.state_dict()[key], value)


def test_dueling_is_rejected_for_mlp():
    with pytest.raises(ValueError, match="only supported with network='cnn'"):
        DQNAgent((2, 2), network="mlp", dueling=True)


def test_legacy_checkpoint_defaults_new_metadata_to_vanilla(tmp_path):
    agent = DQNAgent((2, 2), seed=4)
    path = tmp_path / "legacy.pt"
    checkpoint = {
        "map_shape": agent.map_shape,
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
        "exploration_steps": agent.exploration_steps,
        "updates": agent.updates,
        "network": "mlp",
    }
    torch.save(checkpoint, path)

    restored = DQNAgent((2, 2), seed=9)
    restored.load(path)

    assert restored.double_dqn is False
    assert restored.dueling is False
