import pickle

import numpy as np
import pytest
import torch

from navigation.agents.imitation_agent import ImitationAgent
from navigation.env.grid_world import ACTION_DELTAS, Action, GridWorldEnv
from navigation.imitation_dataset import generate_imitation_dataset, save_imitation_dataset
from navigation.rl.network import CNNPolicyNetwork


def _write_layouts(path, layouts):
    with path.open("wb") as output_file:
        pickle.dump(layouts, output_file)


def test_expert_dataset_has_valid_pairs_and_reaches_goal(tmp_path):
    grid = np.zeros((4, 4), dtype=np.int8)
    layouts_path = tmp_path / "layouts.pkl"
    _write_layouts(layouts_path, [grid])

    observations, actions, task_count = generate_imitation_dataset(
        layouts_path,
        height=4,
        width=4,
        task_seed=7,
        tasks_per_layout=1,
        min_path_length=3,
    )

    assert task_count == 1
    assert observations.shape[1:] == (3, 4, 4)
    assert np.all(np.isin(actions, [int(action) for action in Action]))

    robot = tuple(np.argwhere(observations[0, 1] == 1)[0])
    goal = tuple(np.argwhere(observations[0, 2] == 1)[0])
    position = (int(robot[0]), int(robot[1]))
    for action_value in actions:
        position = tuple(
            position[index] + ACTION_DELTAS[Action(int(action_value))][index] for index in range(2)
        )
    assert position == (int(goal[0]), int(goal[1]))


def test_imitation_dataset_is_reproducible(tmp_path):
    layouts_path = tmp_path / "layouts.pkl"
    _write_layouts(layouts_path, [np.zeros((4, 4), dtype=np.int8)])

    first = generate_imitation_dataset(
        layouts_path,
        height=4,
        width=4,
        task_seed=11,
        tasks_per_layout=2,
        min_path_length=2,
    )
    second = generate_imitation_dataset(
        layouts_path,
        height=4,
        width=4,
        task_seed=11,
        tasks_per_layout=2,
        min_path_length=2,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_cnn_policy_output_and_training_loss_are_finite():
    model = CNNPolicyNetwork((3, 4, 4))
    observations = torch.zeros((2, 3, 4, 4))
    actions = torch.tensor([Action.UP, Action.RIGHT])
    loss = torch.nn.CrossEntropyLoss()(model(observations), actions)

    assert model(observations).shape == (2, 4)
    assert torch.isfinite(loss)


def test_imitation_agent_returns_valid_action_and_round_trips_checkpoint(tmp_path):
    env = GridWorldEnv(np.zeros((3, 3), dtype=np.int8), (0, 0), (2, 2))
    agent = ImitationAgent((3, 3))
    assert agent.select_action(env.get_state()) in list(Action)

    path = tmp_path / "policy.pt"
    agent.save(path)
    restored = ImitationAgent((3, 3))
    restored.load(path)
    for name, value in agent.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value)


def test_saved_imitation_dataset_contains_metadata(tmp_path):
    layouts_path = tmp_path / "layouts.pkl"
    output_path = tmp_path / "dataset.npz"
    _write_layouts(layouts_path, [np.zeros((3, 3), dtype=np.int8)])
    observations, actions, task_count = generate_imitation_dataset(
        layouts_path,
        height=3,
        width=3,
        task_seed=3,
        tasks_per_layout=1,
        min_path_length=2,
    )
    save_imitation_dataset(
        output_path,
        observations,
        actions,
        height=3,
        width=3,
        task_count=task_count,
        task_seed=3,
        source_layouts_path=layouts_path,
    )

    with np.load(output_path) as data:
        assert int(data["task_count"]) == 1
        assert int(data["sample_count"]) == len(actions)
        assert int(data["task_seed"]) == 3


def test_dataset_generation_rejects_incompatible_layout_shape(tmp_path):
    layouts_path = tmp_path / "layouts.pkl"
    _write_layouts(layouts_path, [np.zeros((2, 2), dtype=np.int8)])

    with pytest.raises(ValueError, match="match the requested height and width"):
        generate_imitation_dataset(
            layouts_path,
            height=3,
            width=3,
            task_seed=1,
            tasks_per_layout=1,
        )
