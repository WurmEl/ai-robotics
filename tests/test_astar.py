import numpy as np

from navigation.agents.astar_agent import AStarAgent
from navigation.env.grid_world import Action


def test_astar_finds_shortest_path_on_known_map():
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    agent = AStarAgent()

    path = agent.plan_path(grid, (0, 0), (0, 3))

    assert path is not None
    assert len(path) - 1 == 3
    assert agent.select_action(grid, (0, 0), (0, 3)) == Action.RIGHT


def test_astar_reports_no_path_when_goal_is_sealed():
    grid = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )

    assert AStarAgent().plan_path(grid, (0, 0), (1, 1)) is None
