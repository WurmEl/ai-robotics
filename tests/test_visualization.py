import matplotlib

matplotlib.use("Agg")


def test_visualization_updates_robot_and_path():
    import numpy as np

    from navigation.env.grid_world import Action, GridWorldEnv
    from navigation.visualization.plot_grid import plot_grid

    env = GridWorldEnv(np.zeros((2, 2), dtype=np.int8), (0, 0), (1, 1))
    visualizer = plot_grid(env.get_state(), path=[(0, 0), (0, 1), (1, 1)])
    state, _, _, _ = env.step(Action.RIGHT)

    visualizer.update(state, path=[(0, 1), (1, 1)])

    assert list(visualizer.robot_artist.get_xdata()) == [1]
    assert list(visualizer.robot_artist.get_ydata()) == [0]
    assert list(visualizer.path_artist.get_xdata()) == [1, 1]
