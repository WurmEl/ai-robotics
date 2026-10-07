"""Basic matplotlib visualization for a grid-world episode."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure, SubFigure

from navigation.env.grid_world import GridWorldState, Position


class GridWorldVisualizer:
    """Plot a grid and update its robot/path artists as the episode advances."""

    def __init__(
        self,
        state: GridWorldState,
        path: Sequence[Position] | None = None,
        ax: Axes | None = None,
    ) -> None:
        self.figure: Figure | SubFigure
        self.ax: Axes
        if ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.figure = ax.figure

        height, width = state.grid.shape
        self.ax.imshow(
            state.grid,
            cmap=ListedColormap(["#f2f2f2", "#252525"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self.ax.set_xticks(range(width))
        self.ax.set_yticks(range(height))
        self.ax.set_xticks([value - 0.5 for value in range(width + 1)], minor=True)
        self.ax.set_yticks([value - 0.5 for value in range(height + 1)], minor=True)
        self.ax.grid(which="minor", color="#999999", linewidth=0.5)
        self.ax.tick_params(which="minor", bottom=False, left=False)
        self.ax.set_xlim(-0.5, width - 0.5)
        self.ax.set_ylim(height - 0.5, -0.5)
        self.ax.set_aspect("equal")

        (self.path_artist,) = self.ax.plot([], [], color="#d49b00", linewidth=2)
        (self.robot_artist,) = self.ax.plot([], [], "o", color="#1976d2", markersize=9)
        (self.goal_artist,) = self.ax.plot([], [], "*", color="#2e8b57", markersize=13)
        self.update(state, path)

    def update(
        self,
        state: GridWorldState,
        path: Sequence[Position] | None = None,
    ) -> None:
        """Update robot position and optional path for one environment step."""
        robot_row, robot_col = state.position
        goal_row, goal_col = state.goal
        self.robot_artist.set_data([robot_col], [robot_row])
        self.goal_artist.set_data([goal_col], [goal_row])
        if path:
            self.path_artist.set_data(
                [position[1] for position in path],
                [position[0] for position in path],
            )
        else:
            self.path_artist.set_data([], [])
        self.ax.set_title(f"Steps: {state.steps} | Collisions: {state.collisions}")
        self.figure.canvas.draw_idle()


def plot_grid(
    state: GridWorldState,
    path: Sequence[Position] | None = None,
    ax: Axes | None = None,
) -> GridWorldVisualizer:
    """Create and return a visualization that can be updated step by step."""
    return GridWorldVisualizer(state, path=path, ax=ax)
