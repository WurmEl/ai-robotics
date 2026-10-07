"""Convert shared environment state into DQN observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from navigation.env.grid_world import GridWorldState


class FullMapObservationEncoder:
    """Encode the full current map as obstacle, robot, and goal channels."""

    def __init__(self, map_shape: tuple[int, int]) -> None:
        if len(map_shape) != 2 or any(size <= 0 for size in map_shape):
            raise ValueError("map_shape must contain positive height and width.")
        self.map_shape = map_shape

    def encode(self, state: GridWorldState) -> NDArray[np.float32]:
        """Return a binary ``(3, height, width)`` observation.

        Channel order is obstacle, robot, goal. The state contains only the
        current environment; no future obstacle information is encoded.
        """
        grid = state.grid
        if grid.shape != self.map_shape:
            raise ValueError(f"Expected map shape {self.map_shape}, received {grid.shape}.")
        observation = np.zeros((3, *self.map_shape), dtype=np.float32)
        observation[0] = grid == 1
        observation[1, state.position[0], state.position[1]] = 1.0
        observation[2, state.goal[0], state.goal[1]] = 1.0
        return observation
