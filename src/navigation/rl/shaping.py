"""Optional potential-based reward shaping helpers."""

from __future__ import annotations

from navigation.env.grid_world import GridWorldState


def normalized_manhattan_potential(state: GridWorldState) -> float:
    """Return ``-distance / (height + width - 2)`` for a state."""
    denominator = max(1, state.grid.shape[0] + state.grid.shape[1] - 2)
    distance = abs(state.position[0] - state.goal[0]) + abs(state.position[1] - state.goal[1])
    return -distance / denominator


def potential_shaping_reward(
    state: GridWorldState,
    next_state: GridWorldState,
    *,
    alpha: float,
    gamma: float,
) -> float:
    """Return ``alpha * (gamma * Phi(next) - Phi(state))``."""
    return alpha * (
        gamma * normalized_manhattan_potential(next_state) - normalized_manhattan_potential(state)
    )
