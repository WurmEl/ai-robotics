"""A compact FIFO replay buffer for DQN transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transition:
    observation: NDArray[np.float32]
    action: int
    reward: float
    next_observation: NDArray[np.float32]
    done: bool


class ReplayBuffer:
    """Store transitions and sample uniformly without replacement."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._transitions)

    def add(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a random batch; batch size must fit the current buffer."""
        if not 0 < batch_size <= len(self._transitions):
            raise ValueError("batch_size must be positive and no larger than the buffer.")
        indices = self._rng.choice(len(self._transitions), size=batch_size, replace=False)
        transitions = list(self._transitions)
        return [transitions[int(index)] for index in indices]
