"""Small MLP for estimating Q-values from flattened full-map observations."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Fully connected Q-network; spatial encodings can be swapped later."""

    def __init__(
        self,
        observation_size: int,
        n_actions: int = 4,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_size <= 0 or n_actions <= 0:
            raise ValueError("observation_size and n_actions must be positive.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes.")

        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, n_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value per action for each flattened observation."""
        return self.layers(observations.flatten(start_dim=1))


class CNNQNetwork(nn.Module):
    """Small spatial Q-network for full-map observations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        n_actions: int = 4,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        if channels <= 0 or height <= 0 or width <= 0 or n_actions <= 0 or hidden_size <= 0:
            raise ValueError("Network dimensions must be positive.")
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * height * width, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value per action for each spatial observation."""
        return self.head(self.features(observations).flatten(start_dim=1))
