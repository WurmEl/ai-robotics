"""DQN agent using full current-map observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import Action, GridWorldState
from navigation.rl.network import QNetwork
from navigation.rl.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """Epsilon-greedy DQN; its observations include the full current map."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_update_interval: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if batch_size <= 0 or target_update_interval <= 0:
            raise ValueError("batch_size and target_update_interval must be positive.")
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.encoder = FullMapObservationEncoder(map_shape)
        self.map_shape = map_shape
        self.observation_shape = (3, *map_shape)
        observation_size = int(np.prod(self.observation_shape))
        self.device = torch.device(device)
        self.q_network = QNetwork(observation_size, hidden_sizes=hidden_sizes).to(self.device)
        self.target_network = QNetwork(
            observation_size,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.updates = 0
        self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: GridWorldState, *, explore: bool = True) -> Action:
        """Select an action from the current full-map state."""
        if explore:
            choose_randomly = self._rng.random() < self.epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if choose_randomly:
                return Action(int(self._rng.integers(len(Action))))

        observation = self.encoder.encode(state)
        input_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action_index = int(self.q_network(input_tensor).argmax(dim=1).item())
        return Action(action_index)

    def observe(
        self,
        state: GridWorldState,
        action: Action | int,
        reward: float,
        next_state: GridWorldState,
        done: bool,
    ) -> None:
        """Encode and store one transition from the shared environment."""
        self.replay_buffer.add(
            Transition(
                observation=self.encoder.encode(state),
                action=int(action),
                reward=float(reward),
                next_observation=self.encoder.encode(next_state),
                done=bool(done),
            )
        )

    def learn(self) -> float | None:
        """Run one minibatch update, or return ``None`` until a batch is ready."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        observations = torch.as_tensor(
            np.stack([item.observation for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observation for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        predicted_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_observations).max(dim=1).values
            target_values = rewards + self.gamma * next_values * (1.0 - dones)

        loss = self.loss_function(predicted_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        """Save network parameters and the fixed map shape."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "map_shape": self.map_shape,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "epsilon": self.epsilon,
                "updates": self.updates,
            },
            model_path,
        )

    def load(self, path: str | Path) -> None:
        """Load a saved model, rejecting checkpoints for different map sizes."""
        checkpoint = torch.load(path, map_location=self.device)
        if tuple(checkpoint["map_shape"]) != self.map_shape:
            raise ValueError(
                f"Checkpoint map shape {checkpoint['map_shape']} does not match "
                f"agent map shape {self.map_shape}."
            )
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.epsilon = float(checkpoint["epsilon"])
        self.updates = int(checkpoint["updates"])
