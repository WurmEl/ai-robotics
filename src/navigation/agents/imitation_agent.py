"""A* behavior-cloning policy and closed-loop imitation agent."""

from __future__ import annotations

from pathlib import Path

import torch

from navigation.agents.observation import FullMapObservationEncoder
from navigation.env.grid_world import Action, GridWorldState
from navigation.rl.network import CNNPolicyNetwork


class ImitationAgent:
    """Run a trained CNN policy greedily in the shared grid-world environment."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.map_shape = map_shape
        self.observation_shape = (3, *map_shape)
        self.device = torch.device(device)
        self.encoder = FullMapObservationEncoder(map_shape)
        self.policy = CNNPolicyNetwork(self.observation_shape).to(self.device)
        self.policy.eval()

    def select_action(self, state: GridWorldState) -> Action:
        """Return the greedy action predicted for the current state."""
        observation = self.encoder.encode(state)
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = int(self.policy(tensor).argmax(dim=1).item())
        return Action(action)

    def save(self, path: str | Path) -> None:
        """Save the policy and its map shape."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "map_shape": self.map_shape,
                "policy": self.policy.state_dict(),
                "network": "cnn_policy",
            },
            model_path,
        )

    def load(self, path: str | Path) -> None:
        """Load a policy checkpoint, rejecting incompatible map shapes."""
        checkpoint = torch.load(path, map_location=self.device)
        if tuple(checkpoint["map_shape"]) != self.map_shape:
            raise ValueError(
                f"Checkpoint map shape {checkpoint['map_shape']} does not match "
                f"agent map shape {self.map_shape}."
            )
        if checkpoint.get("network", "cnn_policy") != "cnn_policy":
            raise ValueError("Checkpoint is not an imitation CNN policy.")
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()
