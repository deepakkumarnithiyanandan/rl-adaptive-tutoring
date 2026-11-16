"""Conservative Q-learning for a fixed dataset of student interactions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CQLConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.95
    conservative_weight: float = 0.5
    target_tau: float = 0.01
    hidden_size: int = 128


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class CQLAgent:
    def __init__(
        self,
        state_size: int,
        action_names: tuple[str, ...],
        config: CQLConfig | None = None,
    ) -> None:
        if state_size < 1 or not action_names:
            raise ValueError("state_size and action_names must be non-empty")
        self.state_size = state_size
        self.action_names = action_names
        self.action_count = len(action_names)
        self.config = config or CQLConfig()
        self.q_network = QNetwork(state_size, self.action_count, self.config.hidden_size)
        self.target_network = QNetwork(
            state_size, self.action_count, self.config.hidden_size
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

    def q_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(states, dtype=torch.float32)
            return self.q_network(tensor).cpu().numpy()

    def act(self, states: np.ndarray) -> np.ndarray:
        return np.argmax(self.q_values(states), axis=1).astype(np.int64)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        state_tensor = torch.as_tensor(states, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64).view(-1, 1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float32)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32)

        q_values = self.q_network(state_tensor)
        logged_q = q_values.gather(1, action_tensor).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_state_tensor).max(dim=1).values
            target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q

        bellman_loss = nn.functional.smooth_l1_loss(logged_q, target)
        conservative_loss = (
            torch.logsumexp(q_values, dim=1).mean() - logged_q.mean()
        )
        loss = bellman_loss + self.config.conservative_weight * conservative_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.target_tau).add_(
                    parameter, alpha=self.config.target_tau
                )
        return {
            "loss": float(loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "conservative_loss": float(conservative_loss.item()),
            "mean_logged_q": float(logged_q.mean().item()),
        }

    def validation_td_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        with torch.no_grad():
            state_tensor = torch.as_tensor(states, dtype=torch.float32)
            action_tensor = torch.as_tensor(actions, dtype=torch.int64).view(-1, 1)
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
            next_state_tensor = torch.as_tensor(next_states, dtype=torch.float32)
            done_tensor = torch.as_tensor(dones, dtype=torch.float32)
            logged_q = self.q_network(state_tensor).gather(1, action_tensor).squeeze(1)
            next_q = self.target_network(next_state_tensor).max(dim=1).values
            target = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q
            return float(nn.functional.smooth_l1_loss(logged_q, target).item())

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_size": self.state_size,
                "action_names": self.action_names,
                "config": self.config.__dict__,
                "q_state_dict": self.q_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CQLAgent":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        agent = cls(
            checkpoint["state_size"],
            tuple(checkpoint["action_names"]),
            CQLConfig(**checkpoint["config"]),
        )
        agent.q_network.load_state_dict(checkpoint["q_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_state_dict"])
        agent.q_network.eval()
        agent.target_network.eval()
        return agent
