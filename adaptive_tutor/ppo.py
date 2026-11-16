"""Minimal PPO actor-critic implementation for discrete tutoring actions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_size: int = 64


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_count: int,
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.observation_size = observation_size
        self.action_count = action_count
        self.model = ActorCritic(
            observation_size, action_count, self.config.hidden_size
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model(obs)
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            return int(Categorical(logits=logits).sample().item())

    def sample(self, observation: np.ndarray) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probabilities: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        obs = torch.as_tensor(observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64)
        old_logp = torch.as_tensor(old_log_probabilities, dtype=torch.float32)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.config.minibatch_size):
                batch = indices[start : start + self.config.minibatch_size]
                logits, values = self.model(obs[batch])
                distribution = Categorical(logits=logits)
                new_logp = distribution.log_prob(action_tensor[batch])
                ratio = torch.exp(new_logp - old_logp[batch])
                unclipped = ratio * advantage_tensor[batch]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantage_tensor[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, return_tensor[batch])
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float((old_logp[batch] - new_logp).mean().item()))

        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "observation_size": self.observation_size,
            "action_count": self.action_count,
            "config": self.config.__dict__,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        agent = cls(
            checkpoint["observation_size"],
            checkpoint["action_count"],
            PPOConfig(**checkpoint["config"]),
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.model.eval()
        return agent


def generalized_advantages(
    rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[index] = last_advantage
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns
