"""Train PPO, log diagnostics, and select a checkpoint on validation learners."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from adaptive_tutor.env import TutoringEnv
from adaptive_tutor.evaluation import evaluate_policies, summarize_results
from adaptive_tutor.ppo import PPOAgent, generalized_advantages


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = TutoringEnv(profile_split="train")
    agent = PPOAgent(env.observation_size, env.action_count)
    training_rows: list[dict[str, float | int]] = []
    validation_rows: list[dict[str, float | int]] = []
    best_gain = float("-inf")
    episode_number = 0
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        rollout_observations: list[np.ndarray] = []
        rollout_actions: list[int] = []
        rollout_log_probabilities: list[float] = []
        rollout_advantages: list[float] = []
        rollout_returns: list[float] = []
        episode_returns: list[float] = []
        episode_gains: list[float] = []

        for _ in range(args.episodes_per_update):
            observation, _ = env.reset(seed=args.seed + episode_number)
            episode_number += 1
            observations: list[np.ndarray] = []
            actions: list[int] = []
            log_probabilities: list[float] = []
            rewards: list[float] = []
            values: list[float] = []
            total_reward = 0.0
            initial_mastery = float(env.mastery.mean())
            done = False

            while not done:
                action, log_probability, value = agent.sample(observation)
                next_observation, reward, done, _ = env.step(action)
                observations.append(observation)
                actions.append(action)
                log_probabilities.append(log_probability)
                rewards.append(reward)
                values.append(value)
                total_reward += reward
                observation = next_observation

            advantages, returns = generalized_advantages(
                rewards, values, agent.config.gamma, agent.config.gae_lambda
            )
            rollout_observations.extend(observations)
            rollout_actions.extend(actions)
            rollout_log_probabilities.extend(log_probabilities)
            rollout_advantages.extend(advantages.tolist())
            rollout_returns.extend(returns.tolist())
            episode_returns.append(total_reward)
            episode_gains.append(float(env.mastery.mean()) - initial_mastery)

        update_metrics = agent.update(
            np.asarray(rollout_observations, dtype=np.float32),
            np.asarray(rollout_actions, dtype=np.int64),
            np.asarray(rollout_log_probabilities, dtype=np.float32),
            np.asarray(rollout_advantages, dtype=np.float32),
            np.asarray(rollout_returns, dtype=np.float32),
        )
        training_rows.append(
            {
                "update": update,
                "mean_episode_reward": float(np.mean(episode_returns)),
                "mean_mastery_gain": float(np.mean(episode_gains)),
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "approx_kl": update_metrics["approx_kl"],
            }
        )

        if update % args.eval_interval == 0 or update == 1 or update == args.updates:
            validation = evaluate_policies(
                ["ppo"],
                episodes=args.validation_episodes,
                split="validation",
                base_seed=args.validation_seed,
                agent=agent,
            )["ppo"]
            metrics = summarize_results({"ppo": validation})["ppo"]
            validation_gain = metrics["mastery_gain"]["mean"]
            validation_rows.append(
                {
                    "update": update,
                    "mastery_gain": validation_gain,
                    "mastery_gain_ci95_low": metrics["mastery_gain"]["ci95_low"],
                    "mastery_gain_ci95_high": metrics["mastery_gain"]["ci95_high"],
                    "final_mastery": metrics["final_mastery"]["mean"],
                    "topic_coverage": metrics["topic_coverage"]["mean"],
                }
            )
            agent.save(output_dir / "latest_policy.pt")
            if validation_gain > best_gain:
                best_gain = validation_gain
                agent.save(output_dir / "best_policy.pt")
            print(
                f"update={update:4d} "
                f"train_return={np.mean(episode_returns):.3f} "
                f"validation_mastery_gain={validation_gain:.3f} "
                f"best={best_gain:.3f}"
            )

    _write_rows(output_dir / "training.csv", training_rows)
    _write_rows(output_dir / "validation.csv", validation_rows)
    print(f"Training completed in {time.perf_counter() - started:.1f}s")
    print(f"Selected checkpoint: {output_dir / 'best_policy.pt'}")


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=40)
    parser.add_argument("--validation-seed", type=int, default=20_000)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    if args.updates < 1 or args.episodes_per_update < 1 or args.eval_interval < 1:
        parser.error("updates, episodes-per-update, and eval-interval must be positive")
    train(args)


if __name__ == "__main__":
    main()
