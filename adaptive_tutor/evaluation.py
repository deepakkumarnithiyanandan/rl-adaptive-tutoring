"""Matched-scenario policy evaluation and report helpers."""

import csv
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from adaptive_tutor.baselines import (
    Policy,
    adaptive_heuristic_policy,
    random_policy,
    round_robin_policy,
)
from adaptive_tutor.env import TutoringEnv


class PolicyModel(Protocol):
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int: ...


def _policy(name: str, agent: PolicyModel | None, env: TutoringEnv) -> Policy:
    if name == "ppo":
        if agent is None:
            raise ValueError("PPO policy requires a trained agent")
        return lambda obs, step, rng: agent.act(obs, deterministic=True)
    if name == "random":
        return lambda obs, step, rng: random_policy(
            obs, step, rng, action_count=env.action_count
        )
    if name == "round_robin":
        return lambda obs, step, rng: round_robin_policy(
            obs,
            step,
            rng,
            num_topics=env.num_topics,
            difficulty_count=len(env.difficulty_levels),
            middle_difficulty_index=1,
        )
    if name == "adaptive_heuristic":
        return lambda obs, step, rng: adaptive_heuristic_policy(
            obs,
            step,
            rng,
            num_topics=env.num_topics,
            difficulty_levels=env.difficulty_levels,
        )
    raise ValueError(f"unknown policy: {name}")


def evaluate_policies(
    policy_names: list[str],
    episodes: int,
    split: str,
    base_seed: int,
    agent: PolicyModel | None = None,
) -> dict[str, list[dict[str, float | int | str]]]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    results: dict[str, list[dict[str, float | int | str]]] = {
        name: [] for name in policy_names
    }

    for episode in range(episodes):
        scenario_seed = base_seed + episode
        for name in policy_names:
            env = TutoringEnv(profile_split=split)
            observation, _ = env.reset(seed=scenario_seed)
            initial_mastery = float(env.mastery.mean())
            policy = _policy(name, agent, env)
            rng = np.random.default_rng(scenario_seed + 1_000_003)
            correct_count = 0
            total_reward = 0.0
            latencies_ms: list[float] = []
            done = False
            while not done:
                start = time.perf_counter()
                action = policy(observation, env.steps, rng)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                observation, reward, done, info = env.step(action)
                correct_count += int(info["correct"])
                total_reward += reward

            results[name].append(
                {
                    "scenario_seed": scenario_seed,
                    "policy": name,
                    "initial_mastery": initial_mastery,
                    "final_mastery": float(env.mastery.mean()),
                    "mastery_gain": float(env.mastery.mean()) - initial_mastery,
                    "topic_coverage": float(np.mean(env.mastery >= 0.7)),
                    "correct_rate": correct_count / env.max_steps,
                    "final_fatigue": float(env.fatigue),
                    "episode_reward": total_reward,
                    "steps": env.steps,
                    "mean_decision_latency_ms": float(np.mean(latencies_ms)),
                    "p95_decision_latency_ms": float(np.percentile(latencies_ms, 95)),
                }
            )
    return results


def summarize_results(
    results: dict[str, list[dict[str, float | int | str]]],
    bootstrap_samples: int = 2000,
    seed: int = 123,
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for policy, rows in results.items():
        summary[policy] = {}
        for metric in (
            "mastery_gain",
            "final_mastery",
            "topic_coverage",
            "correct_rate",
            "final_fatigue",
            "mean_decision_latency_ms",
            "p95_decision_latency_ms",
        ):
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True)
            means = samples.mean(axis=1)
            summary[policy][metric] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.percentile(means, 2.5)),
                "ci95_high": float(np.percentile(means, 97.5)),
            }
    return summary


def write_episode_csv(
    results: dict[str, list[dict[str, float | int | str]]], path: str | Path
) -> None:
    rows = [row for policy_rows in results.values() for row in policy_rows]
    if not rows:
        raise ValueError("cannot write an empty evaluation")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_preference_template(
    results: dict[str, list[dict[str, float | int | str]]],
    path: str | Path,
    seed: int = 123,
) -> None:
    ppo_by_seed = {int(row["scenario_seed"]): row for row in results["ppo"]}
    heuristic_by_seed = {
        int(row["scenario_seed"]): row for row in results["adaptive_heuristic"]
    }
    rng = np.random.default_rng(seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario_seed",
        "option_a",
        "option_b",
        "preferred_option",
        "comments",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for scenario_seed in sorted(ppo_by_seed):
            ppo_row = ppo_by_seed[scenario_seed]
            heuristic_row = heuristic_by_seed[scenario_seed]
            summaries = [
                (
                    "PPO",
                    f"mastery gain={float(ppo_row['mastery_gain']):.3f}; "
                    f"coverage={float(ppo_row['topic_coverage']):.2f}; "
                    f"correct={float(ppo_row['correct_rate']):.2f}; "
                    f"fatigue={float(ppo_row['final_fatigue']):.2f}",
                ),
                (
                    "Adaptive heuristic",
                    f"mastery gain={float(heuristic_row['mastery_gain']):.3f}; "
                    f"coverage={float(heuristic_row['topic_coverage']):.2f}; "
                    f"correct={float(heuristic_row['correct_rate']):.2f}; "
                    f"fatigue={float(heuristic_row['final_fatigue']):.2f}",
                ),
            ]
            if rng.integers(2):
                summaries.reverse()
            writer.writerow(
                {
                    "scenario_seed": scenario_seed,
                    "option_a": summaries[0][1],
                    "option_b": summaries[1][1],
                    "preferred_option": "",
                    "comments": "",
                }
            )
