"""Held-out-student evaluation for offline policies."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_tutor.baselines import course_mode_actions, fit_course_mode_policy, random_actions
from adaptive_tutor.cql import CQLAgent
from adaptive_tutor.data import OULADData, TransitionDataset


def evaluate_offline_policies(
    agent: CQLAgent,
    data: OULADData,
    seed: int = 123,
    bootstrap_samples: int = 2000,
    minimum_action_support: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = data.splits["train"]
    test = data.splits["test"]
    if len(training) == 0 or len(test) == 0:
        raise ValueError("training and test splits must both contain transitions")
    if agent.action_names != data.action_names:
        raise ValueError("checkpoint action vocabulary does not match prepared data")

    q_values = agent.q_values(test.states)
    course_modes, fallback = fit_course_mode_policy(training, agent.action_count)
    actions_by_policy = {
        "cql_greedy": np.argmax(q_values, axis=1).astype(np.int64),
        "course_mode": course_mode_actions(test.modules, course_modes, fallback),
        "random": random_actions(len(test), agent.action_count, seed),
    }
    support_counts = _action_support_by_module(training, data.module_names, agent.action_count)
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {
        "evaluation": "held-out students; critic-estimated values and logged-action support",
        "episodes_or_students": int(np.unique(test.student_ids).size),
        "test_transitions": len(test),
        "observed_test_reward_mean": _cluster_mean(test.rewards, test.student_ids),
        "policies": {},
        "observed_reward_by_logged_activity": _logged_reward_by_action(
            test, data.action_names
        ),
        "caveats": [
            "The policy value is a conservative Q-network estimate, not an observed counterfactual outcome.",
            "OULAD is observational; student-selected resources and scores are confounded.",
            "Low action support makes critic estimates unreliable; no causal improvement claim is made.",
        ],
    }
    rows: list[dict[str, Any]] = []
    for policy_name, actions in actions_by_policy.items():
        indices = np.arange(len(test))
        estimated_values = q_values[indices, actions]
        module_indices = {name: index for index, name in enumerate(data.module_names)}
        support = np.asarray(
            [
                support_counts[module_indices[str(module)], int(action)]
                for module, action in zip(test.modules, actions)
            ],
            dtype=np.int64,
        )
        policy_summary = {
            "critic_estimated_value": _cluster_bootstrap(
                estimated_values, test.student_ids, rng, bootstrap_samples
            ),
            "logged_action_match_rate": float(np.mean(actions == test.actions)),
            "supported_action_rate": float(np.mean(support >= minimum_action_support)),
            "mean_training_support_per_selected_action": float(support.mean()),
            "minimum_action_support": minimum_action_support,
        }
        summary["policies"][policy_name] = policy_summary
        for index, action in enumerate(actions):
            rows.append(
                {
                    "policy": policy_name,
                    "module": str(test.modules[index]),
                    "recommended_activity": data.action_names[int(action)],
                    "logged_activity": data.action_names[int(test.actions[index])],
                    "logged_action_match": bool(action == test.actions[index]),
                    "training_support_count": int(support[index]),
                    "critic_estimated_value": float(estimated_values[index]),
                    "observed_logged_score_delta": float(test.rewards[index]),
                }
            )
    return summary, rows


def write_evaluation(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "offline_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    if not rows:
        raise ValueError("cannot write evaluation with no test rows")
    with (output / "policy_estimates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _action_support_by_module(
    training: TransitionDataset,
    module_names: tuple[str, ...],
    action_count: int,
) -> np.ndarray:
    module_index = {name: index for index, name in enumerate(module_names)}
    counts = np.zeros((len(module_names), action_count), dtype=np.int64)
    for module, action in zip(training.modules, training.actions):
        counts[module_index[str(module)], int(action)] += 1
    return counts


def _cluster_mean(values: np.ndarray, student_ids: np.ndarray) -> float:
    means = _per_student_means(values, student_ids)
    return float(means.mean()) if len(means) else 0.0


def _cluster_bootstrap(
    values: np.ndarray,
    student_ids: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    means = _per_student_means(values, student_ids)
    draws = rng.choice(means, size=(samples, len(means)), replace=True).mean(axis=1)
    return {
        "mean_per_student": float(means.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def _per_student_means(values: np.ndarray, student_ids: np.ndarray) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for student, value in zip(student_ids, values):
        grouped.setdefault(str(student), []).append(float(value))
    return np.asarray([np.mean(group) for group in grouped.values()], dtype=np.float64)


def _logged_reward_by_action(
    test: TransitionDataset, action_names: tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for action, name in enumerate(action_names):
        mask = test.actions == action
        if np.any(mask):
            result[name] = {
                "transitions": int(mask.sum()),
                "mean_observed_score_delta": float(test.rewards[mask].mean()),
            }
    return result
