"""Simple policies used as reference points for the learned tutor."""

from collections.abc import Callable

import numpy as np

Policy = Callable[[np.ndarray, int, np.random.Generator], int]


def random_policy(
    observation: np.ndarray, step: int, rng: np.random.Generator, *, action_count: int
) -> int:
    del observation, step
    return int(rng.integers(action_count))


def round_robin_policy(
    observation: np.ndarray,
    step: int,
    rng: np.random.Generator,
    *,
    num_topics: int,
    difficulty_count: int,
    middle_difficulty_index: int,
) -> int:
    del observation, rng
    return (step % num_topics) * difficulty_count + middle_difficulty_index


def adaptive_heuristic_policy(
    observation: np.ndarray,
    step: int,
    rng: np.random.Generator,
    *,
    num_topics: int,
    difficulty_levels: np.ndarray,
) -> int:
    del step, rng
    mastery = observation[:num_topics]
    topic = int(np.argmin(mastery))
    target_difficulty = float(np.clip(mastery[topic] + 0.1, 0.25, 0.75))
    difficulty_index = int(np.argmin(np.abs(difficulty_levels - target_difficulty)))
    return topic * len(difficulty_levels) + difficulty_index
