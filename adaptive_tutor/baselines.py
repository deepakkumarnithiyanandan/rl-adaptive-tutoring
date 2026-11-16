"""Reference policies derived from logged OULAD resource-use behavior."""

from collections import Counter, defaultdict

import numpy as np

from adaptive_tutor.data import TransitionDataset


def fit_course_mode_policy(
    training: TransitionDataset, action_count: int
) -> tuple[dict[str, int], int]:
    counts_by_module: dict[str, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    for module, action in zip(training.modules, training.actions):
        counts_by_module[str(module)][int(action)] += 1
        global_counts[int(action)] += 1
    if not global_counts:
        raise ValueError("training split has no logged actions")
    fallback = max(range(action_count), key=lambda action: global_counts[action])
    modes = {
        module: max(range(action_count), key=lambda action: counts[action])
        for module, counts in counts_by_module.items()
    }
    return modes, fallback


def course_mode_actions(
    modules: np.ndarray, modes: dict[str, int], fallback: int
) -> np.ndarray:
    return np.asarray(
        [modes.get(str(module), fallback) for module in modules], dtype=np.int64
    )


def random_actions(count: int, action_count: int, seed: int) -> np.ndarray:
    if count < 0 or action_count < 1:
        raise ValueError("count must be non-negative and action_count must be positive")
    return np.random.default_rng(seed).integers(
        action_count, size=count, dtype=np.int64
    )
