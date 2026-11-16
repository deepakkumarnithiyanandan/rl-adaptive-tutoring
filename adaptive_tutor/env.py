"""Synthetic, fully local learner simulator for adaptive tutoring."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LearnerProfile:
    initial_mastery: tuple[float, ...]
    learning_rate: float
    forgetting_rate: float
    fatigue_sensitivity: float


class TutoringEnv:
    """A small episodic environment; no real learner data is used."""

    topic_names = ("algebra", "fractions", "geometry", "statistics")
    difficulty_levels = np.asarray((0.25, 0.50, 0.75), dtype=np.float32)

    def __init__(self, profile_split: str = "train", max_steps: int = 20) -> None:
        if profile_split not in {"train", "validation", "test"}:
            raise ValueError("profile_split must be train, validation, or test")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.profile_split = profile_split
        self.max_steps = max_steps
        self.num_topics = len(self.topic_names)
        self.action_count = self.num_topics * len(self.difficulty_levels)
        self.observation_size = 2 * self.num_topics + 3
        self.rng = np.random.default_rng()
        self.profile: LearnerProfile | None = None
        self.mastery = np.zeros(self.num_topics, dtype=np.float32)
        self.initial_mastery = self.mastery.copy()
        self.fatigue = 0.0
        self.last_topic = -1
        self.last_correct = 0.5
        self.steps = 0
        self._episode_done = True

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.profile = self._sample_profile()
        self.mastery = np.asarray(self.profile.initial_mastery, dtype=np.float32)
        self.initial_mastery = self.mastery.copy()
        self.fatigue = 0.0
        self.last_topic = -1
        self.last_correct = 0.5
        self.steps = 0
        self._episode_done = False
        return self._observation(), {"profile": asdict(self.profile)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("reset must be called before stepping a completed episode")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_count:
            raise ValueError(f"action must be an integer in [0, {self.action_count})")

        assert self.profile is not None
        topic = int(action) // len(self.difficulty_levels)
        difficulty_index = int(action) % len(self.difficulty_levels)
        difficulty = float(self.difficulty_levels[difficulty_index])
        before_mean = float(self.mastery.mean())
        current_mastery = float(self.mastery[topic])

        logit = (
            8.0 * (current_mastery - difficulty)
            + 1.5 * (self.profile.learning_rate - 0.09)
            - 0.8 * self.fatigue
        )
        probability_correct = float(1.0 / (1.0 + np.exp(-np.clip(logit, -12.0, 12.0))))
        correct = bool(self.rng.random() < probability_correct)

        difficulty_match = max(0.35, 1.0 - abs(difficulty - current_mastery))
        response_factor = 0.45 + 0.55 * float(correct)
        fatigue_factor = max(0.35, 1.0 - 0.45 * self.fatigue)
        gain = self.profile.learning_rate * (1.0 - current_mastery)
        gain *= difficulty_match * response_factor * fatigue_factor

        forgetting = 0.1 * self.profile.forgetting_rate
        self.mastery = np.maximum(0.0, self.mastery - forgetting)
        self.mastery[topic] = min(1.0, current_mastery + gain)

        fatigue_increase = 0.0
        if not correct and difficulty > current_mastery + 0.1:
            fatigue_increase += 0.07 * self.profile.fatigue_sensitivity
        if topic == self.last_topic:
            fatigue_increase += 0.025
        elif correct:
            fatigue_increase -= 0.02
        self.fatigue = float(np.clip(self.fatigue + fatigue_increase, 0.0, 1.0))

        self.last_topic = topic
        self.last_correct = float(correct)
        self.steps += 1
        mastery_change = float(self.mastery.mean()) - before_mean
        reward = (
            4.0 * mastery_change
            + 0.05 * float(correct)
            - 0.12 * max(0.0, fatigue_increase)
            - 0.01
        )
        done = self.steps >= self.max_steps
        self._episode_done = done

        info = {
            "topic": self.topic_names[topic],
            "difficulty": difficulty,
            "correct": correct,
            "probability_correct": probability_correct,
            "mastery_change": mastery_change,
            "fatigue": self.fatigue,
        }
        return self._observation(), float(reward), done, info

    def _sample_profile(self) -> LearnerProfile:
        if self.profile_split == "test":
            mastery_range = (0.05, 0.82)
            learning_range = (0.025, 0.12)
            forgetting_range = (0.002, 0.012)
            fatigue_range = (0.9, 1.6)
        else:
            mastery_range = (0.15, 0.65)
            learning_range = (0.06, 0.16)
            forgetting_range = (0.001, 0.006)
            fatigue_range = (0.7, 1.3)

        initial = tuple(
            float(value)
            for value in self.rng.uniform(*mastery_range, size=self.num_topics)
        )
        return LearnerProfile(
            initial_mastery=initial,
            learning_rate=float(self.rng.uniform(*learning_range)),
            forgetting_rate=float(self.rng.uniform(*forgetting_range)),
            fatigue_sensitivity=float(self.rng.uniform(*fatigue_range)),
        )

    def _observation(self) -> np.ndarray:
        previous_topic = np.zeros(self.num_topics, dtype=np.float32)
        if self.last_topic >= 0:
            previous_topic[self.last_topic] = 1.0
        progress = self.steps / self.max_steps
        return np.concatenate(
            (
                self.mastery,
                previous_topic,
                np.asarray((self.fatigue, self.last_correct, progress), dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
