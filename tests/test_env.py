import unittest

import numpy as np

from adaptive_tutor.baselines import adaptive_heuristic_policy
from adaptive_tutor.env import LearnerProfile, TutoringEnv


class TutoringEnvTests(unittest.TestCase):
    def test_reset_and_step_obey_observation_and_episode_contract(self) -> None:
        env = TutoringEnv(max_steps=2)
        observation, _ = env.reset(seed=4)
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertTrue(np.all((observation >= 0.0) & (observation <= 1.0)))

        observation, reward, done, info = env.step(0)
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(info["correct"], bool)
        self.assertFalse(done)

        _, _, done, _ = env.step(1)
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            env.step(2)

    def test_action_and_split_validation(self) -> None:
        with self.assertRaises(ValueError):
            TutoringEnv(profile_split="unknown")
        env = TutoringEnv()
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(env.action_count)
        with self.assertRaises(ValueError):
            env.step(1.5)

    def test_seed_reproduces_test_learner_and_rule_action_is_valid(self) -> None:
        first = TutoringEnv(profile_split="test")
        second = TutoringEnv(profile_split="test")
        observation_a, profile_a = first.reset(seed=99)
        observation_b, profile_b = second.reset(seed=99)
        np.testing.assert_array_equal(observation_a, observation_b)
        self.assertEqual(profile_a, profile_b)

        action = adaptive_heuristic_policy(
            observation_a,
            0,
            np.random.default_rng(1),
            num_topics=first.num_topics,
            difficulty_levels=first.difficulty_levels,
        )
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, first.action_count)

    def test_step_changes_mastery_for_supplied_profile(self) -> None:
        env = TutoringEnv(max_steps=1)
        env.reset(seed=2)
        env.profile = LearnerProfile(
            initial_mastery=(0.2, 0.3, 0.4, 0.5),
            learning_rate=0.15,
            forgetting_rate=0.001,
            fatigue_sensitivity=1.0,
        )
        env.mastery = np.asarray(env.profile.initial_mastery, dtype=np.float32)
        before = float(env.mastery.mean())
        env.step(1)
        self.assertGreater(float(env.mastery.mean()), before)


if __name__ == "__main__":
    unittest.main()
