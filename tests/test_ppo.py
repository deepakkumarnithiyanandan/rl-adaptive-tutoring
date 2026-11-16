import unittest

import numpy as np
import torch

from adaptive_tutor.ppo import PPOAgent, PPOConfig, generalized_advantages


class PPOTests(unittest.TestCase):
    def test_generalized_advantages_are_finite_and_include_rewards(self) -> None:
        advantages, returns = generalized_advantages(
            [1.0, 1.0], [0.5, 0.25], gamma=0.9, gae_lambda=0.95
        )
        self.assertEqual(advantages.shape, (2,))
        self.assertTrue(np.all(np.isfinite(advantages)))
        self.assertTrue(np.all(returns > 0.0))

    def test_agent_samples_updates_and_round_trips_checkpoint(self) -> None:
        torch.manual_seed(3)
        agent = PPOAgent(
            observation_size=5,
            action_count=3,
            config=PPOConfig(update_epochs=1, minibatch_size=4),
        )
        observations = np.zeros((4, 5), dtype=np.float32)
        samples = [agent.sample(observation) for observation in observations]
        actions = np.asarray([sample[0] for sample in samples], dtype=np.int64)
        log_probabilities = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        metrics = agent.update(
            observations,
            actions,
            log_probabilities,
            np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32),
            np.asarray([0.2, 0.1, 0.4, -0.1], dtype=np.float32),
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn(agent.act(observations[0]), range(3))

        with self.subTest("checkpoint"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.pt"
                agent.save(path)
                restored = PPOAgent.load(path)
                self.assertEqual(restored.act(observations[0]), agent.act(observations[0]))


if __name__ == "__main__":
    unittest.main()
