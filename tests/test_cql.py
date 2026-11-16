import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_tutor.cql import CQLAgent, CQLConfig


class CQLTests(unittest.TestCase):
    def test_offline_update_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(2)
        agent = CQLAgent(
            state_size=5,
            action_names=("resource", "quiz", "forum"),
            config=CQLConfig(hidden_size=8),
        )
        states = np.zeros((8, 5), dtype=np.float32)
        actions = np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
        rewards = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
        dones = np.asarray([0, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32)
        metrics = agent.update(states, actions, rewards, states, dones)

        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        actions_selected = agent.act(states)
        self.assertEqual(actions_selected.shape, (8,))
        self.assertTrue(np.all((actions_selected >= 0) & (actions_selected < 3)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            agent.save(checkpoint)
            restored = CQLAgent.load(checkpoint)
            np.testing.assert_allclose(restored.q_values(states), agent.q_values(states))


if __name__ == "__main__":
    unittest.main()
