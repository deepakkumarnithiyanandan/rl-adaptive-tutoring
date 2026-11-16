import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_tutor.cql import CQLAgent, CQLConfig
from adaptive_tutor.data import OULADData, TransitionDataset
from adaptive_tutor.evaluation import evaluate_offline_policies, write_evaluation


def _split(modules: list[str], actions: list[int]) -> TransitionDataset:
    states = np.zeros((len(actions), 10), dtype=np.float32)
    for row, module in enumerate(modules):
        states[row, 4 if module == "AAA" else 5] = 1.0
    return TransitionDataset(
        states=states,
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.linspace(-0.1, 0.1, len(actions), dtype=np.float32),
        next_states=states.copy(),
        dones=np.ones(len(actions), dtype=np.float32),
        episode_starts=np.ones(len(actions), dtype=bool),
        student_ids=np.asarray([f"student-{i}" for i in range(len(actions))]),
        modules=np.asarray(modules),
    )


class OfflineEvaluationTests(unittest.TestCase):
    def test_evaluates_diagnostics_and_writes_report(self) -> None:
        splits = {
            "train": _split(["AAA", "AAA", "BBB", "BBB"], [0, 0, 1, 1]),
            "validation": _split(["AAA", "BBB"], [0, 1]),
            "test": _split(["AAA", "BBB"], [0, 1]),
        }
        data = OULADData(
            splits=splits,
            action_names=("resource", "quiz"),
            module_names=("AAA", "BBB"),
            report={},
        )
        agent = CQLAgent(
            state_size=10,
            action_names=data.action_names,
            config=CQLConfig(hidden_size=8),
        )
        summary, rows = evaluate_offline_policies(
            agent, data, seed=3, bootstrap_samples=50, minimum_action_support=1
        )

        self.assertIn("cql_greedy", summary["policies"])
        self.assertEqual(summary["observed_test_reward_mean"], 0.0)
        self.assertEqual(summary["policies"]["course_mode"]["logged_action_match_rate"], 1.0)
        self.assertEqual(len(rows), 6)
        self.assertTrue(
            all(
                row["training_support_count"] >= 1
                for row in rows
                if row["policy"] == "course_mode"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            write_evaluation(summary, rows, directory)
            self.assertTrue((Path(directory) / "offline_evaluation.json").is_file())
            self.assertTrue((Path(directory) / "policy_estimates.csv").is_file())


if __name__ == "__main__":
    unittest.main()
