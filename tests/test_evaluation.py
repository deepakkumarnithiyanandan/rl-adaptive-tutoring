import csv
import tempfile
import unittest
from pathlib import Path

from adaptive_tutor.evaluation import (
    evaluate_policies,
    summarize_results,
    write_preference_template,
)


class EvaluationTests(unittest.TestCase):
    def test_matched_scenarios_summary_and_preference_template(self) -> None:
        results = evaluate_policies(
            ["adaptive_heuristic", "round_robin"],
            episodes=3,
            split="test",
            base_seed=400,
        )
        heuristic_seeds = [row["scenario_seed"] for row in results["adaptive_heuristic"]]
        round_robin_seeds = [row["scenario_seed"] for row in results["round_robin"]]
        self.assertEqual(heuristic_seeds, round_robin_seeds)

        summary = summarize_results(results, bootstrap_samples=100)
        interval = summary["adaptive_heuristic"]["mastery_gain"]
        self.assertLessEqual(interval["ci95_low"], interval["mean"])
        self.assertGreaterEqual(interval["ci95_high"], interval["mean"])

        preference_results = evaluate_policies(
            ["ppo", "adaptive_heuristic"],
            episodes=2,
            split="test",
            base_seed=500,
            agent=_FixedPolicy(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_preference_template(preference_results, path)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["preferred_option"], "")
            self.assertNotIn("PPO", rows[0]["option_a"])
            self.assertNotIn("Adaptive heuristic", rows[0]["option_b"])


class _FixedPolicy:
    def act(self, observation, deterministic=True):
        del observation, deterministic
        return 0


if __name__ == "__main__":
    unittest.main()
