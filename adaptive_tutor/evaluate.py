"""Evaluate a CQL policy on students held out from training."""

import argparse
import json
from pathlib import Path

import torch

from adaptive_tutor.cql import CQLAgent
from adaptive_tutor.data import load_or_prepare
from adaptive_tutor.evaluation import evaluate_offline_policies, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/best_policy.pt")
    parser.add_argument("--archive", default="data/raw/oulad.zip")
    parser.add_argument("--cache", default="artifacts/oulad_transitions.npz")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}; train a policy first")
    torch.set_num_threads(1)
    data = load_or_prepare(args.archive, args.cache)
    agent = CQLAgent.load(checkpoint_path)
    summary, rows = evaluate_offline_policies(agent, data, seed=args.seed)
    write_evaluation(summary, rows, args.output_dir)
    for name, metrics in summary["policies"].items():
        value = metrics["critic_estimated_value"]
        print(
            f"{name:14s} estimated_Q={value['mean_per_student']:.4f} "
            f"95% CI=[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}] "
            f"action_support={metrics['supported_action_rate']:.1%} "
            f"logged_match={metrics['logged_action_match_rate']:.1%}"
        )
    print(f"Observed held-out score delta: {summary['observed_test_reward_mean']:.4f}")
    print(f"Evaluation report: {Path(args.output_dir) / 'offline_evaluation.json'}")


if __name__ == "__main__":
    main()
