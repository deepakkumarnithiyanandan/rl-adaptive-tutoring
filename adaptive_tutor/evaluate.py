"""Evaluate the selected PPO checkpoint against simple tutoring baselines."""

import argparse
import json
from pathlib import Path

import torch

from adaptive_tutor.evaluation import (
    evaluate_policies,
    summarize_results,
    write_episode_csv,
    write_preference_template,
)
from adaptive_tutor.ppo import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/best_policy.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--preference-template")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    torch.set_num_threads(1)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}; train a policy first")
    agent = PPOAgent.load(checkpoint)
    names = ["ppo", "adaptive_heuristic", "round_robin", "random"]
    results = evaluate_policies(
        names,
        episodes=args.episodes,
        split="test",
        base_seed=args.seed,
        agent=agent,
    )
    summary = summarize_results(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_episode_csv(results, output_dir / "test_episodes.csv")
    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "evaluation_split": "test",
                "episodes_per_policy": args.episodes,
                "summary": summary,
            },
            file,
            indent=2,
        )
    if args.preference_template:
        write_preference_template(results, args.preference_template, seed=args.seed)

    for name in names:
        score = summary[name]["mastery_gain"]
        latency = summary[name]["mean_decision_latency_ms"]["mean"]
        print(
            f"{name:18s} mastery_gain={score['mean']:.3f} "
            f"95% CI=[{score['ci95_low']:.3f}, {score['ci95_high']:.3f}] "
            f"latency={latency:.4f}ms/decision"
        )
    print(f"Episode results: {output_dir / 'test_episodes.csv'}")
    print(f"Aggregate summary: {output_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
