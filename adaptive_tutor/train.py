"""Train Conservative Q-Learning on OULAD's logged student transitions."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from adaptive_tutor.cql import CQLAgent, CQLConfig
from adaptive_tutor.data import load_or_prepare


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data = load_or_prepare(args.archive, args.cache, seed=args.seed)
    training = data.splits["train"]
    validation = data.splits["validation"]
    if len(training) == 0 or len(validation) == 0:
        raise ValueError("prepared OULAD data must have non-empty train and validation splits")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(
        json.dumps(data.report, indent=2), encoding="utf-8"
    )
    agent = CQLAgent(
        data.state_size,
        data.action_names,
        CQLConfig(learning_rate=args.learning_rate, conservative_weight=args.cql_weight),
    )
    rng = np.random.default_rng(args.seed)
    training_rows: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        indices = rng.integers(0, len(training), size=args.batch_size)
        metrics = agent.update(
            training.states[indices],
            training.actions[indices],
            training.rewards[indices],
            training.next_states[indices],
            training.dones[indices],
        )
        training_rows.append({"update": update, **metrics})

        if update % args.eval_interval == 0 or update == args.updates:
            validation_loss = agent.validation_td_loss(
                validation.states,
                validation.actions,
                validation.rewards,
                validation.next_states,
                validation.dones,
            )
            agent.save(output_dir / "latest_policy.pt")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                agent.save(output_dir / "best_policy.pt")
            print(
                f"update={update:5d} "
                f"train_bellman={metrics['bellman_loss']:.4f} "
                f"validation_td={validation_loss:.4f} "
                f"best={best_validation_loss:.4f}"
            )

    _write_rows(output_dir / "training.csv", training_rows)
    print(f"Training completed in {time.perf_counter() - started:.1f}s")
    print(f"Selected checkpoint: {output_dir / 'best_policy.pt'}")


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/oulad.zip")
    parser.add_argument("--cache", default="artifacts/oulad_transitions.npz")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cql-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.eval_interval < 1:
        parser.error("updates, batch-size, and eval-interval must be positive")
    train(args)


if __name__ == "__main__":
    main()
