# Adaptive Tutoring with Reinforcement Learning

A standalone, reproducible reinforcement-learning project that trains a tutor to
choose **what topic to teach and at what difficulty** for a simulated learner.
It is a non-voice project and does not use real student records.

## What it demonstrates

- A custom episodic environment with observable learner mastery, fatigue, and
  recent tutoring context.
- A PyTorch PPO actor-critic with generalized advantage estimation, clipped
  policy updates, entropy regularization, and validation-based checkpoint
  selection.
- Three comparable policies: random, fixed-medium round robin, and an adaptive
  weakest-topic rule.
- Fixed validation scenarios for checkpoint selection and a separate,
  distribution-shifted test population for final evaluation.
- Per-episode metrics, bootstrap 95% confidence intervals, and inference
  latency measurements.
- An optional blind preference-review CSV template. It is an annotation
  workflow only; no human preferences are fabricated or used to claim learning.

## The RL task

Each episode is a short tutoring session across four topics. At each step the
agent selects one topic and one of three difficulty levels. The observation
contains topic mastery estimates, the previous topic and answer outcome,
fatigue, and session progress. The simulator samples whether the learner answers
correctly, updates mastery and fatigue, and applies small forgetting to
untouched knowledge.

The reward combines measured mastery change, a modest correctness signal, and
fatigue cost. This makes reward design inspectable rather than hiding the
objective inside a black-box score. Final mastery gain, topic coverage,
correctness, fatigue, and wall-clock inference latency are reported separately
so reward alone is not treated as learning quality.

The synthetic learner model is deliberately an approximation, not an educational
assessment instrument. Test learners have different parameter ranges from
training learners, but robust results in this simulator do **not** establish
effectiveness with real students. Any real deployment would require educator
review, consent, privacy safeguards, and controlled evaluation.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If your Python launcher uses a different version, replace `py -3.12` with your
installed Python command.

## Run

Train PPO and select the best checkpoint using fixed validation scenarios:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.train --updates 200
```

Evaluate the selected checkpoint and all baselines on held-out, shifted learner
profiles:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.evaluate --episodes 100
```

Optionally export a blind side-by-side review template for PPO versus the
adaptive heuristic:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.evaluate --episodes 100 --preference-template artifacts\preference_review.csv
```

Run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Training writes `artifacts/best_policy.pt`, `artifacts/latest_policy.pt`,
`artifacts/training.csv`, and `artifacts/validation.csv`. Evaluation writes
episode-level results and aggregate metrics under `artifacts/`.

## Reference run

One CPU run with seed 7, 200 PPO updates, and 100 shifted test scenarios
produced the following simulated mean mastery gains (bootstrap 95% CIs):

| Policy | Mastery gain | Correct rate | Final fatigue |
| --- | ---: | ---: | ---: |
| PPO | 0.105 [0.096, 0.115] | 0.773 | 0.037 |
| Adaptive heuristic | 0.090 [0.083, 0.097] | 0.227 | 0.835 |
| Fixed-medium round robin | 0.077 [0.070, 0.084] | 0.451 | 0.446 |
| Random | 0.072 [0.065, 0.079] | 0.460 | 0.567 |

These are simulator measurements, not real-student outcomes. The per-policy
intervals are not a paired significance test; the result should be treated as
evidence for further investigation, not proof of educational effectiveness.

## Reading the evaluation

Compare policies on the same held-out learner seeds. Use mastery gain and
coverage as primary outcome measures; inspect correctness and fatigue to expose
trade-offs. The bootstrap intervals summarize variation across simulated
episodes and are not confidence intervals for real-world student outcomes.
Latency is a local simulation measurement, not a production service SLO.
