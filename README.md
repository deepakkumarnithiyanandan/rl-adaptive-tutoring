# Offline Adaptive Learning with Real Student Data

This project uses the **Open University Learning Analytics Dataset (OULAD)**,
. It trains a Conservative Q-Learning (CQL)
policy from historical student interactions to recommend a course activity
type (for example, `resource`, `quiz`, or `forumng`) before the next assessment.

The source is observational: students chose what to access, and their courses
were not randomized to tutor recommendations. The policy's value estimates are
therefore hypotheses for research, **not causal evidence** that a recommendation
improves learning.

## Dataset

- **Source:** [UCI Machine Learning Repository: OULAD](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)
- **DOI:** [10.24432/C5KK69](https://doi.org/10.24432/C5KK69)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Scale:** seven course modules, over 30,000 students, and about 10.7 million
  VLE interaction rows; the official archive is about 44.6 MB compressed.
- **Attribution:** Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017), “Open
  University Learning Analytics dataset,” *Scientific Data*, 4, 170171.
  [https://doi.org/10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171)

The pipeline uses `studentAssessment.csv`, `studentVle.csv`, `vle.csv`, and
`assessments.csv`. It excludes banked assessment records and rows with missing,
invalid, or out-of-range scores. It joins interactions to VLE activity metadata,
then builds transitions between consecutive scored assessments. The logged
action is the activity type with the most clicks in that interval; reward is
the normalized difference between the two assessment scores.

On the official archive, preparation excluded 1,909 banked results and 172
missing or invalid score rows. It produced 126,189 transitions from 21,170
students: 88,220 train, 18,858 validation, and 19,111 test transitions. These
counts are recorded in the generated `dataset_report.json`.

Only prior assessment results, engagement history, course module, presentation
semester, and prior assessment type enter the state. Demographic fields are
deliberately excluded. Splits are made by student ID before policy training so
no student appears in multiple splits.

Raw data, processed transitions, and model artifacts remain local and are
ignored by Git. The repository contains code and attribution, not OULAD files.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download and prepare the official dataset:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.data --download
```

The download command reuses `data\raw\oulad.zip` if it already exists. The
preparation step streams the archive and writes cached student-grouped
transitions and a data-quality report under `artifacts\`.

## Train and evaluate

Train CQL on the training students, selecting checkpoints by validation
Bellman error:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.train --updates 3000
```

Evaluate the selected policy on held-out students:

```powershell
.\.venv\Scripts\python.exe -m adaptive_tutor.evaluate
```

The evaluation reports:

- conservative-critic value estimates with student-clustered bootstrap
  intervals;
- the fraction of selected actions supported by training data for that course;
- agreement with each held-out student's logged activity choice; and
- descriptive score changes by logged activity category.

These are offline diagnostics, not observed outcomes under the learned policy.
Low action support warns that the critic is extrapolating. Even high support or
logged-action agreement does not remove confounding or establish causality.

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## RL formulation and limitations

- **State:** prior score, cumulative VLE clicks, course progress, course and
  assessment metadata.
- **Action:** the logged VLE activity type associated with the assessment
  interval; CQL learns a conservative action-value function over these actions.
- **Reward:** next score minus prior score, divided by 100.
- **Algorithm:** PyTorch CQL with a target network and a conservative
  log-sum-exp penalty to discourage unsupported actions.
- **Baselines:** random actions and the most common logged action per course.
- **Evaluation split:** held-out students, never random interaction rows.

Assessment scores differ in content and difficulty, so score differences are a
noisy reward. Activity use is self-selected and may reflect motivation,
teacher decisions, or course design. The critic's estimate should not be read
as an intervention effect. A real tutoring deployment would require
randomized evaluation, educator oversight, privacy safeguards, and student
consent.
