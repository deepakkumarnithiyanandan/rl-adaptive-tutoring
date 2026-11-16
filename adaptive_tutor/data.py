"""Download and turn the public OULAD archive into student-grouped transitions."""

import argparse
import bisect
import csv
import io
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open+university+learning+analytics+dataset.zip"
)
SPLITS = ("train", "validation", "test")
ASSESSMENT_TYPES = ("TMA", "CMA", "Exam")
STATE_FIXED_SIZE = 4
ASSESSMENT_STEP_SCALE = 10.0
CLICK_SCALE = float(np.log1p(10_000))
DAY_SCALE = 300.0

GroupKey = tuple[str, str, str]


@dataclass(frozen=True)
class TransitionDataset:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    episode_starts: np.ndarray
    student_ids: np.ndarray
    modules: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class OULADData:
    splits: dict[str, TransitionDataset]
    action_names: tuple[str, ...]
    module_names: tuple[str, ...]
    report: dict[str, Any]

    @property
    def state_size(self) -> int:
        return self.splits["train"].states.shape[1]


def download_archive(path: str | Path, force: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        with temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
    with zipfile.ZipFile(temporary) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Dataset archive contains a corrupt member: {corrupt_member}")
    temporary.replace(output)
    return output


def build_transitions(
    archive_path: str | Path,
    seed: int = 7,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> OULADData:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    with zipfile.ZipFile(archive_path) as archive:
        assessment_meta = _read_assessments(archive)
        student_scores, score_report = _read_student_scores(archive, assessment_meta)
        activity_by_site, activity_names = _read_activity_metadata(archive)

        interval_specs: dict[GroupKey, list[dict[str, Any]]] = {}
        for group, records in student_scores.items():
            ordered = sorted(
                records.values(),
                key=lambda record: (record["day"], record["assessment_id"]),
            )
            intervals: list[dict[str, Any]] = []
            for index in range(len(ordered) - 1):
                previous = ordered[index]
                following = ordered[index + 1]
                if following["day"] <= previous["day"]:
                    continue
                intervals.append(
                    {
                        "start": previous["day"],
                        "end": following["day"],
                        "previous_score": previous["score"],
                        "next_score": following["score"],
                        "previous_type": previous["type"],
                        "next_type": following["type"],
                        "step": index,
                    }
                )
            if intervals:
                interval_specs[group] = intervals

        student_ids = sorted({group[2] for group in interval_specs})
        split_by_student = _assign_student_splits(
            student_ids, train_fraction, validation_fraction, seed
        )
        interval_clicks: dict[tuple[GroupKey, int], Counter[str]] = defaultdict(Counter)
        interaction_report = _aggregate_activity(
            archive, interval_specs, activity_by_site, interval_clicks
        )

    module_names = tuple(
        sorted({group[0] for group in interval_specs})
    )
    module_index = {name: index for index, name in enumerate(module_names)}
    action_index = {name: index for index, name in enumerate(activity_names)}
    state_size = STATE_FIXED_SIZE + len(module_names) + 1 + len(ASSESSMENT_TYPES)
    split_rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    no_activity_intervals = 0

    for group, intervals in interval_specs.items():
        split = split_by_student[group[2]]
        cumulative_clicks = 0
        contiguous = False
        for index, interval in enumerate(intervals):
            counts = interval_clicks.get((group, index), Counter())
            if not counts:
                no_activity_intervals += 1
                contiguous = False
                continue

            action_name = min(counts, key=lambda name: (-counts[name], name))
            total_clicks = sum(counts.values())
            current_state = _state(
                score=interval["previous_score"],
                cumulative_clicks=cumulative_clicks,
                assessment_step=interval["step"],
                day=interval["start"],
                module=group[0],
                presentation=group[1],
                assessment_type=interval["previous_type"],
                module_index=module_index,
                module_names=module_names,
            )
            cumulative_clicks += total_clicks
            next_state = _state(
                score=interval["next_score"],
                cumulative_clicks=cumulative_clicks,
                assessment_step=interval["step"] + 1,
                day=interval["end"],
                module=group[0],
                presentation=group[1],
                assessment_type=interval["next_type"],
                module_index=module_index,
                module_names=module_names,
            )
            next_counts = (
                interval_clicks.get((group, index + 1), Counter())
                if index + 1 < len(intervals)
                else Counter()
            )
            done = not bool(next_counts)
            split_rows[split].append(
                {
                    "state": current_state,
                    "action": action_index[action_name],
                    "reward": (interval["next_score"] - interval["previous_score"])
                    / 100.0,
                    "next_state": next_state,
                    "done": done,
                    "episode_start": not contiguous,
                    "student_id": group[2],
                    "module": group[0],
                }
            )
            contiguous = not done

    splits: dict[str, TransitionDataset] = {}
    for split, rows in split_rows.items():
        splits[split] = _rows_to_dataset(rows, state_size)

    report = {
        "dataset": "Open University Learning Analytics Dataset (OULAD)",
        "source_url": "https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset",
        "dataset_doi": "10.24432/C5KK69",
        "license": "CC BY 4.0",
        "split_unit": "student id; no student appears in multiple splits",
        "state_features": [
            "previous assessment score",
            "log cumulative VLE clicks",
            "assessment sequence position",
            "course day",
            "course module one-hot",
            "presentation semester",
            "previous assessment type",
        ],
        "action_names": list(activity_names),
        "module_names": list(module_names),
        "assessment_types": list(ASSESSMENT_TYPES),
        "reward": "(next assessment score - previous assessment score) / 100",
        "score_quality": score_report,
        "interaction_quality": interaction_report,
        "intervals_without_activity": no_activity_intervals,
        "splits": {
            split: {
                "transitions": len(dataset),
                "students": int(np.unique(dataset.student_ids).size),
                "episodes": int(dataset.episode_starts.sum()),
            }
            for split, dataset in splits.items()
        },
        "limitations": [
            "This is observational learning-platform data, not a randomized tutor intervention.",
            "Resource-use and assessment-score associations are not causal effects.",
            "Different assessments may not be directly comparable despite score normalization.",
            "Demographic fields are deliberately excluded from policy states.",
        ],
    }
    return OULADData(splits, activity_names, module_names, report)


def save_cache(data: OULADData, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "metadata": np.asarray(
            json.dumps(
                {
                    "action_names": data.action_names,
                    "module_names": data.module_names,
                    "report": data.report,
                }
            )
        )
    }
    for split, dataset in data.splits.items():
        for field in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "episode_starts",
            "student_ids",
            "modules",
        ):
            payload[f"{split}_{field}"] = getattr(dataset, field)
    np.savez_compressed(output, **payload)
    return output


def load_cache(path: str | Path) -> OULADData:
    with np.load(Path(path), allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        splits = {
            split: TransitionDataset(
                **{
                    field: cache[f"{split}_{field}"]
                    for field in (
                        "states",
                        "actions",
                        "rewards",
                        "next_states",
                        "dones",
                        "episode_starts",
                        "student_ids",
                        "modules",
                    )
                }
            )
            for split in SPLITS
        }
    return OULADData(
        splits=splits,
        action_names=tuple(metadata["action_names"]),
        module_names=tuple(metadata["module_names"]),
        report=metadata["report"],
    )


def load_or_prepare(
    archive_path: str | Path,
    cache_path: str | Path,
    seed: int = 7,
) -> OULADData:
    cache = Path(cache_path)
    if cache.is_file():
        return load_cache(cache)
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(
            f"OULAD archive not found at {archive}. "
            "Run `python -m adaptive_tutor.data --download` first."
        )
    data = build_transitions(archive, seed=seed)
    save_cache(data, cache)
    report_path = cache.with_name("dataset_report.json")
    report_path.write_text(json.dumps(data.report, indent=2), encoding="utf-8")
    return data


def _read_assessments(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _csv_rows(archive, "assessments.csv"):
        result[row["id_assessment"]] = {
            "module": row["code_module"],
            "presentation": row["code_presentation"],
            "type": row["assessment_type"],
        }
    return result


def _read_student_scores(
    archive: zipfile.ZipFile, assessment_meta: dict[str, dict[str, Any]]
) -> tuple[dict[GroupKey, dict[str, dict[str, Any]]], dict[str, int]]:
    grouped: dict[GroupKey, dict[str, dict[str, Any]]] = defaultdict(dict)
    report = {
        "rows": 0,
        "banked_rows_excluded": 0,
        "missing_or_invalid_scores_excluded": 0,
        "unknown_assessments_excluded": 0,
        "duplicate_student_assessment_rows": 0,
    }
    for row in _csv_rows(archive, "studentAssessment.csv"):
        report["rows"] += 1
        if row["is_banked"] == "1":
            report["banked_rows_excluded"] += 1
            continue
        try:
            score = float(row["score"])
            day = int(row["date_submitted"])
        except (ValueError, TypeError):
            report["missing_or_invalid_scores_excluded"] += 1
            continue
        if not 0.0 <= score <= 100.0:
            report["missing_or_invalid_scores_excluded"] += 1
            continue
        metadata = assessment_meta.get(row["id_assessment"])
        if metadata is None:
            report["unknown_assessments_excluded"] += 1
            continue
        group = (metadata["module"], metadata["presentation"], row["id_student"])
        record = {
            "assessment_id": row["id_assessment"],
            "day": day,
            "score": score,
            "type": metadata["type"],
        }
        if row["id_assessment"] in grouped[group]:
            report["duplicate_student_assessment_rows"] += 1
            if day <= grouped[group][row["id_assessment"]]["day"]:
                continue
        grouped[group][row["id_assessment"]] = record
    return grouped, report


def _read_activity_metadata(
    archive: zipfile.ZipFile,
) -> tuple[dict[tuple[str, str, str], str], tuple[str, ...]]:
    activity_by_site: dict[tuple[str, str, str], str] = {}
    for row in _csv_rows(archive, "vle.csv"):
        activity_by_site[
            (row["code_module"], row["code_presentation"], row["id_site"])
        ] = row["activity_type"]
    return activity_by_site, tuple(sorted(set(activity_by_site.values())))


def _aggregate_activity(
    archive: zipfile.ZipFile,
    intervals: dict[GroupKey, list[dict[str, Any]]],
    activity_by_site: dict[tuple[str, str, str], str],
    interval_clicks: dict[tuple[GroupKey, int], Counter[str]],
) -> dict[str, int]:
    bounds = {
        group: (
            [item["start"] for item in group_intervals],
            [item["end"] for item in group_intervals],
        )
        for group, group_intervals in intervals.items()
    }
    report = {
        "student_vle_rows": 0,
        "rows_inside_scored_assessment_intervals": 0,
        "rows_with_known_activity_type": 0,
        "invalid_click_rows_excluded": 0,
        "unmatched_vle_metadata_rows": 0,
    }
    for row in _csv_rows(archive, "studentVle.csv"):
        report["student_vle_rows"] += 1
        group = (row["code_module"], row["code_presentation"], row["id_student"])
        group_bounds = bounds.get(group)
        if group_bounds is None:
            continue
        try:
            day = int(row["date"])
            clicks = int(row["sum_click"])
        except (ValueError, TypeError):
            report["invalid_click_rows_excluded"] += 1
            continue
        if clicks <= 0:
            report["invalid_click_rows_excluded"] += 1
            continue
        starts, ends = group_bounds
        index = bisect.bisect_left(ends, day)
        if index >= len(ends) or day <= starts[index]:
            continue
        report["rows_inside_scored_assessment_intervals"] += 1
        activity = activity_by_site.get(
            (row["code_module"], row["code_presentation"], row["id_site"])
        )
        if activity is None:
            report["unmatched_vle_metadata_rows"] += 1
            continue
        report["rows_with_known_activity_type"] += 1
        interval_clicks[(group, index)][activity] += clicks
    return report


def _assign_student_splits(
    student_ids: list[str],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    if len(student_ids) < 3:
        raise ValueError("At least three students with repeated assessments are required")
    shuffled = np.asarray(student_ids, dtype=str)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = min(max(1, int(len(shuffled) * train_fraction)), len(shuffled) - 2)
    validation_end = min(
        max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction))),
        len(shuffled) - 1,
    )
    assignments = {student: "train" for student in shuffled[:train_end]}
    assignments.update({student: "validation" for student in shuffled[train_end:validation_end]})
    assignments.update({student: "test" for student in shuffled[validation_end:]})
    return assignments


def _state(
    *,
    score: float,
    cumulative_clicks: int,
    assessment_step: int,
    day: int,
    module: str,
    presentation: str,
    assessment_type: str,
    module_index: dict[str, int],
    module_names: tuple[str, ...],
) -> np.ndarray:
    module_features = np.zeros(len(module_names), dtype=np.float32)
    module_features[module_index[module]] = 1.0
    assessment_features = np.zeros(len(ASSESSMENT_TYPES), dtype=np.float32)
    if assessment_type in ASSESSMENT_TYPES:
        assessment_features[ASSESSMENT_TYPES.index(assessment_type)] = 1.0
    fixed_features = np.asarray(
        (
            score / 100.0,
            min(float(np.log1p(cumulative_clicks) / CLICK_SCALE), 1.0),
            min(assessment_step / ASSESSMENT_STEP_SCALE, 1.0),
            min(max(day, 0) / DAY_SCALE, 1.0),
            float(presentation.endswith("J")),
        ),
        dtype=np.float32,
    )
    return np.concatenate(
        (fixed_features[:4], module_features, fixed_features[4:], assessment_features)
    )


def _rows_to_dataset(rows: list[dict[str, Any]], state_size: int) -> TransitionDataset:
    if not rows:
        return TransitionDataset(
            states=np.empty((0, state_size), dtype=np.float32),
            actions=np.empty(0, dtype=np.int64),
            rewards=np.empty(0, dtype=np.float32),
            next_states=np.empty((0, state_size), dtype=np.float32),
            dones=np.empty(0, dtype=np.float32),
            episode_starts=np.empty(0, dtype=bool),
            student_ids=np.empty(0, dtype=str),
            modules=np.empty(0, dtype=str),
        )
    return TransitionDataset(
        states=np.asarray([row["state"] for row in rows], dtype=np.float32),
        actions=np.asarray([row["action"] for row in rows], dtype=np.int64),
        rewards=np.asarray([row["reward"] for row in rows], dtype=np.float32),
        next_states=np.asarray([row["next_state"] for row in rows], dtype=np.float32),
        dones=np.asarray([row["done"] for row in rows], dtype=np.float32),
        episode_starts=np.asarray([row["episode_start"] for row in rows], dtype=bool),
        student_ids=np.asarray([row["student_id"] for row in rows], dtype=str),
        modules=np.asarray([row["module"] for row in rows], dtype=str),
    )


def _csv_rows(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        yield from csv.DictReader(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/oulad.zip")
    parser.add_argument("--output", default="artifacts/oulad_transitions.npz")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    archive = Path(args.archive)
    if args.download or args.force_download:
        download_archive(archive, force=args.force_download)
    data = build_transitions(archive, seed=args.seed)
    save_cache(data, args.output)
    report_path = Path(args.output).with_name("dataset_report.json")
    report_path.write_text(json.dumps(data.report, indent=2), encoding="utf-8")
    print(json.dumps(data.report["splits"], indent=2))
    print(f"Prepared transitions: {args.output}")


if __name__ == "__main__":
    main()
