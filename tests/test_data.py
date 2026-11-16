import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from adaptive_tutor.data import (
    _assign_student_splits,
    build_transitions,
    load_cache,
    save_cache,
)


def _write_csv(archive: zipfile.ZipFile, name: str, fields: list[str], rows: list[list[str]]) -> None:
    import io

    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(fields)
    writer.writerows(rows)
    archive.writestr(name, text.getvalue())


def _make_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        _write_csv(
            archive,
            "assessments.csv",
            ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
            [
                ["AAA", "2013J", "1", "TMA", "10", "50"],
                ["AAA", "2013J", "2", "Exam", "20", "50"],
            ],
        )
        _write_csv(
            archive,
            "vle.csv",
            ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"],
            [
                ["10", "AAA", "2013J", "resource", "1", "1"],
                ["11", "AAA", "2013J", "quiz", "1", "1"],
            ],
        )
        scores = []
        clicks = []
        for student in ("101", "102", "103", "104"):
            scores.extend(
                [
                    ["1", student, "9", "0", "50"],
                    ["2", student, "19", "0", "60"],
                ]
            )
            clicks.extend(
                [
                    ["AAA", "2013J", student, "10", "15", "3"],
                    ["AAA", "2013J", student, "11", "15", "1"],
                ]
            )
        scores.extend(
            [
                ["1", "101", "8", "1", "80"],
                ["1", "999", "9", "0", ""],
            ]
        )
        _write_csv(
            archive,
            "studentAssessment.csv",
            ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
            scores,
        )
        _write_csv(
            archive,
            "studentVle.csv",
            ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
            clicks,
        )


class OULADDataTests(unittest.TestCase):
    def test_three_students_populate_all_splits(self) -> None:
        assignments = _assign_student_splits(
            ["101", "102", "103"], 0.70, 0.15, seed=5
        )
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_builds_realistic_transitions_and_student_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "oulad.zip"
            _make_archive(archive_path)
            data = build_transitions(archive_path, seed=5)

            self.assertEqual(sum(len(split) for split in data.splits.values()), 4)
            self.assertEqual(data.action_names, ("quiz", "resource"))
            self.assertEqual(data.module_names, ("AAA",))
            self.assertEqual(data.state_size, 9)
            self.assertTrue(all(np.all(np.isfinite(s.states)) for s in data.splits.values()))
            self.assertEqual(data.report["score_quality"]["banked_rows_excluded"], 1)
            self.assertEqual(
                data.report["score_quality"]["missing_or_invalid_scores_excluded"], 1
            )

            student_sets = {
                name: set(split.student_ids.tolist())
                for name, split in data.splits.items()
            }
            self.assertFalse(student_sets["train"] & student_sets["validation"])
            self.assertFalse(student_sets["train"] & student_sets["test"])
            self.assertFalse(student_sets["validation"] & student_sets["test"])
            for split in data.splits.values():
                if len(split):
                    self.assertTrue(np.all(split.actions == 1))
                    np.testing.assert_allclose(split.rewards, 0.1)
                    self.assertTrue(np.all(split.dones))
                    self.assertTrue(np.all(split.episode_starts))

            cache_path = Path(directory) / "prepared.npz"
            save_cache(data, cache_path)
            cached = load_cache(cache_path)
            self.assertEqual(cached.action_names, data.action_names)
            for split in data.splits:
                np.testing.assert_array_equal(
                    cached.splits[split].actions, data.splits[split].actions
                )


if __name__ == "__main__":
    unittest.main()
