"""Unit tests for verify_phaseA_no_leakage.py helpers.

Tests hash invariants, mutation safety, and temp-copy isolation.
Does NOT run full model inference.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copytree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_phaseA_no_leakage import (
    canonical_hash,
    mutate_scoring_points,
    delete_record_csv,
)


class LeakageHashInvariantsTest(unittest.TestCase):
    """Tests for canonical_hash()."""

    def test_key_order_does_not_affect_hash(self):
        obj1 = {"b": 2, "a": 1}
        obj2 = {"a": 1, "b": 2}
        self.assertEqual(canonical_hash(obj1), canonical_hash(obj2))

    def test_different_content_produces_different_hash(self):
        obj1 = {"a": 1}
        obj2 = {"a": 2}
        self.assertNotEqual(canonical_hash(obj1), canonical_hash(obj2))

    def test_nested_objects_key_order_invariant(self):
        obj1 = {"b": {"y": 2, "x": 1}, "a": 3}
        obj2 = {"a": 3, "b": {"x": 1, "y": 2}}
        self.assertEqual(canonical_hash(obj1), canonical_hash(obj2))

    def test_list_order_is_stable(self):
        obj1 = [{"z": 1, "a": 2}, {"c": 3, "b": 4}]
        obj2 = [{"a": 2, "z": 1}, {"b": 4, "c": 3}]
        self.assertEqual(canonical_hash(obj1), canonical_hash(obj2))

    def test_json_roundtrip_key_order_invariant(self):
        raw1 = '{"x": {"nested": 1}, "y": [1, 2]}'
        raw2 = '{"y": [1, 2], "x": {"nested": 1}}'
        self.assertEqual(
            canonical_hash(json.loads(raw1)),
            canonical_hash(json.loads(raw2)),
        )


class LeakageMutationSafetyTest(unittest.TestCase):
    """Tests for mutate_scoring_points() and delete_record_csv()."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_query_csv(self, rows, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task_index", "instruction", "scoring_points"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _read_query_csv(self, path):
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def test_mutate_preserves_instruction_and_task_index(self):
        original = [
            {"task_index": "task_1", "instruction": "Find time",
             "scoring_points": "The only root cause occurrence time is within "
             "1 minutes (i.e., <=1min) of 2024-01-01 09:12:00"},
            {"task_index": "task_5", "instruction": "Find time and component",
             "scoring_points": "The 1-th root cause occurrence time is within "
             "1 minutes (i.e., <=1min) of 2024-01-01 09:12:00\nThe 1-th "
             "predicted root cause component is mysql01"},
        ]
        csv_path = self.tmpdir / "query.csv"
        self._write_query_csv(original, str(csv_path))

        mutate_scoring_points(str(csv_path))
        mutated = self._read_query_csv(str(csv_path))

        self.assertEqual(len(mutated), len(original))
        for orig, mut in zip(original, mutated):
            self.assertEqual(mut["task_index"], orig["task_index"])
            self.assertEqual(mut["instruction"], orig["instruction"])

    def test_mutate_scoring_points_are_different(self):
        original = [
            {"task_index": "task_3", "instruction": "Find component",
             "scoring_points": "The only predicted root cause component "
             "is mysql01"},
        ]
        csv_path = self.tmpdir / "query.csv"
        self._write_query_csv(original, str(csv_path))

        mutate_scoring_points(str(csv_path))
        mutated = self._read_query_csv(str(csv_path))

        self.assertNotEqual(
            mutated[0]["scoring_points"],
            original[0]["scoring_points"],
        )
        self.assertIn("MUTATED", mutated[0]["scoring_points"])

    def test_delete_record_csv_only_in_temp_copy(self):
        # Create a minimal original data area with record.csv
        original_area = self.tmpdir / "original"
        original_area.mkdir()
        record_path = original_area / "record.csv"
        record_path.write_text("timestamp,component,action\n"
                               "0,mysql01,deploy\n")

        # Copy to temp area
        copy_area = self.tmpdir / "copy"
        copytree(str(original_area), str(copy_area))

        # Assert original record.csv is still intact
        self.assertTrue((original_area / "record.csv").exists())

        # Delete in the copy only
        delete_record_csv(str(copy_area))

        # Original must still exist
        self.assertTrue((original_area / "record.csv").exists(),
                        "record.csv in original area must survive")
        # Copy must NOT have it
        self.assertFalse((copy_area / "record.csv").exists(),
                         "record.csv in temp copy must be deleted")

    def test_delete_record_csv_noop_when_file_missing(self):
        empty_dir = self.tmpdir / "no_record"
        empty_dir.mkdir()
        # Should not raise
        delete_record_csv(str(empty_dir))


if __name__ == "__main__":
    unittest.main()
