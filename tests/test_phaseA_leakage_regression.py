"""Unit tests for verify_phaseA_no_leakage.py helpers.

Tests hash invariants, mutation safety, temp-copy isolation,
boundary error conditions, and prediction output validation.

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
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_phaseA_no_leakage import (
    canonical_hash,
    mutate_scoring_points,
    delete_record_csv,
    LeakageVerificationError,
    _load_model_and_state,
    _run_and_hash,
    _read_query_csv_ids,
    _validate_prediction_output,
    create_safe_temp_dataset_view,
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
        original_area = self.tmpdir / "original"
        original_area.mkdir()
        record_path = original_area / "record.csv"
        record_path.write_text("timestamp,component,action\n"
                               "0,mysql01,deploy\n")

        copy_area = self.tmpdir / "copy"
        copytree(str(original_area), str(copy_area))

        self.assertTrue((original_area / "record.csv").exists())

        delete_record_csv(str(copy_area))

        self.assertTrue((original_area / "record.csv").exists(),
                        "record.csv in original area must survive")
        self.assertFalse((copy_area / "record.csv").exists(),
                         "record.csv in temp copy must be deleted")

    def test_delete_record_csv_noop_when_file_missing(self):
        empty_dir = self.tmpdir / "no_record"
        empty_dir.mkdir()
        delete_record_csv(str(empty_dir))


class LeakageBoundaryTest(unittest.TestCase):
    """Tests for false-PASS boundary conditions."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_query_csv(self, data_dir, num_rows):
        """Write a minimal query.csv with num_rows data rows."""
        query_path = Path(data_dir) / "query.csv"
        with open(query_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task_index", "instruction", "scoring_points"])
            writer.writeheader()
            for i in range(num_rows):
                writer.writerow({
                    "task_index": f"task_{i + 1}",
                    "instruction": f"Instruction {i}",
                    "scoring_points": f"GT point {i}",
                })

    def test_load_model_raises_on_missing_checkpoint(self):
        with self.assertRaises(LeakageVerificationError) as ctx:
            _load_model_and_state(
                checkpoint_path="/nonexistent/checkpoint/best",
                data_path="/nonexistent/data.h5",
            )
        self.assertIn("Checkpoint not found", str(ctx.exception))

    def test_load_model_raises_on_missing_data_path(self):
        checkpoint_dir = self.tmpdir / "fake_ckpt"
        checkpoint_dir.mkdir()
        with self.assertRaises(LeakageVerificationError) as ctx:
            _load_model_and_state(
                checkpoint_path=str(checkpoint_dir),
                data_path="/nonexistent/data.h5",
            )
        self.assertIn("Data file not found", str(ctx.exception))

    # --- Output structure validation (unit-level, no run_inference mock) ---

    def test_validate_empty_list_raises(self):
        expected = {0, 1, 2}
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output([], expected)
        self.assertIn("zero predictions", str(ctx.exception))

    def test_validate_empty_predictions_per_query_raises(self):
        self._make_query_csv(self.tmpdir, 2)
        expected = _read_query_csv_ids(self.tmpdir)
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": []},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output(preds, expected)
        self.assertIn("predictions' list is empty", str(ctx.exception))

    def test_validate_missing_query_id_field_raises(self):
        self._make_query_csv(self.tmpdir, 1)
        expected = _read_query_csv_ids(self.tmpdir)
        preds = [{"predictions": [{"component": "a"}]}]
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output(preds, expected)
        self.assertIn("missing required field 'query_id'", str(ctx.exception))

    def test_validate_duplicate_query_id_raises(self):
        self._make_query_csv(self.tmpdir, 2)
        expected = _read_query_csv_ids(self.tmpdir)
        preds = [
            {"query_id": 0, "predictions": [{"component": "x"}]},
            {"query_id": 0, "predictions": [{"component": "y"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output(preds, expected)
        self.assertIn("Duplicate query_id", str(ctx.exception))

    def test_validate_missing_query_in_output_raises(self):
        self._make_query_csv(self.tmpdir, 3)
        expected = _read_query_csv_ids(self.tmpdir)
        # Only 2 of 3 queries in output
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": [{"component": "b"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output(preds, expected)
        self.assertIn("missing query_ids", str(ctx.exception))

    def test_validate_extra_query_in_output_raises(self):
        self._make_query_csv(self.tmpdir, 1)
        expected = _read_query_csv_ids(self.tmpdir)
        # Output has query_id=1 which is not expected
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": [{"component": "b"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            _validate_prediction_output(preds, expected)
        self.assertIn("extra query_ids", str(ctx.exception))

    def test_validate_valid_output_passes(self):
        self._make_query_csv(self.tmpdir, 2)
        expected = _read_query_csv_ids(self.tmpdir)
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": [
                {"component": "b"}, {"component": "c"},
            ]},
        ]
        # Should not raise
        _validate_prediction_output(preds, expected)

    # --- _run_and_hash integration tests (with mocked run_inference) ---

    def _run_and_hash_wrapper(self, mock_return, data_dir=None):
        """Helper to call _run_and_hash with mocked run_inference."""
        if data_dir is None:
            data_dir = self.tmpdir
        system = "Bank"
        output_path = self.tmpdir / "out.json"
        from verify_phaseA_no_leakage import SYSTEM_CONFIGS
        orig_cfg = dict(SYSTEM_CONFIGS)

        with patch("verify_phaseA_no_leakage.run_inference",
                   return_value=mock_return):
            h = _run_and_hash(
                system=system,
                temp_data_dir=str(data_dir),
                model=None, state=None,
                ob_mean=None, ob_std=None,
                method="calibrated", all_zero_type=False,
                burn_in_min=60, resample_sec=120,
                output_path=str(output_path),
            )
        self.assertEqual(SYSTEM_CONFIGS, orig_cfg,
                         "SYSTEM_CONFIGS must be restored")
        return h

    def test_run_and_hash_valid_output_succeeds(self):
        self._make_query_csv(self.tmpdir, 2)
        h = self._run_and_hash_wrapper([
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": [{"component": "b"}]},
        ])
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)
        self.assertTrue((self.tmpdir / "out.json").exists())

    def test_run_and_hash_empty_predictions_raises(self):
        preds = [
            {"query_id": 0, "predictions": []},
        ]
        self._make_query_csv(self.tmpdir, 1)
        with self.assertRaises(LeakageVerificationError) as ctx:
            self._run_and_hash_wrapper(preds)
        self.assertIn("predictions' list is empty", str(ctx.exception))

    def test_run_and_hash_output_missing_query_raises(self):
        """Output has 1 query but query.csv expects 2."""
        self._make_query_csv(self.tmpdir, 2)
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            self._run_and_hash_wrapper(preds)
        self.assertIn("missing query_ids", str(ctx.exception))

    def test_run_and_hash_duplicate_query_id_raises(self):
        self._make_query_csv(self.tmpdir, 2)
        preds = [
            {"query_id": 0, "predictions": [{"component": "x"}]},
            {"query_id": 0, "predictions": [{"component": "y"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            self._run_and_hash_wrapper(preds)
        self.assertIn("Duplicate query_id", str(ctx.exception))

    def test_run_and_hash_extra_query_id_raises(self):
        """query.csv has 1 row but output has query_id=1 too."""
        self._make_query_csv(self.tmpdir, 1)
        preds = [
            {"query_id": 0, "predictions": [{"component": "a"}]},
            {"query_id": 1, "predictions": [{"component": "b"}]},
        ]
        with self.assertRaises(LeakageVerificationError) as ctx:
            self._run_and_hash_wrapper(preds)
        self.assertIn("extra query_ids", str(ctx.exception))

    # --- _read_query_csv_ids unit tests ---

    def test_read_query_csv_ids_empty_csv_returns_empty_set(self):
        query_path = self.tmpdir / "query.csv"
        with open(query_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task_index", "instruction", "scoring_points"])
            writer.writeheader()
        ids = _read_query_csv_ids(str(self.tmpdir))
        self.assertEqual(ids, set())

    def test_read_query_csv_ids_three_rows_returns_0_1_2(self):
        self._make_query_csv(self.tmpdir, 3)
        ids = _read_query_csv_ids(str(self.tmpdir))
        self.assertEqual(ids, {0, 1, 2})

    # --- Safe temp dataset view tests ---

    def _make_orig_dataset(self, base_dir):
        """Create a minimal original dataset with query.csv and telemetry/."""
        orig = Path(base_dir) / "orig_data"
        orig.mkdir()
        query = orig / "query.csv"
        with open(query, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["task_index", "instruction", "scoring_points"])
            writer.writeheader()
            writer.writerow({"task_index": "task_1", "instruction": "Find",
                             "scoring_points": "GT"})
        tele = orig / "telemetry"
        tele.mkdir()
        (tele / "dummy.txt").write_text("big data")
        return orig

    def test_query_csv_temp_copy_mutation_does_not_affect_original(self):
        orig = self._make_orig_dataset(self.tmpdir)
        original_content = (orig / "query.csv").read_text()

        temp = self.tmpdir / "temp_view"
        create_safe_temp_dataset_view(str(orig), str(temp))

        # Modify temp copy
        (temp / "query.csv").write_text("MODIFIED CONTENT")

        # Original must be unchanged
        self.assertEqual((orig / "query.csv").read_text(), original_content)

    def test_query_csv_symlink_becomes_regular_file(self):
        orig = self._make_orig_dataset(self.tmpdir)
        # Replace query.csv with a symlink to a real file elsewhere
        real_content = "task_index,instruction,scoring_points\n"
        "task_1,Find,GT\n"
        real_file = self.tmpdir / "real_query.csv"
        real_file.write_text(real_content)
        (orig / "query.csv").unlink()
        (orig / "query.csv").symlink_to(real_file.resolve())

        temp = self.tmpdir / "temp_view"
        create_safe_temp_dataset_view(str(orig), str(temp))

        # Temp copy must be a regular file, not a symlink
        temp_query = temp / "query.csv"
        self.assertFalse(temp_query.is_symlink(),
                         "temp query.csv must be a regular file, not symlink")
        self.assertEqual(temp_query.read_text(), real_content)

        # Modify temp copy; original (via symlink target) must be unchanged
        temp_query.write_text("MODIFIED")
        self.assertEqual(real_file.read_text(), real_content)

    def test_record_csv_symlink_delete_preserves_original(self):
        orig = self._make_orig_dataset(self.tmpdir)
        real_content = "timestamp,component,action\n0,mysql01,deploy\n"
        real_file = self.tmpdir / "real_record.csv"
        real_file.write_text(real_content)
        rec = orig / "record.csv"
        rec.symlink_to(real_file.resolve())

        temp = self.tmpdir / "temp_view"
        create_safe_temp_dataset_view(str(orig), str(temp))

        # Temp copy must be a regular file
        temp_rec = temp / "record.csv"
        self.assertFalse(temp_rec.is_symlink(),
                         "temp record.csv must be a regular file")

        # Delete temp copy
        temp_rec.unlink()

        # Original (via symlink target) must still exist and be intact
        self.assertTrue(real_file.exists(),
                        "original record.csv must survive temp deletion")
        self.assertEqual(real_file.read_text(), real_content)

    def test_telemetry_temp_path_is_symlink_to_original(self):
        orig = self._make_orig_dataset(self.tmpdir)
        original_tele = orig / "telemetry"

        temp = self.tmpdir / "temp_view"
        create_safe_temp_dataset_view(str(orig), str(temp))

        temp_tele = temp / "telemetry"
        self.assertTrue(temp_tele.is_symlink(),
                        "temp telemetry must be a symlink")
        self.assertEqual(temp_tele.resolve(), original_tele.resolve())

    def test_telemetry_not_recursively_copied(self):
        """Telemetry contents must not be duplicated as real files in temp."""
        orig = self._make_orig_dataset(self.tmpdir)
        original_tele = orig / "telemetry"
        # Add a large-ish dummy file
        (original_tele / "big_file.bin").write_bytes(b"x" * 10000)

        temp = self.tmpdir / "temp_view"
        create_safe_temp_dataset_view(str(orig), str(temp))

        temp_tele = temp / "telemetry"
        self.assertTrue(temp_tele.is_symlink())

        # Verify no real file duplicate by checking that the big file's
        # content at temp path is the same INODE (symlink → original)
        big_orig = original_tele / "big_file.bin"
        big_temp = temp_tele / "big_file.bin"
        self.assertTrue(big_temp.exists(), "big file accessible via symlink")
        self.assertEqual(
            big_temp.resolve(), big_orig.resolve(),
            "file must point to original, not a duplicate")

    # --- New temp dir safety guard tests (6.2.1.4) ---

    def test_temp_dir_same_as_orig_dir_raises(self):
        orig = self._make_orig_dataset(self.tmpdir)
        with self.assertRaises(LeakageVerificationError) as ctx:
            create_safe_temp_dataset_view(str(orig), str(orig))
        self.assertIn("must not be the same", str(ctx.exception))

    def test_temp_dir_is_subdir_of_orig_raises(self):
        orig = self._make_orig_dataset(self.tmpdir)
        subdir = orig / "sub_view"
        with self.assertRaises(LeakageVerificationError) as ctx:
            create_safe_temp_dataset_view(str(orig), str(subdir))
        self.assertIn("must not be a subdirectory", str(ctx.exception))

    def test_temp_dir_exists_and_nonempty_raises(self):
        orig = self._make_orig_dataset(self.tmpdir)
        temp = self.tmpdir / "existing_nonempty"
        temp.mkdir()
        (temp / "some_file.txt").write_text("pre-existing content")
        with self.assertRaises(LeakageVerificationError) as ctx:
            create_safe_temp_dataset_view(str(orig), str(temp))
        self.assertIn("already exists and is non-empty", str(ctx.exception))

    def test_temp_dir_new_external_created_ok(self):
        orig = self._make_orig_dataset(self.tmpdir)
        temp = self.tmpdir / "brand_new_view"
        create_safe_temp_dataset_view(str(orig), str(temp))
        self.assertTrue(temp.exists())
        self.assertTrue((temp / "query.csv").exists())
        self.assertTrue((temp / "telemetry").is_symlink())

    def test_telemetry_missing_in_orig_raises(self):
        orig_dir = self.tmpdir / "orig_no_tele"
        orig_dir.mkdir()
        (orig_dir / "query.csv").write_text(
            "task_index,instruction,scoring_points\n"
            "task_1,Find,GT\n")
        temp = self.tmpdir / "temp_view"
        with self.assertRaises(LeakageVerificationError) as ctx:
            create_safe_temp_dataset_view(str(orig_dir), str(temp))
        self.assertIn("telemetry/ not found", str(ctx.exception))


class LeakageCUDAPortabilityTest(unittest.TestCase):
    """CUDA visibility portability tests (6.2.1.5)."""

    def test_import_does_not_overwrite_preset_cuda_visible_devices(self):
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            import importlib
            import scripts.verify_phaseA_no_leakage as _mod
            importlib.reload(_mod)
            self.assertEqual(
                os.environ["CUDA_VISIBLE_DEVICES"], "0",
                "import must not overwrite caller's CUDA_VISIBLE_DEVICES")
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def test_source_contains_no_cuda_visible_devices_assignment(self):
        script_path = REPO_ROOT / "scripts" / "verify_phaseA_no_leakage.py"
        source = script_path.read_text()
        self.assertNotIn(
            'os.environ["CUDA_VISIBLE_DEVICES"]',
            source,
            "script source must not contain "
            "os.environ[\"CUDA_VISIBLE_DEVICES\"] assignment")

    def test_run_inference_import_does_not_overwrite_cuda_visible_devices(self):
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            import importlib
            import scripts.phaseA_run_inference as _mod
            importlib.reload(_mod)
            self.assertEqual(
                os.environ["CUDA_VISIBLE_DEVICES"], "0",
                "import must not overwrite caller's CUDA_VISIBLE_DEVICES")
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def test_run_inference_source_contains_no_cuda_assignment(self):
        script_path = REPO_ROOT / "scripts" / "phaseA_run_inference.py"
        source = script_path.read_text()
        self.assertNotIn(
            'os.environ["CUDA_VISIBLE_DEVICES"]',
            source,
            "phaseA_run_inference.py must not contain "
            "os.environ[\"CUDA_VISIBLE_DEVICES\"] assignment")
        self.assertNotIn(
            "os.environ['CUDA_VISIBLE_DEVICES']",
            source,
            "phaseA_run_inference.py must not contain "
            "os.environ['CUDA_VISIBLE_DEVICES'] assignment")


class OnsetHeadInferenceTest(unittest.TestCase):
    """Onset head integration tests (Phase 0.9.1)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO_ROOT / "src"))

    def _make_test_data(self, T=100, N=5):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, (T, N)).astype(np.float32)
        residuals[50:60, 2] += 5.0
        latents = rng.normal(0, 1, (T, N, 64)).astype(np.float32)
        burn_in_mask = np.zeros(T, dtype=bool)
        burn_in_mask[:40] = True
        return residuals, latents, burn_in_mask

    # ── Test A: default lambda_onset=0.0 preserves existing behavior ──

    def test_a_default_lambda_zero_preserves_behavior(self):
        from foundation.evaluation.strict_eval import compute_joint_scores

        residuals, latents, burn_in_mask = self._make_test_data()
        onset_dummy = np.ones_like(residuals) * 100.0

        result_no_onset = compute_joint_scores(
            residuals=residuals, latents=latents,
            model_onset_scores=None, method="calibrated",
            burn_in_mask=burn_in_mask,
            lambda_onset=0.0,
        )
        result_with_onset_zero = compute_joint_scores(
            residuals=residuals, latents=latents,
            model_onset_scores=onset_dummy, method="calibrated",
            burn_in_mask=burn_in_mask,
            lambda_onset=0.0,
        )
        np.testing.assert_array_almost_equal(
            result_no_onset.S, result_with_onset_zero.S, decimal=6,
            err_msg="lambda_onset=0.0 must produce identical S "
            "regardless of onset input")

    # ── Test B: onset score actually affects S when lambda_onset > 0 ──

    def test_b_onset_score_affects_result(self):
        from foundation.evaluation.strict_eval import compute_joint_scores

        T, N = 50, 3
        rng = np.random.default_rng(1)
        residuals = rng.normal(0, 0.5, (T, N)).astype(np.float32)
        latents = rng.normal(0, 1, (T, N, 64)).astype(np.float32)
        burn_in_mask = np.zeros(T, dtype=bool)
        burn_in_mask[:15] = True

        onset = np.zeros((T, N), dtype=np.float32)
        onset[20, 1] = 10.0

        result_off = compute_joint_scores(
            residuals=residuals, latents=latents,
            model_onset_scores=onset, method="calibrated",
            burn_in_mask=burn_in_mask,
            lambda_onset=0.0,
        )
        result_on = compute_joint_scores(
            residuals=residuals, latents=latents,
            model_onset_scores=onset, method="calibrated",
            burn_in_mask=burn_in_mask,
            lambda_onset=1.0,
        )
        self.assertNotAlmostEqual(
            result_off.S[20, 1], result_on.S[20, 1], places=4,
            msg="lambda_onset=1.0 must change S at onset peak")
        self.assertGreater(
            result_on.S[20, 1], result_off.S[20, 1],
            msg="onset score must increase S at peak position")

        idx_off = np.argmax(result_off.S.reshape(-1))
        idx_on = np.argmax(result_on.S.reshape(-1))
        t_off, c_off = idx_off // N, idx_off % N
        t_on, c_on = idx_on // N, idx_on % N
        self.assertTrue(
            (c_on == 1 and t_on == 20) or (c_off != c_on),
            msg=f"onset should influence argmax: "
                f"off=({t_off},{c_off}) on=({t_on},{c_on})")

    # ── Test C: overlapping window aggregation average ──

    def test_c_overlapping_window_onset_averaging(self):
        import numpy as np
        from phaseA_run_inference import compute_residuals_and_latents, WS

        T, N, D = 30, 2, 16
        tensor = np.ones((T, N, D), dtype=np.float32)
        type_idx = np.zeros(N, dtype=np.int32)
        obs_mask = np.ones((T, N, D), dtype=np.float32)

        class _MockModel:
            def __init__(self):
                self._call_count = 0

            def apply(self, variables, x, type_idx, rng, obs_mask=None,
                      use_posterior=False):
                B = x.shape[0]
                ws_inner = x.shape[1] - 1
                Nx = x.shape[2]
                res = np.zeros((B, ws_inner, Nx), dtype=np.float32)
                h_seq = np.zeros((B, ws_inner, Nx, 256), dtype=np.float32)

                onset_val = float(self._call_count + 1) * 2.0
                self._call_count += 1
                onset = np.full((B, ws_inner, Nx), onset_val,
                                dtype=np.float32)

                return {
                    "residual": res,
                    "h_seq": h_seq,
                    "onset": {"onset_score": onset},
                }

        mock = _MockModel()
        resid, h_states, onset_acc, coverage = compute_residuals_and_latents(
            mock, None, tensor, type_idx, obs_mask, use_posterior=False,
            ws=23)

        covered = np.where(coverage)[0]
        self.assertGreater(len(covered), 0, "at least some timesteps covered")

        vals = [float(onset_acc[t, 0]) for t in covered[:5] if onset_acc[t, 0] > 0]
        self.assertGreater(len(vals), 0, "at least one timestep has onset score > 0")

        non_covered = np.where(~coverage)[0]
        if len(non_covered) > 0:
            for t in non_covered:
                self.assertEqual(float(onset_acc[t, 0]), 0.0,
                                 f"non-covered t={t} must have onset=0")

    # ── Test D: prior-only is enforced ──

    def test_d_prior_only_enforced(self):
        import inspect
        from phaseA_run_inference import run_inference, compute_residuals_and_latents

        run_source = inspect.getsource(run_inference)
        self.assertIn("use_posterior", run_source,
                      "run_inference must reference use_posterior")
        self.assertNotIn("use_posterior=True", run_source,
                         "run_inference must not hardcode use_posterior=True")

        cral_source = inspect.getsource(compute_residuals_and_latents)
        self.assertIn("use_posterior=use_posterior", cral_source,
                      "compute_residuals_and_latents must forward use_posterior")


if __name__ == "__main__":
    unittest.main()
