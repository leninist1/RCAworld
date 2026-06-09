import csv
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from foundation.evaluation.query_parser import (
    OPENRCA_TZ, EvalTarget, parse_inference_queries,
)
from foundation.evaluation.strict_eval import JointScores
from phaseA_evaluate_predictions import (
    match_predictions_to_targets, _build_eval_targets,
    _evaluate_prediction_target_pair, _aggregate_by_applicability,
)
from phaseA_run_inference import (
    serialize_prediction_item, build_prediction_entry,
)
from phaseA_export_openrca_csv import (
    build_prediction_payload, build_export_row, export_prediction_csv,
    ExportError, _validate_entry,
)


def _ts(dt_str):
    from datetime import datetime

    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=OPENRCA_TZ).timestamp()


class PhaseAMultiFaultMatchingTest(unittest.TestCase):
    def test_match_predictions_to_targets_is_order_independent(self):
        targets = [
            {
                "component": "mysql01",
                "component_idx": 0,
                "timestamp": _ts("2024-01-01 09:10:00"),
                "tolerance": 5,
            },
            {
                "component": "redis01",
                "component_idx": 1,
                "timestamp": _ts("2024-01-01 09:40:00"),
                "tolerance": 5,
            },
        ]
        predictions = [
            {
                "component": "redis01",
                "component_idx": 1,
                "timestamp": _ts("2024-01-01 09:39:00"),
            },
            {
                "component": "mysql01",
                "component_idx": 0,
                "timestamp": _ts("2024-01-01 09:12:00"),
            },
        ]

        matched_pairs, unmatched_predictions, unmatched_targets = match_predictions_to_targets(
            predictions, targets
        )

        self.assertEqual(len(matched_pairs), 2)
        self.assertEqual(unmatched_predictions, [])
        self.assertEqual(unmatched_targets, [])
        self.assertEqual(
            {
                (pair["prediction"]["component"], pair["target"]["component"])
                for pair in matched_pairs
            },
            {("mysql01", "mysql01"), ("redis01", "redis01")},
        )

    def test_decode_multiple_faults_respects_query_expected_fault_count(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, two failures were detected. "
            "Please identify the occurrence time, component, and reason."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            with query_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["task_index", "instruction"])
                writer.writeheader()
                writer.writerow({"task_index": "task_7", "instruction": instruction})

            queries = parse_inference_queries(str(query_csv))

        joint = JointScores(
            S=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=np.float32,
            ),
            residual=np.zeros((6, 3), dtype=np.float32),
            shift=np.zeros((6, 3), dtype=np.float32),
            early_rise=np.zeros((6, 3), dtype=np.float32),
            onset_raw=np.zeros((6, 3), dtype=np.float32),
            comp_raw=np.zeros(3, dtype=np.float32),
            query_window_mask=np.ones(6, dtype=bool),
        )

        predictions = joint.decode_multiple_faults(
            max_faults=queries[0].expected_fault_count,
            min_time_distance=1,
        )

        self.assertEqual(queries[0].expected_fault_count, 2)
        self.assertEqual(len(predictions), 2)


class PhaseAEvalTargetConstructionTest(unittest.TestCase):
    def test_build_eval_targets_not_influenced_by_record_csv_order(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            root_causes=[
                ("redis01", "2024-01-01 09:12:00", "network timeout", 5),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids)

        self.assertEqual(len(targets), 1)
        self.assertEqual(len(unresolved), 0)
        self.assertEqual(targets[0]["component"], "redis01")
        self.assertEqual(targets[0]["component_idx"], 1)
        expected_ts = _ts("2024-01-01 09:12:00")
        self.assertAlmostEqual(targets[0]["timestamp"], expected_ts, places=1)
        self.assertEqual(targets[0]["reason"], "network timeout")
        self.assertEqual(targets[0]["tolerance"], 5)

    def test_unresolved_component_goes_to_unresolved_not_silently_dropped(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            root_causes=[
                ("unknown_component_xyz", "2024-01-01 09:12:00",
                 "some reason", 5),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids)

        self.assertEqual(len(targets), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(
            unresolved[0],
            ("unknown_component_xyz", "2024-01-01 09:12:00",
             "some reason", 5),
        )

    def test_unresolved_bad_datetime_goes_to_unresolved(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            root_causes=[
                ("mysql01", "", "some reason", 5),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids)

        self.assertEqual(len(targets), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(
            unresolved[0],
            ("mysql01", "", "some reason", 5),
        )


class PhaseARealRankingTest(unittest.TestCase):
    def test_gt_component_rank_2_produces_correct_top1_top3_mrr(self):
        entity_ids = ["comp_a", "comp_b", "comp_c", "comp_d", "comp_e"]
        component_ranking = [
            ("comp_a", 0.95),
            ("comp_b", 0.82),
            ("comp_c", 0.41),
            ("comp_d", 0.12),
            ("comp_e", 0.05),
        ]
        target = {
            "component_idx": 1,
            "timestamp": _ts("2024-01-01 09:12:00"),
            "tolerance": 5,
        }
        prediction = {
            "component": "comp_b",
            "component_idx": 1,
            "timestamp": _ts("2024-01-01 09:14:00"),
        }

        metrics = _evaluate_prediction_target_pair(
            prediction, target, entity_ids,
            component_ranking=component_ranking,
            need_component=True, need_time=True,
        )

        self.assertFalse(metrics["component_top1"])
        self.assertTrue(metrics["component_top3"])
        self.assertAlmostEqual(metrics["reciprocal_rank"], 0.5, places=3)
        self.assertEqual(metrics["component_rank"], 2)

    def test_unresolved_root_cause_is_counted_as_miss_in_main_metrics(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            root_causes=[
                ("mysql01", "2024-01-01 09:12:00", "some reason", 5),
                ("unknown_comp", "2024-01-01 09:15:00", "other reason", 3),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids,
                                                    need_component=True, need_time=True)

        self.assertEqual(len(targets), 1)
        self.assertEqual(len(unresolved), 1)

        resolved_prediction = {
            "component": "mysql01",
            "component_idx": 0,
            "timestamp": _ts("2024-01-01 09:13:00"),
        }
        resolved_target = targets[0]
        metrics_resolved = _evaluate_prediction_target_pair(
            resolved_prediction, resolved_target, entity_ids,
            component_ranking=None, need_component=True, need_time=True,
        )
        self.assertTrue(metrics_resolved["component_top1"])
        self.assertTrue(metrics_resolved["joint_hit"])

        unresolved_metrics = _evaluate_prediction_target_pair(
            None, {"component_idx": -1, "timestamp": 0.0, "tolerance": 1},
            entity_ids, component_ranking=None,
            need_component=True, need_time=True,
        )
        unresolved_metrics["unresolved"] = True

        self.assertFalse(unresolved_metrics["component_top1"])
        self.assertEqual(unresolved_metrics["component_rank"], len(entity_ids))

    def test_gt_eval_targets_work_without_record_csv(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            root_causes=[
                ("redis01", "2024-01-01 09:12:00", "network timeout", 5),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids)

        self.assertEqual(len(targets), 1)
        self.assertEqual(len(unresolved), 0)
        self.assertEqual(targets[0]["component"], "redis01")

    def test_component_only_query_outputs_component_not_time_or_reason(self):
        entity_ids = ["mysql01", "redis01"]
        eval_target = EvalTarget(
            query_id=0,
            need_time=False,
            need_component=True,
            need_reason=False,
            root_causes=[
                ("redis01", "", "", 5),
            ],
        )

        targets, unresolved = _build_eval_targets(eval_target, entity_ids,
                                                    need_component=True, need_time=False)

        self.assertEqual(len(targets), 1)
        self.assertEqual(len(unresolved), 0)
        self.assertEqual(targets[0]["component"], "redis01")
        self.assertEqual(targets[0]["component_idx"], 1)
        self.assertEqual(targets[0]["timestamp"], 0.0)
        self.assertEqual(targets[0]["reason"], "")

        target = targets[0]
        prediction = {
            "component": "redis01",
            "component_idx": 1,
            "timestamp": None,
        }
        metrics = _evaluate_prediction_target_pair(
            prediction, target, entity_ids,
            component_ranking=None, need_component=True, need_time=False,
        )
        self.assertTrue(metrics["component_top1"])
        self.assertFalse(metrics["time_hit"])

    def test_reason_only_query_marked_unsupported(self):
        eval_target = EvalTarget(
            query_id=0,
            need_time=False,
            need_component=False,
            need_reason=True,
            root_causes=[
                ("", "", "some reason", 1),
            ],
        )

        self.assertTrue(eval_target.need_reason)
        self.assertFalse(eval_target.need_component)
        self.assertFalse(eval_target.need_time)


class CrossSystemQueryIdIsolationTest(unittest.TestCase):
    def test_same_query_id_across_systems_does_not_overwrite(self):
        bank_entry = {
            "system": "Bank",
            "query_id": 0,
            "predictions": [
                {"datetime": "2024-01-01 09:12:00", "component": "mysql01",
                 "score": 0.95},
            ],
            "component_ranking": [("mysql01", 0.95), ("redis01", 0.42)],
        }
        telecom_entry = {
            "system": "Telecom",
            "query_id": 0,
            "predictions": [
                {"datetime": "2024-01-01 10:05:00", "component": "router01",
                 "score": 0.88},
            ],
            "component_ranking": [("router01", 0.88), ("switch01", 0.31)],
        }

        all_predictions = [bank_entry, telecom_entry]
        predictions_by_key = {(p["system"], p["query_id"]): p
                              for p in all_predictions}

        self.assertEqual(len(predictions_by_key), 2)

        bank_key = ("Bank", 0)
        telecom_key = ("Telecom", 0)
        self.assertIn(bank_key, predictions_by_key)
        self.assertIn(telecom_key, predictions_by_key)

        bank_pred = predictions_by_key.get(bank_key,
                                            {"query_id": 0, "predictions": []})
        telecom_pred = predictions_by_key.get(telecom_key,
                                               {"query_id": 0, "predictions": []})

        self.assertEqual(bank_pred["predictions"][0]["component"], "mysql01")
        self.assertEqual(telecom_pred["predictions"][0]["component"], "router01")

    def test_old_predictions_without_system_field_are_detected(self):
        old_entry = {
            "query_id": 0,
            "predictions": [
                {"datetime": "2024-01-01 09:12:00", "component": "mysql01",
                 "score": 0.95},
            ],
        }
        missing_system = [p for p in [old_entry] if "system" not in p]
        self.assertEqual(len(missing_system), 1)


class PhaseAApplicabilityFilteringTest(unittest.TestCase):
    def test_all_unresolved_query_counts_unresolved_once_not_double(self):
        query_results = [
            {"component_rank": 4, "component_top1": False, "component_top3": False,
             "reciprocal_rank": 0.25, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": True,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
            {"component_rank": 4, "component_top1": False, "component_top3": False,
             "reciprocal_rank": 0.25, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": True,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
            {"component_rank": 4, "component_top1": False, "component_top3": False,
             "reciprocal_rank": 0.25, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": True,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertEqual(agg["n"], 3)
        self.assertEqual(agg["n_resolved"], 0)
        self.assertEqual(agg["component_top1"], 0.0)
        self.assertEqual(agg["component_metric_count"], 3)

    def test_time_only_query_not_in_component_metric_count(self):
        query_results = [
            {"component_rank": 1, "component_top1": True, "component_top3": True,
             "reciprocal_rank": 1.0, "component_applicable": False,
             "time_applicable": True, "joint_applicable": False, "unresolved": False,
             "time_error_min": 2.5, "time_hit": True,
             "time_hit_5min": True, "time_hit_10min": True, "time_hit_15min": True,
             "joint_hit": False},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertEqual(agg["component_metric_count"], 0)
        self.assertEqual(agg["time_metric_count"], 1)
        self.assertEqual(agg["joint_metric_count"], 0)

    def test_component_only_query_not_in_time_metric_count(self):
        query_results = [
            {"component_rank": 1, "component_top1": True, "component_top3": True,
             "reciprocal_rank": 1.0, "component_applicable": True,
             "time_applicable": False, "joint_applicable": False, "unresolved": False,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertEqual(agg["component_metric_count"], 1)
        self.assertEqual(agg["time_metric_count"], 0)
        self.assertEqual(agg["joint_metric_count"], 0)

    def test_time_component_query_in_joint_metric_count(self):
        query_results = [
            {"component_rank": 1, "component_top1": True, "component_top3": True,
             "reciprocal_rank": 1.0, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": False,
             "time_error_min": 1.5, "time_hit": True,
             "time_hit_5min": True, "time_hit_10min": True, "time_hit_15min": True,
             "joint_hit": True},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertEqual(agg["component_metric_count"], 1)
        self.assertEqual(agg["time_metric_count"], 1)
        self.assertEqual(agg["joint_metric_count"], 1)

    def test_component_only_query_does_not_cause_time_mae_nan(self):
        query_results = [
            {"component_rank": 1, "component_top1": True, "component_top3": True,
             "reciprocal_rank": 1.0, "component_applicable": True,
             "time_applicable": False, "joint_applicable": False, "unresolved": False,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertEqual(agg["time_mae_min"], 0.0)
        self.assertFalse(np.isnan(agg["time_mae_min"]))
        self.assertEqual(agg["time_metric_count"], 0)

    def test_unresolved_need_component_miss_in_main_component_metrics(self):
        query_results = [
            {"component_rank": 4, "component_top1": False, "component_top3": False,
             "reciprocal_rank": 0.25, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": True,
             "time_error_min": float('nan'), "time_hit": False,
             "time_hit_5min": False, "time_hit_10min": False, "time_hit_15min": False,
             "joint_hit": False},
            {"component_rank": 1, "component_top1": True, "component_top3": True,
             "reciprocal_rank": 1.0, "component_applicable": True,
             "time_applicable": True, "joint_applicable": True, "unresolved": False,
             "time_error_min": 1.0, "time_hit": True,
             "time_hit_5min": True, "time_hit_10min": True, "time_hit_15min": True,
             "joint_hit": True},
        ]
        agg = _aggregate_by_applicability(query_results)
        self.assertLess(agg["component_top1"], 1.0)
        self.assertEqual(agg["component_metric_count"], 2)
        self.assertEqual(agg["n_resolved"], 1)
        self.assertEqual(agg["resolved_component_top1"], 1.0)


class PhaseAJointHitSemanticsTest(unittest.TestCase):
    def test_wrong_selected_component_with_gt_in_top3_gives_joint_hit_false(self):
        entity_ids = ["redis01", "mysql01", "nginx01"]
        component_ranking = [
            ("redis01", 0.95),
            ("mysql01", 0.82),
            ("nginx01", 0.41),
        ]
        target = {
            "component": "mysql01",
            "component_idx": 1,
            "timestamp": _ts("2024-01-01 09:12:00"),
            "tolerance": 5,
        }
        prediction = {
            "component": "redis01",
            "component_idx": 0,
            "timestamp": _ts("2024-01-01 09:10:00"),
        }

        metrics = _evaluate_prediction_target_pair(
            prediction, target, entity_ids,
            component_ranking=component_ranking,
            need_component=True, need_time=True,
        )

        self.assertFalse(metrics["component_top1"])
        self.assertTrue(metrics["component_top3"])
        self.assertAlmostEqual(metrics["reciprocal_rank"], 0.5, places=3)
        self.assertEqual(metrics["component_rank"], 2)
        self.assertFalse(metrics["joint_hit"])
        self.assertFalse(metrics["joint_component_hit"])
        self.assertTrue(metrics["joint_time_hit"])

    def test_unresolved_gt_small_entity_set_no_false_top3_or_nonzero_mrr(self):
        entity_ids = ["mysql01", "redis01"]
        target = {
            "component_idx": -1,
            "timestamp": 0.0,
            "tolerance": 1,
        }

        metrics = _evaluate_prediction_target_pair(
            None, target, entity_ids,
            component_ranking=None,
            need_component=True, need_time=False,
        )

        self.assertFalse(metrics["component_top1"])
        self.assertFalse(metrics["component_top3"])
        self.assertAlmostEqual(metrics["reciprocal_rank"], 0.0, places=3)
        self.assertFalse(metrics["joint_hit"])


class PhaseAQueryMaskSerializationTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_query_csv(self, task, instruction):
        query_csv = self.tmpdir / "query.csv"
        with query_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_index", "instruction"])
            writer.writeheader()
            writer.writerow({"task_index": task, "instruction": instruction})
        return str(query_csv)

    def test_component_only_query_prediction_has_component_not_datetime(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the component."
        )
        query_csv_path = self._write_query_csv("task_3", instruction)
        iq = parse_inference_queries(query_csv_path)[0]

        self.assertTrue(iq.need_component)
        self.assertFalse(iq.need_time)
        self.assertFalse(iq.need_reason)

        item = serialize_prediction_item(
            iq, dt="2024-01-01 09:12:00", component="mysql01", score=0.95)
        item_json = json.loads(json.dumps(item))

        self.assertIn("component", item_json)
        self.assertNotIn("datetime", item_json)
        self.assertNotIn("reason", item_json)
        self.assertEqual(item_json["component"], "mysql01")
        self.assertIn("score", item_json)

    def test_time_only_query_prediction_has_datetime_not_component(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time."
        )
        query_csv_path = self._write_query_csv("task_1", instruction)
        iq = parse_inference_queries(query_csv_path)[0]

        self.assertTrue(iq.need_time)
        self.assertFalse(iq.need_component)
        self.assertFalse(iq.need_reason)

        item = serialize_prediction_item(
            iq, dt="2024-01-01 09:12:00", component="mysql01", score=0.95)
        item_json = json.loads(json.dumps(item))

        self.assertIn("datetime", item_json)
        self.assertNotIn("component", item_json)
        self.assertNotIn("reason", item_json)
        self.assertEqual(item_json["datetime"], "2024-01-01 09:12:00")
        self.assertIn("score", item_json)

    def test_time_component_query_prediction_has_both_not_reason(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time and component."
        )
        query_csv_path = self._write_query_csv("task_5", instruction)
        iq = parse_inference_queries(query_csv_path)[0]

        self.assertTrue(iq.need_time)
        self.assertTrue(iq.need_component)
        self.assertFalse(iq.need_reason)

        item = serialize_prediction_item(
            iq, dt="2024-01-01 09:12:00", component="mysql01", score=0.95)
        item_json = json.loads(json.dumps(item))

        self.assertIn("datetime", item_json)
        self.assertIn("component", item_json)
        self.assertNotIn("reason", item_json)
        self.assertEqual(item_json["datetime"], "2024-01-01 09:12:00")
        self.assertEqual(item_json["component"], "mysql01")
        self.assertIn("score", item_json)

    def test_query_level_result_contains_need_flags(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time, component, and reason."
        )
        query_csv_path = self._write_query_csv("task_7", instruction)
        iq = parse_inference_queries(query_csv_path)[0]

        self.assertTrue(iq.need_time)
        self.assertTrue(iq.need_component)
        self.assertTrue(iq.need_reason)

        preds = [serialize_prediction_item(
            iq, dt="2024-01-01 09:12:00", component="mysql01", score=0.95)]
        entry = build_prediction_entry(
            system="Bank", query_id=iq.query_id,
            predictions=preds, component_ranking=[], iq=iq)
        entry_json = json.loads(json.dumps(entry))

        self.assertTrue(entry_json["need_time"])
        self.assertTrue(entry_json["need_component"])
        self.assertTrue(entry_json["need_reason"])
        self.assertEqual(entry_json["system"], "Bank")
        self.assertEqual(entry_json["query_id"], iq.query_id)
        self.assertIsInstance(entry_json["predictions"], list)
        self.assertIsInstance(entry_json["component_ranking"], list)


class PhaseAOpenRCAExportTest(unittest.TestCase):
    """Tests for OpenRCA-format prediction.csv export.

    Each query is one CSV row with `row_id` and `prediction` columns.
    The `prediction` cell is a JSON string with numbered root causes:
        {"1": {"root cause component": "..."}, "2": {...}}
    """

    def test_component_only_query_row_has_component_in_prediction_json(self):
        entry = {
            "system": "Bank",
            "query_id": 0,
            "need_time": False,
            "need_component": True,
            "need_reason": False,
            "predictions": [
                {"score": 0.95, "component": "mysql01"},
            ],
        }
        payload = build_prediction_payload(entry)
        self.assertEqual(len(payload), 1)
        self.assertIn("1", payload)
        self.assertEqual(payload["1"]["root cause component"], "mysql01")
        self.assertNotIn("root cause occurrence datetime", payload["1"])

    def test_time_only_query_row_has_datetime_in_prediction_json(self):
        entry = {
            "system": "Telecom",
            "query_id": 1,
            "need_time": True,
            "need_component": False,
            "need_reason": False,
            "predictions": [
                {"score": 0.88, "datetime": "2024-01-01 09:12:00"},
            ],
        }
        payload = build_prediction_payload(entry)
        self.assertEqual(len(payload), 1)
        self.assertEqual(
            payload["1"]["root cause occurrence datetime"],
            "2024-01-01 09:12:00")
        self.assertNotIn("root cause component", payload["1"])

    def test_time_component_query_has_both_fields(self):
        entry = {
            "system": "Bank",
            "query_id": 2,
            "need_time": True,
            "need_component": True,
            "need_reason": False,
            "predictions": [
                {"score": 0.95, "datetime": "2024-01-01 09:12:00",
                 "component": "mysql01"},
            ],
        }
        payload = build_prediction_payload(entry)
        self.assertEqual(
            payload["1"]["root cause occurrence datetime"],
            "2024-01-01 09:12:00")
        self.assertEqual(
            payload["1"]["root cause component"], "mysql01")

    def test_multifault_time_component_single_row_sorted_by_datetime(self):
        entry = {
            "system": "Bank",
            "query_id": 3,
            "need_time": True,
            "need_component": True,
            "need_reason": False,
            "predictions": [
                {"score": 0.85, "datetime": "2024-01-01 09:40:00",
                 "component": "redis01"},
                {"score": 0.95, "datetime": "2024-01-01 09:12:00",
                 "component": "mysql01"},
            ],
        }
        payload = build_prediction_payload(entry)
        self.assertEqual(len(payload), 2)
        self.assertIn("1", payload)
        self.assertIn("2", payload)
        self.assertEqual(
            payload["1"]["root cause component"], "mysql01")
        self.assertEqual(
            payload["1"]["root cause occurrence datetime"],
            "2024-01-01 09:12:00")
        self.assertEqual(
            payload["2"]["root cause component"], "redis01")
        self.assertEqual(
            payload["2"]["root cause occurrence datetime"],
            "2024-01-01 09:40:00")

    def test_reason_only_query_raises_export_error_not_silent_skip(self):
        entry = {
            "system": "Bank",
            "query_id": 4,
            "need_time": False,
            "need_component": False,
            "need_reason": True,
            "predictions": [
                {"score": 0.90},
            ],
        }
        with self.assertRaises(ExportError) as ctx:
            build_prediction_payload(entry)
        self.assertIn("not supported", str(ctx.exception))

    def test_mixed_systems_raises_export_error(self):
        entries = [
            {"system": "Bank",  "query_id": 0, "need_time": False,
             "need_component": True, "need_reason": False,
             "predictions": [{"score": 0.9, "component": "a"}]},
            {"system": "Market", "query_id": 1, "need_time": False,
             "need_component": True, "need_reason": False,
             "predictions": [{"score": 0.9, "component": "b"}]},
        ]
        with self.assertRaises(ExportError) as ctx:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                             delete=False) as f:
                export_prediction_csv(entries, f.name)
        self.assertIn("Multiple systems", str(ctx.exception))

    def test_missing_required_field_raises_export_error(self):
        entry = {
            "system": "Bank",
            "query_id": 5,
            "need_time": False,
            "need_component": True,
            "need_reason": False,
            "predictions": [
                {"score": 0.95},
            ],
        }
        with self.assertRaises(ExportError) as ctx:
            build_prediction_payload(entry)
        self.assertIn("component", str(ctx.exception))

    def test_old_format_missing_system_raises_error(self):
        entry = {
            "query_id": 6,
            "need_time": False,
            "need_component": True,
            "need_reason": False,
            "predictions": [],
        }
        with self.assertRaises(ExportError):
            _validate_entry(entry, 0)

    def test_schema_csv_has_prediction_column_one_row_per_query(self):
        entries = [
            {"system": "Bank", "query_id": 0, "need_time": False,
             "need_component": True, "need_reason": False,
             "predictions": [{"score": 0.95, "component": "mysql01"}]},
            {"system": "Bank", "query_id": 1, "need_time": True,
             "need_component": False, "need_reason": False,
             "predictions": [{"score": 0.88, "datetime": "2024-01-01 09:12:00"}]},
            {"system": "Bank", "query_id": 2, "need_time": True,
             "need_component": True, "need_reason": False,
             "predictions": [
                 {"score": 0.85, "datetime": "2024-01-01 09:40:00",
                  "component": "redis01"},
                 {"score": 0.95, "datetime": "2024-01-01 09:12:00",
                  "component": "mysql01"},
             ]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            output_path = f.name
        try:
            written = export_prediction_csv(entries, output_path)
            self.assertEqual(written, len(entries))

            with open(output_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), len(entries),
                             "CSV row count must equal number of query entries")

            for row in rows:
                self.assertIn("prediction", row)
                payload = json.loads(row["prediction"])
                self.assertIsInstance(payload, dict)
                for key in payload:
                    self.assertRegex(key, r"^\d+$")

            self.assertEqual(json.loads(rows[2]["prediction"]),
                             OrderedDict([
                                 ("1", OrderedDict([
                                     ("root cause occurrence datetime",
                                      "2024-01-01 09:12:00"),
                                     ("root cause component", "mysql01"),
                                 ])),
                                 ("2", OrderedDict([
                                     ("root cause occurrence datetime",
                                      "2024-01-01 09:40:00"),
                                     ("root cause component", "redis01"),
                                 ])),
                             ]))
        finally:
            os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()
