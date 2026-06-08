import csv
import sys
import tempfile
import unittest
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
        queries = parse_inference_queries(query_csv_path)
        iq = queries[0]

        self.assertTrue(iq.need_component)
        self.assertFalse(iq.need_time)
        self.assertFalse(iq.need_reason)

        item = {"score": 0.95}
        if iq.need_time:
            item["datetime"] = "2024-01-01 09:12:00"
        if iq.need_component:
            item["component"] = "mysql01"

        self.assertIn("component", item)
        self.assertNotIn("datetime", item)
        self.assertNotIn("reason", item)

    def test_time_only_query_prediction_has_datetime_not_component(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time."
        )
        query_csv_path = self._write_query_csv("task_1", instruction)
        queries = parse_inference_queries(query_csv_path)
        iq = queries[0]

        self.assertTrue(iq.need_time)
        self.assertFalse(iq.need_component)
        self.assertFalse(iq.need_reason)

        item = {"score": 0.95}
        if iq.need_time:
            item["datetime"] = "2024-01-01 09:12:00"
        if iq.need_component:
            item["component"] = "mysql01"

        self.assertIn("datetime", item)
        self.assertNotIn("component", item)
        self.assertNotIn("reason", item)

    def test_time_component_query_prediction_has_both_not_reason(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time and component."
        )
        query_csv_path = self._write_query_csv("task_5", instruction)
        queries = parse_inference_queries(query_csv_path)
        iq = queries[0]

        self.assertTrue(iq.need_time)
        self.assertTrue(iq.need_component)
        self.assertFalse(iq.need_reason)

        item = {"score": 0.95}
        if iq.need_time:
            item["datetime"] = "2024-01-01 09:12:00"
        if iq.need_component:
            item["component"] = "mysql01"

        self.assertIn("datetime", item)
        self.assertIn("component", item)
        self.assertNotIn("reason", item)

    def test_query_level_result_contains_need_flags(self):
        instruction = (
            "On January 1, 2024, from 09:00 to 10:00, a single failure was detected. "
            "Please identify the occurrence time, component, and reason."
        )
        query_csv_path = self._write_query_csv("task_7", instruction)
        queries = parse_inference_queries(query_csv_path)
        iq = queries[0]

        self.assertTrue(iq.need_time)
        self.assertTrue(iq.need_component)
        self.assertTrue(iq.need_reason)

        entry = {
            "system": "Bank",
            "query_id": int(iq.query_id),
            "predictions": [],
            "component_ranking": [],
            "need_time": iq.need_time,
            "need_component": iq.need_component,
            "need_reason": iq.need_reason,
        }

        self.assertTrue(entry["need_time"])
        self.assertTrue(entry["need_component"])
        self.assertTrue(entry["need_reason"])
        self.assertNotIn("reason", entry.get("predictions", [{}])[0]
                         if entry.get("predictions") else {})


if __name__ == "__main__":
    unittest.main()
