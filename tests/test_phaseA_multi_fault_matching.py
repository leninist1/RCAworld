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
    _evaluate_prediction_target_pair,
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


if __name__ == "__main__":
    unittest.main()
