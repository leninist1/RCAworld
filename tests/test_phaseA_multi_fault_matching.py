import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from foundation.evaluation.query_parser import OPENRCA_TZ, parse_inference_queries
from foundation.evaluation.strict_eval import JointScores
from phaseA_evaluate_predictions import match_predictions_to_targets


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


if __name__ == "__main__":
    unittest.main()
