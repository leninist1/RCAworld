import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from foundation.evaluation.query_parser import parse_inference_queries


INSTRUCTION = (
    "On January 1, 2024, from 01:00 to 02:00, a single failure was detected. "
    "Please identify the occurrence time, component, and reason."
)


def write_query_csv(path: Path, fieldnames, row):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


class ParseInferenceQueriesGtIsolationTest(unittest.TestCase):
    def test_uses_only_visible_columns_when_scoring_points_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            write_query_csv(
                query_csv,
                ["task_index", "instruction", "scoring_points"],
                {
                    "task_index": "task_7",
                    "instruction": INSTRUCTION,
                    "scoring_points": "The predicted root cause component is secret_gt",
                },
            )

            original_read_csv = pd.read_csv
            calls = []
            loaded_columns = []

            def read_csv_spy(*args, **kwargs):
                calls.append(kwargs.get("usecols"))
                df = original_read_csv(*args, **kwargs)
                loaded_columns.append(list(df.columns))
                return df

            with patch(
                "foundation.evaluation.query_parser.pd.read_csv",
                side_effect=read_csv_spy,
            ):
                queries = parse_inference_queries(str(query_csv))

            self.assertEqual(calls, [["task_index", "instruction"]])
            self.assertEqual(loaded_columns, [["task_index", "instruction"]])
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].instruction, INSTRUCTION)

    def test_runs_when_scoring_points_column_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            write_query_csv(
                query_csv,
                ["task_index", "instruction"],
                {
                    "task_index": "task_7",
                    "instruction": INSTRUCTION,
                },
            )

            queries = parse_inference_queries(str(query_csv))

            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0].query_id, 0)
            self.assertEqual(queries[0].expected_fault_count, 1)


if __name__ == "__main__":
    unittest.main()
