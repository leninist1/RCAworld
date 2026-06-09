import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foundation.evaluation.query_parser import (
    load_eval_targets,
    parse_fault_count,
    parse_inference_queries,
    parse_observation_window,
    QueryParseError,
)


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
                    "scoring_points": (
                        "The 1-th predicted root cause component is secret_gt_1\n"
                        "The 2-th predicted root cause component is secret_gt_2\n"
                        "The 3-th predicted root cause component is secret_gt_3"
                    ),
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
            self.assertEqual(queries[0].expected_fault_count, 1)

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

    def test_parse_fault_count_from_instruction_text(self):
        self.assertEqual(parse_fault_count("number of failures: 3"), 3)
        self.assertEqual(parse_fault_count("a single failure"), 1)
        self.assertEqual(parse_fault_count("unknown text"), 1)

    def test_load_eval_targets_parses_each_root_cause_time(self):
        scoring_points = "\n".join([
            "The 1-th root cause occurrence time is within 1 minutes (i.e., <=1min) of 2024-01-01 01:10:00",
            "The 1-th predicted root cause component is service_a",
            "The 1-th predicted root cause reason is cpu saturation",
            "The 2-th root cause occurrence time is within 2 minutes (i.e., <=2min) of 2024-01-01 01:20:00",
            "The 2-th predicted root cause component is service_b",
            "The 2-th predicted root cause reason is disk latency",
            "The 3-th root cause occurrence time is within 3 minutes (i.e., <=3min) of 2024-01-01 01:30:00",
            "The 3-th predicted root cause component is service_c",
            "The 3-th predicted root cause reason is network loss",
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            write_query_csv(
                query_csv,
                ["task_index", "instruction", "scoring_points"],
                {
                    "task_index": "task_7",
                    "instruction": INSTRUCTION,
                    "scoring_points": scoring_points,
                },
            )

            eval_targets = load_eval_targets(str(query_csv))

            root_causes = eval_targets[0].root_causes
            self.assertEqual(len(root_causes), 3)
            self.assertEqual(
                [rc[1] for rc in root_causes],
                [
                    "2024-01-01 01:10:00",
                    "2024-01-01 01:20:00",
                    "2024-01-01 01:30:00",
                ],
            )
            self.assertTrue(all(rc[1] for rc in root_causes))
            self.assertEqual(
                root_causes,
                [
                    ("service_a", "2024-01-01 01:10:00", "cpu saturation", 1),
                    ("service_b", "2024-01-01 01:20:00", "disk latency", 2),
                    ("service_c", "2024-01-01 01:30:00", "network loss", 3),
                ],
            )

    def test_load_eval_targets_parses_official_only_root_cause(self):
        scoring_points = "\n".join([
            "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2024-01-01 01:10:00",
            "The only predicted root cause component is service_a",
            "The only predicted root cause reason is cpu saturation",
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            write_query_csv(
                query_csv,
                ["task_index", "instruction", "scoring_points"],
                {
                    "task_index": "task_7",
                    "instruction": INSTRUCTION,
                    "scoring_points": scoring_points,
                },
            )

            eval_targets = load_eval_targets(str(query_csv))

            self.assertEqual(
                eval_targets[0].root_causes,
                [("service_a", "2024-01-01 01:10:00", "cpu saturation", 1)],
            )


class ParseObservationWindowTest(unittest.TestCase):
    """Tests for robust observation window parsing (6.2.1.6b)."""

    def test_standard_from_to_format(self):
        inst = "On March 4, 2021, from 14:30 to 15:00, a single failure was detected."
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.year, 2021)
        self.assertEqual(ws.month, 3)
        self.assertEqual(ws.day, 4)
        self.assertEqual(ws.hour, 14)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.hour, 15)
        self.assertEqual(we.minute, 0)

    def test_within_the_time_range_of(self):
        inst = ("On March 4, 2021, within the time range of 14:30 to 15:00, "
                "a single failure was detected.")
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.hour, 14)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.hour, 15)
        self.assertEqual(we.minute, 0)
        self.assertEqual(ws.day, 4)

    def test_between_the_time_range_of(self):
        inst = ("On March 10, 2021, between the time range of 04:30 to 05:00, "
                "the system experienced one failure.")
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.hour, 4)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.hour, 5)
        self.assertEqual(we.minute, 0)
        self.assertEqual(ws.day, 10)

    def test_during_the_time_range_of(self):
        inst = ("On March 23, 2021, during the time range of 00:00 to 00:30, "
                "there was a recorded failure.")
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.hour, 0)
        self.assertEqual(ws.minute, 0)
        self.assertEqual(we.hour, 0)
        self.assertEqual(we.minute, 30)
        self.assertEqual(ws.day, 23)

    def test_cross_midnight_with_second_date_at(self):
        inst = ("During the specified time range of March 6, 2021, "
                "from 23:30 to March 7, 2021, at 00:00, "
                "there was one failure observed.")
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.month, 3)
        self.assertEqual(ws.day, 6)
        self.assertEqual(ws.hour, 23)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.month, 3)
        self.assertEqual(we.day, 7)
        self.assertEqual(we.hour, 0)
        self.assertEqual(we.minute, 0)
        self.assertGreater(we, ws)

    def test_cross_midnight_with_second_date_no_at(self):
        inst = ("During the specified time range of March 9, 2021, "
                "from 23:30 to March 10, 2021, 00:00, "
                "there was a single failure.")
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.month, 3)
        self.assertEqual(ws.day, 9)
        self.assertEqual(ws.hour, 23)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.month, 3)
        self.assertEqual(we.day, 10)
        self.assertEqual(we.hour, 0)
        self.assertEqual(we.minute, 0)
        self.assertGreater(we, ws)

    def test_single_date_cross_midnight(self):
        inst = "On March 6, 2021, from 23:30 to 00:00, a failure was detected."
        ws, we = parse_observation_window(inst)
        self.assertEqual(ws.day, 6)
        self.assertEqual(ws.hour, 23)
        self.assertEqual(ws.minute, 30)
        self.assertEqual(we.day, 7)
        self.assertEqual(we.hour, 0)
        self.assertEqual(we.minute, 0)
        self.assertGreater(we, ws)

    def test_unparseable_raises_QueryParseError_with_ids(self):
        instruction = "This instruction has no date and no time at all."
        with tempfile.TemporaryDirectory() as tmpdir:
            query_csv = Path(tmpdir) / "query.csv"
            write_query_csv(
                query_csv,
                ["task_index", "instruction"],
                {
                    "task_index": "task_1",
                    "instruction": instruction,
                },
            )
            with self.assertRaises(QueryParseError) as ctx:
                parse_inference_queries(str(query_csv))
            self.assertIn("query_ids=[0]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
