"""Minimal OpenRCA-compatible prediction parser.

Source: extracted from OpenRCA official evaluate.py
  (/home/dell2/RCA513/syh/OpenRCA-main/main/evaluate.py)

The core prediction-parsing regex (predict_pattern) is copied verbatim
from evaluate.py:19-24.  This fixture provides only the parsing logic;
scoring, permutation matching, and report generation are omitted.

Used by tests to verify that the CSV produced by phaseA_export_openrca_csv.py
is readable and parseable by the official OpenRCA evaluator.
"""

import csv
import re

# ---------------------------------------------------------------------------
# Copied verbatim from OpenRCA evaluate.py, lines 19-24
# ---------------------------------------------------------------------------
_PREDICT_PATTERN = (
    r'{\s*'
    r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
    r'(?:"root cause component":\s*"(.*?)")?,?\s*'
    r'(?:"root cause reason":\s*"(.*?)")?\s*}'
)


def parse_prediction_column(prediction_str):
    """Parse the `prediction` column of an OpenRCA prediction.csv.

    Uses the *exact same regex* as the official evaluate.py → evaluate()
    function to extract root cause dicts from the prediction JSON string.

    Args:
        prediction_str: The full prediction cell string (e.g. a JSON string
                        like '{"1":{...},"2":{...}}').

    Returns:
        List of dicts, each with keys:
            "root cause occurrence datetime"  (str or "")
            "root cause component"            (str or "")
            "root cause reason"               (str or "")
        Exactly mirrors the format produced by evaluate() lines 30-36.

    Raises:
        ValueError: If the prediction string cannot be parsed.
    """
    matches = re.findall(_PREDICT_PATTERN, prediction_str)
    if not matches:
        raise ValueError(
            f"prediction column could not be parsed: {prediction_str!r}")
    results = []
    for match in matches:
        datetime_str, component, reason = match
        results.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason,
        })
    return results


def load_prediction_csv(csv_path):
    """Read an OpenRCA prediction.csv and return its contents.

    Args:
        csv_path: Path to prediction.csv file (or file-like object).

    Returns:
        List of dicts, one per CSV row, with keys matching the CSV columns.
        Typically: {"row_id": ..., "prediction": ..., "system": ...}
    """
    if isinstance(csv_path, str):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        reader = csv.DictReader(csv_path)
        return list(reader)


def validate_prediction_csv(csv_path):
    """Validate a prediction.csv against the OpenRCA format.

    Checks performed:
      1. CSV is readable (csv.DictReader succeeds).
      2. `prediction` column exists.
      3. Every prediction cell can be parsed by parse_prediction_column().

    Args:
        csv_path: Path to prediction.csv.

    Returns:
        (parsed_rows, csv_rows) tuple:
        - parsed_rows: list of lists of root cause dicts (one list per CSV row)
        - csv_rows: raw csv.DictReader rows

    Raises:
        ValueError: If any validation check fails.
    """
    csv_rows = load_prediction_csv(csv_path)

    if not csv_rows:
        raise ValueError("prediction.csv is empty")

    if "prediction" not in csv_rows[0]:
        raise ValueError("prediction.csv missing required 'prediction' column")

    parsed_rows = []
    for idx, row in enumerate(csv_rows):
        pred_str = row["prediction"]
        try:
            rcs = parse_prediction_column(pred_str)
        except ValueError:
            raise
        if not rcs:
            raise ValueError(
                f"Row {idx}: prediction column parsed to zero root causes: "
                f"{pred_str!r}")
        parsed_rows.append(rcs)

    return parsed_rows, csv_rows


# ---------------------------------------------------------------------------
# OpenRCA file_evaluate() compatible logic
# Source: OpenRCA official main/evaluate.py, lines 104-149
# Replicates row_id sorting, length-mismatch check, and row-by-row
# prediction/scoring_points/instruction/task_index iteration.
# Omits: full evaluate() scoring (permutation matching), report file writing,
# and pandas DataFrame usage (replaced by csv.DictReader).
# ---------------------------------------------------------------------------

def file_evaluate_compatible(prediction_file, query_file):
    """Mimic OpenRCA official file_evaluate() row-by-row protocol.

    Logic mirrored from OpenRCA evaluate.py file_evaluate(), lines 104-149:
      1. Read prediction.csv and query.csv
      2. If 'row_id' column exists in prediction, sort by it
      3. Assert prediction rows == query rows (raise ValueError on mismatch)
      4. For each row: parse `prediction` column via parse_prediction_column(),
         pair with `scoring_points`, `instruction`, and `task_index` from query.csv

    Args:
        prediction_file: Path to prediction.csv.
        query_file: Path to query.csv (must have columns:
                     instruction, scoring_points, task_index).

    Returns:
        List of dicts, one per row, each containing:
            prediction_parsed: list of root cause dicts (from parse_prediction_column)
            scoring_points: str, ground truth scoring_points
            instruction: str, task instruction text
            task_index: str, task identifier (e.g. 'task_1')
            row_id: original row_id from prediction.csv (if present)

    Raises:
        ValueError: If row counts mismatch.
    """
    pred_rows = load_prediction_csv(prediction_file)
    query_rows = load_prediction_csv(query_file)

    if not pred_rows:
        raise ValueError("prediction.csv is empty")
    if not query_rows:
        raise ValueError("query.csv is empty")

    # Replicate: if 'row_id' in pred_df.columns → sort
    if "row_id" in pred_rows[0]:
        pred_rows.sort(key=lambda r: _to_int(r.get("row_id", 0)))

    if len(pred_rows) != len(query_rows):
        raise ValueError(
            "The length of prediction file and record file should be the same")

    results = []
    for idx in range(len(pred_rows)):
        pred_row = pred_rows[idx]
        query_row = query_rows[idx]

        prediction_str = pred_row.get("prediction", "")
        scoring_points = query_row.get("scoring_points", "")
        instruction = query_row.get("instruction", "")
        task_index = query_row.get("task_index", "")

        parsed = parse_prediction_column(prediction_str)

        result = {
            "prediction_parsed": parsed,
            "scoring_points": scoring_points,
            "instruction": instruction,
            "task_index": task_index,
        }
        if "row_id" in pred_row:
            result["row_id"] = pred_row["row_id"]
        results.append(result)

    return results


def _to_int(value):
    """Convert a CSV string value to int, falling back to 0."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
