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
