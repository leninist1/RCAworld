"""Phase 0.8 R6.1: Export predictions.json to OpenRCA-format prediction.csv.

Reads predictions.json and produces prediction.csv with columns controlled
by Query Mask (need_time, need_component). Reason-requiring queries are
explicitly rejected (not yet supported).
"""

import csv
import json
import os
import sys
import argparse
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


_REQUIRED_ENTRY_FIELDS = [
    "system", "query_id", "predictions",
    "need_time", "need_component", "need_reason",
]


class ExportError(Exception):
    """Raised on invalid or unsupported prediction entries."""


def _validate_entry(entry, idx):
    """Validate a single prediction entry has all required fields."""
    for field in _REQUIRED_ENTRY_FIELDS:
        if field not in entry:
            raise ExportError(
                f"Entry {idx}: missing required field '{field}'")
    if not isinstance(entry["predictions"], list):
        raise ExportError(f"Entry {idx}: 'predictions' must be a list")
    for field in ("need_time", "need_component", "need_reason"):
        if not isinstance(entry[field], bool):
            raise ExportError(
                f"Entry {idx}: '{field}' must be a boolean, "
                f"got {type(entry[field]).__name__}")


def _entry_fieldnames(entry):
    """Determine CSV columns for an entry based on Query Mask."""
    cols = ["query_id"]
    if entry["need_time"]:
        cols.append("occurrence_time")
    if entry["need_component"]:
        cols.append("component")
    return cols


def build_export_rows(entry):
    """Build list of row dicts for one prediction entry.

    Args:
        entry: Dict from predictions.json with system, query_id, predictions,
               need_time, need_component, need_reason.

    Returns:
        List of OrderedDict rows, one per root cause.

    Raises:
        ExportError: On unsupported masks, missing fields, or absent prediction fields.
    """
    _validate_entry(entry, f"system={entry.get('system', '?')} "
                    f"query_id={entry.get('query_id', '?')}")

    need_time = entry["need_time"]
    need_component = entry["need_component"]
    need_reason = entry["need_reason"]
    query_id = entry["query_id"]
    predictions = entry["predictions"]

    if need_reason:
        raise ExportError(
            f"query_id={query_id}: need_reason=True is not supported "
            f"(reason inference not yet implemented)")

    fieldnames = _entry_fieldnames(entry)
    rows = []

    for pidx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise ExportError(
                f"query_id={query_id}: prediction[{pidx}] is not a dict")
        if "score" not in pred:
            raise ExportError(
                f"query_id={query_id}: prediction[{pidx}] missing 'score'")

        if need_time and "datetime" not in pred:
            raise ExportError(
                f"query_id={query_id}: prediction[{pidx}] missing "
                f"'datetime' (need_time=True)")
        if need_component and "component" not in pred:
            raise ExportError(
                f"query_id={query_id}: prediction[{pidx}] missing "
                f"'component' (need_component=True)")

        row = OrderedDict()
        row["query_id"] = query_id
        if need_time:
            row["occurrence_time"] = pred["datetime"]
        if need_component:
            row["component"] = pred["component"]
        rows.append(row)

    if not rows:
        return rows

    if need_time:
        rows.sort(key=lambda r: r["occurrence_time"])
    return rows


def export_prediction_csv(predictions, output_path):
    """Export a list of prediction entries to prediction.csv.

    Args:
        predictions: List of prediction entry dicts (from predictions.json).
        output_path: Path to write CSV file.

    Raises:
        ExportError: On validation failures.

    Returns:
        (written_count, skipped_reason_count) tuple.
    """
    all_rows = []
    reason_suppressed = 0

    for idx, entry in enumerate(predictions):
        _validate_entry(entry, idx)
        if entry["need_reason"]:
            reason_suppressed += 1
            continue

        rows = build_export_rows(entry)
        all_rows.extend(rows)

    if not all_rows:
        return 0, reason_suppressed

    fieldnames = list(all_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows), reason_suppressed


def main():
    parser = argparse.ArgumentParser(
        description="Export predictions.json to OpenRCA prediction.csv")
    parser.add_argument("--predictions", type=str, default="predictions.json",
                        help="Path to predictions.json")
    parser.add_argument("--output", type=str, default="prediction.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions file not found: {args.predictions}")
        return

    with open(args.predictions, "r") as f:
        all_predictions = json.load(f)

    print(f"Loaded {len(all_predictions)} prediction entries")

    try:
        written, suppressed = export_prediction_csv(all_predictions, args.output)
    except ExportError as e:
        print(f"ERROR: {e}")
        return

    print(f"Exported {written} rows to {args.output}"
          + (f" ({suppressed} reason queries skipped)" if suppressed else ""))


if __name__ == "__main__":
    main()
