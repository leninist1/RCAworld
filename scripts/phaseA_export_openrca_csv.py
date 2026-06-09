"""Phase 0.8 R6.1: Export predictions.json to OpenRCA-format prediction.csv.

Each query is exactly one CSV row with columns:
    row_id, prediction, [system]

The `prediction` column is a JSON string with numbered root causes:
    {
      "1": {"root cause occurrence datetime": "...", "root cause component": "..."},
      "2": {"root cause occurrence datetime": "...", "root cause component": "..."}
    }

Reason-requiring queries raise ExportError (not silently skipped).
Mixed-system input raises ExportError.
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

EXPORT_FIELDNAMES = ["row_id", "prediction", "system"]


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


def build_prediction_payload(entry):
    """Build a single prediction JSON payload dict for one query.

    Args:
        entry: Dict from predictions.json with system, query_id, predictions,
               need_time, need_component, need_reason.

    Returns:
        Dict with string keys "1", "2", ... each mapping to a dict with
        "root cause occurrence datetime" (if need_time) and/or
        "root cause component" (if need_component).

    Raises:
        ExportError: On unsupported masks, missing prediction fields.
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

    if not predictions:
        raise ExportError(
            f"query_id={query_id}: 'predictions' list is empty")

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

    ordered = list(predictions)
    if need_time:
        ordered.sort(key=lambda p: p["datetime"])

    payload = OrderedDict()
    for idx, pred in enumerate(ordered, start=1):
        item = OrderedDict()
        if need_time:
            item["root cause occurrence datetime"] = pred["datetime"]
        if need_component:
            item["root cause component"] = pred["component"]
        payload[str(idx)] = item

    return payload


def build_export_row(entry):
    """Build a single CSV row dict for one query.

    Args:
        entry: A prediction entry dict.

    Returns:
        {"row_id": ..., "prediction": ..., "system": ...}

    Raises:
        ExportError: On validation or unsupported masks.
    """
    _validate_entry(entry, f"system={entry.get('system', '?')} "
                    f"query_id={entry.get('query_id', '?')}")

    payload = build_prediction_payload(entry)

    row = OrderedDict()
    row["row_id"] = entry["query_id"]
    row["prediction"] = json.dumps(payload, ensure_ascii=False)
    row["system"] = entry["system"]
    return row


def export_prediction_csv(predictions, output_path, system=None):
    """Export a list of prediction entries to prediction.csv.

    Args:
        predictions: List of prediction entry dicts (from predictions.json).
        output_path: Path to write CSV file.
        system: Optional system filter. If given, only export entries
                matching this system name.

    Raises:
        ExportError: On mixed systems, reason queries, or validation failures.

    Returns:
        Number of rows written.
    """
    if not predictions:
        raise ExportError("No prediction entries to export")

    systems = set()
    entries = []
    for idx, entry in enumerate(predictions):
        _validate_entry(entry, idx)
        systems.add(entry["system"])

        if system is not None and entry["system"] != system:
            continue
        entries.append(entry)

    if not entries:
        if system is not None:
            raise ExportError(
                f"No prediction entries found for system='{system}'")
        raise ExportError("No prediction entries to export")

    if len(systems) > 1 and system is None:
        raise ExportError(
            f"Multiple systems detected: {sorted(systems)}. "
            f"Use --system to select one.")

    rows = []
    reason_err = None
    for idx, entry in enumerate(entries):
        try:
            row = build_export_row(entry)
        except ExportError as e:
            reason_err = e
            raise
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export predictions.json to OpenRCA prediction.csv")
    parser.add_argument("--predictions", type=str, default="predictions.json",
                        help="Path to predictions.json")
    parser.add_argument("--output", type=str, default="prediction.csv",
                        help="Output CSV path")
    parser.add_argument("--system", type=str, default=None,
                        help="Only export predictions for this system")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions file not found: {args.predictions}")
        return

    with open(args.predictions, "r") as f:
        all_predictions = json.load(f)

    print(f"Loaded {len(all_predictions)} prediction entries")

    try:
        written = export_prediction_csv(all_predictions, args.output, args.system)
    except ExportError as e:
        print(f"ERROR: {e}")
        return

    print(f"Exported {written} rows to {args.output}")


if __name__ == "__main__":
    main()
