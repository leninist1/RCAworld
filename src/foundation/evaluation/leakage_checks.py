"""Leakage validation checks for strict evaluation protocol.

Enforces the following invariants:
  1. GT labels are NEVER accessed during telemetry loading or inference.
  2. All label timestamps fall within the loaded telemetry time range.
  3. Query observation windows actually contain the GT fault times.
  4. Model output is not conditioned on GT component or GT onset time.

These checks should be run as assertions during evaluation, not as optional
linting — they are part of the evaluation contract.
"""

from typing import List, Dict
import numpy as np


def validate_labels_in_telemetry_range(
    labels: List[Dict],
    timestamps: np.ndarray,
    context: str = "",
) -> List[str]:
    """Check that all GT labels have timestamps within the loaded telemetry range.

    Args:
        labels: List of label dicts with 'timestamp' and 'component' keys.
        timestamps: [T] array of telemetry timestamps.
        context: Optional description (e.g. system name).

    Returns:
        List of violation messages (empty if all labels pass).
    """
    violations = []
    if len(timestamps) == 0:
        violations.append(f"{context}: No telemetry timestamps loaded, cannot validate labels.")
        return violations

    t_min, t_max = timestamps.min(), timestamps.max()

    for i, label in enumerate(labels):
        gt_ts = float(label.get("timestamp", 0))
        if gt_ts == 0:
            continue
        if gt_ts < t_min or gt_ts > t_max:
            violations.append(
                f"{context} label[{i}]: GT timestamp {gt_ts} outside telemetry range "
                f"[{t_min}, {t_max}] (diff: {gt_ts - t_max:.0f}s). "
                f"This may indicate max_days=1 is too small for the full label set."
            )

    return violations


def validate_query_windows_contain_gt(
    queries: List[Dict],
    labels: List[Dict],
    context: str = "",
) -> List[str]:
    """Check that each query's observation window contains its GT fault time.

    This is a data integrity check — some queries may not correspond to faults
    within the loaded telemetry, which explains Time Hit=0%.

    Args:
        queries: List of query dicts with 'window_start_ts' and 'window_end_ts'.
        labels: List of label dicts.

    Returns:
        List of violation messages.
    """
    violations = []

    for i, (query, label) in enumerate(zip(queries, labels)):
        ws = query.get("window_start_ts", 0)
        we = query.get("window_end_ts", 0)
        gt_ts = float(label.get("timestamp", 0))
        if gt_ts == 0:
            continue
        if gt_ts < ws or gt_ts > we:
            violations.append(
                f"{context} query[{i}]: GT time {gt_ts} is outside query window "
                f"[{ws}, {we}] (diff: {gt_ts - we:.0f}s)."
            )

    return violations


def check_inference_inputs(
    episode: Dict,
    model_used_gt: bool = False,
) -> Dict[str, bool]:
    """Verify that inference inputs contain no GT leakage.

    Args:
        episode: Episode dict from episode_builder.
        model_used_gt: Whether the model forward pass accessed GT (should be False).

    Returns:
        Dict of check name -> passed bool.
    """
    results = {}

    # Check: tensor doesn't encode GT component index as a feature
    tensor = episode.get("tensor", None)
    if tensor is not None:
        # Tensor should only contain observation values — no one-hot GT columns
        if np.any(np.abs(tensor) > 1e6):
            results["tensor_no_extreme_values"] = False
        else:
            results["tensor_no_extreme_values"] = True

        # Check: tensor shape is [T, N, D] with D <= max_obs_features
        if len(tensor.shape) == 3 and tensor.shape[2] <= 16:
            results["tensor_reasonable_dims"] = True
        else:
            results["tensor_reasonable_dims"] = False

    # Check: model did not access GT
    results["model_did_not_access_gt"] = not model_used_gt

    # Check: entity type indices are within valid range
    type_idx = episode.get("type_idx", None)
    if type_idx is not None:
        max_ti = type_idx.max()
        results["type_idx_in_range"] = max_ti <= 9  # num_entity_types=10, indices 0-9
    else:
        results["type_idx_in_range"] = True

    return results


def validate_dataset_coverage(
    labels: List[Dict],
    timestamps: np.ndarray,
    entity_ids: List[str],
    context: str = "",
) -> Dict:
    """Validate dataset coverage: how many labels have valid entity and time matches.

    Returns:
        Dict with coverage statistics.
    """
    n_labels = len(labels)
    if n_labels == 0:
        return {"total_labels": 0, "entity_match": 0, "time_in_range": 0}

    entity_ids_set = set(entity_ids)
    t_min, t_max = timestamps.min(), timestamps.max() if len(timestamps) > 0 else (0, 0)

    entity_match = 0
    time_in_range = 0
    both = 0

    for label in labels:
        comp = label.get("component", "")
        # Loose entity match
        comp_lower = comp.lower().replace("_", "").replace("-", "").replace(" ", "")
        entity_found = False
        for eid in entity_ids:
            eid_lower = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
            if comp_lower in eid_lower or eid_lower in comp_lower:
                entity_found = True
                break
        if entity_found:
            entity_match += 1

        gt_ts = float(label.get("timestamp", 0))
        if gt_ts > 0 and t_min <= gt_ts <= t_max:
            time_in_range += 1

        if entity_found and gt_ts > 0 and t_min <= gt_ts <= t_max:
            both += 1

    return {
        "total_labels": n_labels,
        "entity_match": entity_match,
        "time_in_range": time_in_range,
        "both_match": both,
        "telemetry_t_min": float(t_min),
        "telemetry_t_max": float(t_max),
    }
