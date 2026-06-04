"""Evaluation metrics for RCAWorld-Foundation.

Implements OpenRCA-style evaluation:
  - Exact match: all requested fields (time, component, reason) match exactly.
  - Component-only metrics: Top-1, Top-3, MRR for entity localization.
  - Onset accuracy: time window tolerance for onset detection.
  - Calibration metrics: confidence vs accuracy alignment.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class RCAPrediction:
    """Single RCA prediction with optional fields."""
    component: str = ""
    occurrence_datetime: str = ""
    reason: str = ""
    confidence: float = 0.0


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics."""
    # Component metrics
    top1: float = 0.0
    top3: float = 0.0
    top5: float = 0.0
    mrr: float = 0.0
    avg_rank: float = 0.0

    # Onset metrics
    onset_accuracy: float = 0.0
    onset_mae: float = 0.0

    # Exact match
    exact_match: float = 0.0

    # Per-query breakdown
    component_only_exact: float = 0.0
    component_time_exact: float = 0.0
    component_reason_exact: float = 0.0
    full_exact: float = 0.0

    # Sample counts
    num_samples: int = 0
    num_entities: int = 0


def compute_component_metrics(
    scores: np.ndarray,
    true_idx: np.ndarray,
    masks: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Top-K and MRR for component localization.

    Args:
        scores: [B, N] per-entity component scores (higher = more likely).
        true_idx: [B] indices of true root cause entities.
        masks: [B] optional validity mask (1.0 = valid).

    Returns:
        Dict with top1, top3, top5, mrr, avg_rank.
    """
    B, N = scores.shape
    if masks is None:
        masks = np.ones(B, dtype=np.float32)

    valid = masks > 0.5
    if not valid.any():
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0,
                "mrr": 0.0, "avg_rank": 0.0}

    ranks = np.argsort(-scores, axis=1)  # [B, N] descending
    true_pos = np.argmax(ranks == true_idx[:, None], axis=1) + 1  # 1-indexed ranks

    top1 = np.sum((true_pos <= 1).astype(np.float32) * masks) / max(1, masks.sum())
    top3 = np.sum((true_pos <= 3).astype(np.float32) * masks) / max(1, masks.sum())
    top5 = np.sum((true_pos <= 5).astype(np.float32) * masks) / max(1, masks.sum())
    mrr = np.sum((1.0 / true_pos) * masks) / max(1, masks.sum())
    avg_rank = np.sum(true_pos * masks) / max(1, masks.sum())

    return {
        "top1": float(top1),
        "top3": float(top3),
        "top5": float(top5),
        "mrr": float(mrr),
        "avg_rank": float(avg_rank),
    }


def compute_onset_metrics(
    onset_scores: np.ndarray,
    true_onset_times: np.ndarray,
    masks: Optional[np.ndarray] = None,
    tolerance: int = 3,
) -> Dict[str, float]:
    """Compute onset detection accuracy.

    Args:
        onset_scores: [B, T, N] onset probabilities.
        true_onset_times: [B, N] ground truth onset timestep indices.
        masks: [B, N] validity mask.
        tolerance: Allowed timestep error for "correct" detection.

    Returns:
        Dict with accuracy (within tolerance) and MAE.
    """
    B, T, N = onset_scores.shape
    if masks is None:
        masks = np.ones((B, N), dtype=np.float32)

    pred_times = np.argmax(onset_scores, axis=1)  # [B, N]
    errors = np.abs(pred_times - true_onset_times)  # [B, N]
    accurate = (errors <= tolerance).astype(np.float32)

    total_valid = masks.sum() + 1e-8
    accuracy = np.sum(accurate * masks) / total_valid
    mae = np.sum(errors * masks) / total_valid

    return {"onset_accuracy": float(accuracy), "onset_mae": float(mae)}


def evaluate_exact_match(
    predictions: List[RCAPrediction],
    labels: List[Dict],
    query_masks: Optional[List[Dict]] = None,
    time_tolerance_sec: float = 60.0,
) -> Dict[str, float]:
    """Evaluate exact match according to OpenRCA protocol.

    A prediction is "exact match" if ALL requested fields match the ground
    truth. Fields not requested are ignored.

    Args:
        predictions: List of RCAPrediction objects.
        labels: List of ground truth dicts with 'component', 'datetime', 'reason'.
        query_masks: Per-sample query masks specifying which fields are required.
        time_tolerance_sec: Tolerance window for time matching.

    Returns:
        Dict with exact_match and per-field breakdown metrics.
    """
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0}

    exact_matches = 0
    comp_matches = 0
    comp_time_matches = 0
    comp_reason_matches = 0
    full_matches = 0

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        qm = query_masks[i] if query_masks else {
            "need_time": True, "need_component": True, "need_reason": True}

        need_time = qm.get("need_time", True)
        need_component = qm.get("need_component", True)
        need_reason = qm.get("need_reason", True)

        # Component match
        comp_ok = (not need_component or
                   pred.component.lower().strip() ==
                   label.get("component", "").lower().strip())

        # Time match
        time_ok = True
        if need_time:
            try:
                pred_ts = float(pred.occurrence_datetime)
                true_ts = float(label.get("datetime", 0))
                time_ok = abs(pred_ts - true_ts) <= time_tolerance_sec
            except (ValueError, TypeError):
                time_ok = False

        # Reason match
        reason_ok = (not need_reason or
                     pred.reason.lower().strip() ==
                     label.get("reason", "").lower().strip())

        if comp_ok:
            comp_matches += 1
        if comp_ok and time_ok:
            comp_time_matches += 1
        if comp_ok and reason_ok:
            comp_reason_matches += 1

        # Exact: all requested fields match
        all_ok = ((not need_component or comp_ok) and
                  (not need_time or time_ok) and
                  (not need_reason or reason_ok))
        if all_ok:
            exact_matches += 1
        if comp_ok and time_ok and reason_ok:
            full_matches += 1

    return {
        "exact_match": exact_matches / n,
        "component_only_exact": comp_matches / n,
        "component_time_exact": comp_time_matches / n,
        "component_reason_exact": comp_reason_matches / n,
        "full_exact": full_matches / n,
        "num_samples": n,
    }


def evaluate_predictions(
    scores: np.ndarray,
    onset_scores: Optional[np.ndarray],
    true_component_idx: np.ndarray,
    true_onset_times: Optional[np.ndarray],
    entity_ids: List[str],
    component_masks: Optional[np.ndarray] = None,
    onset_masks: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Run full evaluation suite.

    Args:
        scores: [B, N] component scores.
        onset_scores: [B, T, N] optional onset scores.
        true_component_idx: [B] true entity indices.
        true_onset_times: [B, N] optional true onset times.
        entity_ids: [N] entity identifiers.
        component_masks: [B] optional validity mask.
        onset_masks: [B, N] optional validity mask.

    Returns:
        EvaluationResult with all metrics.
    """
    result = EvaluationResult()
    result.num_entities = len(entity_ids)

    B = scores.shape[0]
    if component_masks is None:
        component_masks = np.ones(B, dtype=np.float32)
    valid_count = int(component_masks.sum())

    if valid_count == 0:
        return result

    result.num_samples = valid_count

    # Component metrics
    comp_metrics = compute_component_metrics(
        scores, true_component_idx, component_masks)
    result.top1 = comp_metrics["top1"]
    result.top3 = comp_metrics["top3"]
    result.top5 = comp_metrics["top5"]
    result.mrr = comp_metrics["mrr"]
    result.avg_rank = comp_metrics["avg_rank"]

    # Onset metrics
    if onset_scores is not None and true_onset_times is not None:
        onset_metrics = compute_onset_metrics(
            onset_scores, true_onset_times, onset_masks)
        result.onset_accuracy = onset_metrics["onset_accuracy"]
        result.onset_mae = onset_metrics["onset_mae"]

    return result


def report_metrics(result: EvaluationResult,
                   system_name: str = "") -> str:
    """Format evaluation result as a readable string."""
    lines = []
    if system_name:
        lines.append(f"=== {system_name} ===")
    lines.append(f"Entities: {result.num_entities}, Samples: {result.num_samples}")
    lines.append(f"Component Top-1: {result.top1:.1%}")
    lines.append(f"Component Top-3: {result.top3:.1%}")
    lines.append(f"Component Top-5: {result.top5:.1%}")
    lines.append(f"Component MRR:   {result.mrr:.3f}")
    lines.append(f"Component AvgR:  {result.avg_rank:.2f}")
    if result.onset_accuracy > 0 or result.onset_mae > 0:
        lines.append(f"Onset Accuracy: {result.onset_accuracy:.1%}")
        lines.append(f"Onset MAE:      {result.onset_mae:.2f}")
    if result.exact_match > 0:
        lines.append(f"Exact Match:    {result.exact_match:.1%}")
    return "\n".join(lines)
