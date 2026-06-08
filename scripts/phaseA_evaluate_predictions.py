"""Phase 0.8 Round 5: Evaluation script — separated counts, real ranking, query mask.

This script reads:
  - predictions.json (from phaseA_run_inference.py)
  - query.csv scoring_points column (via load_eval_targets)

Target construction is from scoring_points ONLY (not record.csv).
record.csv is optional diagnostic only.
Query mask (need_time / need_component / need_reason) controls output fields.
Reason-requiring queries are unsupported and written to skipped_queries.csv.
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime as dt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from foundation.evaluation.query_parser import (
    load_eval_targets,
    OPENRCA_TZ,
)

OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"

SYSTEM_CONFIGS = {
    "Bank": {"data_dir": os.path.join(OPENRCA, "Bank", "Bank"),
             "adapter_cls": "OpenRCABankAdapter"},
    "Market/cloudbed-1": {"data_dir": os.path.join(OPENRCA, "Market", "Market", "cloudbed-1"),
                           "adapter_cls": "OpenRCAMarketAdapter"},
    "Market/cloudbed-2": {"data_dir": os.path.join(OPENRCA, "Market", "Market", "cloudbed-2"),
                           "adapter_cls": "OpenRCAMarketAdapter"},
    "Telecom": {"data_dir": os.path.join(OPENRCA, "Telecom", "Telecom"),
                "adapter_cls": "OpenRCATelecomAdapter"},
}


def _make_adapter(adapter_cls_name, include_dates):
    from foundation.adapters import (
        OpenRCABankAdapter, OpenRCATelecomAdapter, OpenRCAMarketAdapter,
    )
    cls_map = {
        "OpenRCABankAdapter": lambda: OpenRCABankAdapter(
            max_days=99, max_container_events=500000, max_container_rows=5000000,
            include_dates=include_dates),
        "OpenRCATelecomAdapter": lambda: OpenRCATelecomAdapter(
            max_days=99, max_container_timestamps=5000,
            include_dates=include_dates),
        "OpenRCAMarketAdapter": lambda: OpenRCAMarketAdapter(
            max_days=99, max_container_events=500000, max_container_rows=5000000,
            include_dates=include_dates),
    }
    return cls_map[adapter_cls_name]()


def _extract_labels(data_dir, entity_ids, adapter):
    """Extract labels from record.csv — EVALUATION ONLY."""
    all_entities = adapter.discover_entities(data_dir)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities
    labels_raw = adapter.extract_labels(data_dir, cont_entities)
    labels = []
    for lb in labels_raw:
        matched = -1
        if lb.component in entity_ids:
            matched = entity_ids.index(lb.component)
        else:
            cl = lb.component.lower().replace("_", "").replace("-", "").replace(" ", "")
            for i, eid in enumerate(entity_ids):
                el = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
                if cl in el or el in cl:
                    matched = i
                    break
        if matched >= 0:
            ts_val = float(lb.occurrence_datetime) if lb.occurrence_datetime else 0.0
            labels.append({
                "component": lb.component,
                "component_idx": matched,
                "timestamp": ts_val,
                "reason": lb.reason,
            })
    return labels


def _normalize_component_name(name):
    return str(name).lower().replace("_", "").replace("-", "").replace(" ", "")


def _resolve_component_idx(component_name, entity_ids):
    if not component_name:
        return -1
    if component_name in entity_ids:
        return entity_ids.index(component_name)

    comp_norm = _normalize_component_name(component_name)
    for idx, entity_id in enumerate(entity_ids):
        entity_norm = _normalize_component_name(entity_id)
        if comp_norm and entity_norm and (comp_norm in entity_norm or entity_norm in comp_norm):
            return idx
    return -1


def _parse_prediction(prediction, entity_ids):
    pred_dt_str = prediction.get("datetime", "")
    pred_ts = None
    if pred_dt_str:
        try:
            pred_dt = dt.strptime(pred_dt_str, "%Y-%m-%d %H:%M:%S")
            pred_ts = pred_dt.replace(tzinfo=OPENRCA_TZ).timestamp()
        except (ValueError, OSError):
            pred_ts = None

    pred_component = prediction.get("component", "")
    return {
        "component": pred_component,
        "component_idx": _resolve_component_idx(pred_component, entity_ids),
        "timestamp": pred_ts,
        "datetime": pred_dt_str,
        "score": prediction.get("score"),
        "raw": prediction,
    }


def _build_eval_targets(eval_target, entity_ids, need_component=True, need_time=True):
    """Build official eval targets directly from EvalTarget.root_causes.

    Each root_cause is (component, datetime_str, reason, tolerance_min).
    Targets are constructed directly from scoring_points — record.csv labels
    are NOT used for component or timestamp decisions.

    Respects query mask:
    - need_component=False: component_idx set to -1 (any component ok).
    - need_time=False: timestamp set to 0.0 (no time constraint).

    Returns:
        targets: List of target dicts.
        unresolved_targets: List of root_cause tuples that could not be resolved.
    """
    targets = []
    unresolved_targets = []

    for comp, dt_str, reason, tolerance in eval_target.root_causes:
        parsed_ts = None
        if dt_str:
            try:
                gt_dt = dt.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                parsed_ts = gt_dt.replace(tzinfo=OPENRCA_TZ).timestamp()
            except (ValueError, OSError):
                pass

        comp_idx = _resolve_component_idx(comp, entity_ids) if need_component else -1

        is_valid = True
        if need_component and comp_idx < 0:
            is_valid = False
        if need_time and parsed_ts is None:
            is_valid = False

        if is_valid:
            targets.append({
                "component": comp if comp else "",
                "component_idx": comp_idx if need_component else -1,
                "timestamp": parsed_ts if parsed_ts is not None else 0.0,
                "reason": reason,
                "tolerance": tolerance,
                "target_component": comp,
                "target_datetime": dt_str,
                "target_reason": reason,
            })
        else:
            unresolved_targets.append((comp, dt_str, reason, tolerance))

    return targets, unresolved_targets


def _matching_cost(prediction, target, component_mismatch_penalty=10000.0):
    cost = 0.0
    target_cidx = target.get("component_idx", -1)
    pred_cidx = prediction.get("component_idx", -1)

    if target_cidx >= 0 and pred_cidx != target_cidx:
        cost += component_mismatch_penalty

    pred_ts = prediction.get("timestamp")
    target_ts = target.get("timestamp")
    if target_ts is not None and target_ts > 0:
        if pred_ts is None:
            cost += component_mismatch_penalty
        else:
            cost += abs(pred_ts - target_ts) / 60.0
    return cost


def match_predictions_to_targets(predictions, targets):
    """Greedy one-to-one matching between predictions and GT targets."""
    unmatched_prediction_indices = list(range(len(predictions)))
    unmatched_target_indices = list(range(len(targets)))
    matched_pairs = []

    while unmatched_prediction_indices and unmatched_target_indices:
        best = None
        for pred_idx in unmatched_prediction_indices:
            for target_idx in unmatched_target_indices:
                cost = _matching_cost(predictions[pred_idx], targets[target_idx])
                candidate = (cost, pred_idx, target_idx)
                if best is None or candidate < best:
                    best = candidate

        _, pred_idx, target_idx = best
        matched_pairs.append({
            "prediction": predictions[pred_idx],
            "target": targets[target_idx],
        })
        unmatched_prediction_indices.remove(pred_idx)
        unmatched_target_indices.remove(target_idx)

    unmatched_predictions = [predictions[idx] for idx in unmatched_prediction_indices]
    unmatched_targets = [targets[idx] for idx in unmatched_target_indices]
    return matched_pairs, unmatched_predictions, unmatched_targets


def _evaluate_prediction_target_pair(prediction, target, entity_ids,
                                       component_ranking=None,
                                        need_component=True, need_time=True):
    """Evaluate one prediction-target pair with real component ranking.

    Args:
        prediction: Parsed prediction dict or None (for unmatched targets).
        target: GT target dict.
        entity_ids: List of entity ID strings.
        component_ranking: List of (component_name, score) sorted desc, or None.
        need_component: Whether component is evaluated (from query mask).
        need_time: Whether time is evaluated (from query mask).

    Returns:
        Dict of per-root-cause metrics.

    Key semantics:
        - component_rank: position of GT in full component_ranking (1..N).
          Drives Top-1, Top-3, MRR, avg_rank.
        - selected_component_hit: whether the model's chosen prediction
          component matches GT. Drives Joint Hit.
        - For unresolved GT (component_idx < 0): explicit miss in all
          component/joint metrics regardless of entity count.
    """
    gt_c = target.get("component_idx", -1)
    gt_ts = target.get("timestamp", 0.0)
    tolerance = target.get("tolerance", 1)
    gt_unresolved = (gt_c < 0 and need_component)

    component_rank = len(entity_ids)
    selected_component_hit = False

    if gt_unresolved:
        component_rank = len(entity_ids)
        selected_component_hit = False
    elif need_component and component_ranking is not None:
        found_in_ranking = False
        for rank, (cname, _cscore) in enumerate(component_ranking):
            cidx = _resolve_component_idx(cname, entity_ids)
            if cidx == gt_c:
                component_rank = rank + 1
                found_in_ranking = True
                break
        if not found_in_ranking:
            component_rank = len(entity_ids)
        selected_component_hit = (
            prediction is not None
            and prediction.get("component_idx", -1) == gt_c
        )
    elif need_component:
        selected_component_hit = (
            prediction is not None
            and prediction.get("component_idx", -1) == gt_c
        )
        component_rank = 1 if selected_component_hit else len(entity_ids)

    time_hit = False
    time_error_min = float('inf')
    if need_time and gt_ts > 0:
        if prediction is not None and prediction.get("timestamp") is not None:
            time_error_sec = abs(prediction["timestamp"] - gt_ts)
            time_error_min = time_error_sec / 60.0
            time_hit = time_error_sec <= tolerance * 60

    if need_component and need_time:
        joint_hit = selected_component_hit and time_hit
    elif need_component:
        joint_hit = selected_component_hit
    elif need_time:
        joint_hit = time_hit
    else:
        joint_hit = False

    if gt_unresolved:
        top1 = False
        top3 = False
        rr = 0.0
    else:
        top1 = (component_rank == 1)
        top3 = (component_rank <= 3)
        rr = 1.0 / max(1, component_rank)

    return {
        "component_rank": component_rank,
        "component_top1": top1,
        "component_top3": top3,
        "time_error": float('nan') if time_error_min == float('inf') else time_error_min,
        "time_error_min": time_error_min if time_error_min != float('inf') else float('nan'),
        "time_hit": time_hit,
        "time_hit_5min": bool(time_error_min <= 5.0),
        "time_hit_10min": bool(time_error_min <= 10.0),
        "time_hit_15min": bool(time_error_min <= 15.0),
        "joint_component_hit": selected_component_hit,
        "joint_time_hit": time_hit,
        "joint_hit": joint_hit,
        "reciprocal_rank": rr,
        "component_applicable": need_component,
        "time_applicable": need_time,
        "joint_applicable": (need_component and need_time),
    }


def _aggregate_by_applicability(query_results):
    """Aggregate metrics with separate denominators by task type.

    - Component metrics (Top-1, Top-3, MRR): only component_applicable results.
    - Time metrics (Hit@5/10/15, MAE): only time_applicable results.
    - Joint metrics: only joint_applicable results.
    - Resolved-only variants: exclude unresolved from each subset.
    - Time MAE: only valid finite values from time_applicable results.

    Returns dict with all metrics, resolved-only, and denominator counts.
    """
    def _mean(items, key):
        if not items:
            return 0.0
        vals = [r[key] for r in items]
        return float(np.mean(vals))

    def _time_mae(items):
        valid = [r["time_error_min"] for r in items
                 if not np.isnan(r["time_error_min"]) and np.isfinite(r["time_error_min"])]
        if not valid:
            return 0.0
        return float(np.mean(valid))

    def _time_hit(items, key):
        if not items:
            return 0.0
        return float(np.mean([r[key] for r in items]))

    component_results = [r for r in query_results if r.get("component_applicable", True)]
    time_results = [r for r in query_results if r.get("time_applicable", True)]
    joint_results = [r for r in query_results if r.get("joint_applicable", False)]
    resolved_all = [r for r in query_results if not r.get("unresolved", False)]
    resolved_component = [r for r in resolved_all if r.get("component_applicable", True)]
    resolved_time = [r for r in resolved_all if r.get("time_applicable", True)]
    resolved_joint = [r for r in resolved_all if r.get("joint_applicable", False)]

    return {
        "n": len(query_results),
        "n_resolved": len(resolved_all),
        "component_metric_count": len(component_results),
        "time_metric_count": len(time_results),
        "joint_metric_count": len(joint_results),
        "resolved_component_count": len(resolved_component),
        "resolved_time_count": len(resolved_time),
        "resolved_joint_count": len(resolved_joint),

        "component_top1": _mean(component_results, "component_top1"),
        "component_top3": _mean(component_results, "component_top3"),
        "mrr": _mean(component_results, "reciprocal_rank"),
        "avg_rank": _mean(component_results, "component_rank"),

        "time_hit_5min": _time_hit(time_results, "time_hit_5min"),
        "time_hit_10min": _time_hit(time_results, "time_hit_10min"),
        "time_hit_15min": _time_hit(time_results, "time_hit_15min"),
        "time_mae_min": _time_mae(time_results),

        "joint_hit_rate": _mean(joint_results, "joint_hit"),

        "resolved_component_top1": _mean(resolved_component, "component_top1"),
        "resolved_component_top3": _mean(resolved_component, "component_top3"),
        "resolved_mrr": _mean(resolved_component, "reciprocal_rank"),
    }


def evaluate_system(sys_name, predictions_by_key, resample_sec):
    """Evaluate predictions for one system with separated counts and query mask.

    Returns:
        agg: Aggregated metrics dict.
        skipped_queries: List of skipped query records for skipped_queries.csv.
    """
    cfg = SYSTEM_CONFIGS[sys_name]
    data_dir = cfg["data_dir"]
    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return {"n": 0}, []

    query_path = os.path.join(data_dir, "query.csv")
    if not os.path.exists(query_path):
        print(f"  query.csv not found: {query_path}")
        return {"n": 0}, []

    eval_targets = load_eval_targets(query_path)
    print(f"  Loaded {len(eval_targets)} eval targets from query.csv scoring_points")

    adapter = _make_adapter(cfg["adapter_cls"], None)
    all_entities = adapter.discover_entities(data_dir)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities
    entity_ids = [e.entity_id for e in cont_entities]
    print(f"  Entities: {len(entity_ids)}")

    record_csv_path = os.path.join(data_dir, "record.csv")
    if os.path.exists(record_csv_path):
        labels = _extract_labels(data_dir, entity_ids, adapter)
        print(f"  Loaded {len(labels)} labels from record.csv (diagnostic only)")
    else:
        labels = []
        print("  record.csv not found — evaluation continues without diagnostic labels")

    official_query_count = len(eval_targets)
    parsed_query_count = 0
    evaluated_query_count = 0
    skipped_query_count = 0
    total_official_rc = 0
    total_resolved_rc = 0
    total_unresolved_rc = 0
    total_predicted_rc = 0
    total_matched_rc = 0
    total_unmatched_rc = 0

    query_results = []
    skipped_queries = []

    for qid, eval_tgt in eval_targets.items():
        official_rc_count = len(eval_tgt.root_causes)
        total_official_rc += official_rc_count
        parsed_query_count += 1

        need_component = eval_tgt.need_component
        need_time = eval_tgt.need_time
        need_reason = eval_tgt.need_reason

        if need_reason:
            skipped_query_count += 1
            skipped_queries.append({
                "query_id": int(qid),
                "reason": "unsupported",
                "details": "reason inference not yet supported",
            })
            continue

        target_component_count = 0
        target_time_count = 0
        for rc in eval_tgt.root_causes:
            _, dt_str, _, _ = rc
            if dt_str:
                target_time_count += 1
            target_component_count += 1

        pred_entry = predictions_by_key.get((sys_name, qid),
                                           {"query_id": qid, "predictions": []})
        raw_predictions = pred_entry.get("predictions", [])
        component_ranking = pred_entry.get("component_ranking")

        targets, unresolved = _build_eval_targets(eval_tgt, entity_ids,
                                                   need_component=need_component,
                                                   need_time=need_time)
        total_resolved_rc += len(targets)
        total_unresolved_rc += len(unresolved)

        resolved_count = len(targets)
        unresolved_count = len(unresolved)

        num_faults_in_query = resolved_count + unresolved_count

        if not targets and not unresolved:
            continue

        predictions = [_parse_prediction(prediction, entity_ids)
                       for prediction in raw_predictions]
        total_predicted_rc += len(predictions)

        if not targets:
            skipped_query_count += 1
            skipped_queries.append({
                "query_id": int(qid),
                "reason": "no_resolved_targets",
                "details": f"all {len(unresolved)} root cause(s) unresolved",
            })
            for _ in range(unresolved_count):
                metrics = _evaluate_prediction_target_pair(
                    None, {"component_idx": -1, "timestamp": 0.0, "tolerance": 1},
                    entity_ids, component_ranking=None,
                    need_component=need_component, need_time=need_time,
                )
                metrics["num_faults_in_query"] = num_faults_in_query
                metrics["unresolved"] = True
                query_results.append(metrics)
            continue

        matched_pairs, unmatched_predictions, unmatched_targets = \
            match_predictions_to_targets(predictions, targets)

        total_matched_rc += len(matched_pairs)
        total_unmatched_rc += len(unmatched_targets)

        for pair in matched_pairs:
            metrics = _evaluate_prediction_target_pair(
                pair["prediction"], pair["target"], entity_ids,
                component_ranking=component_ranking,
                need_component=need_component, need_time=need_time,
            )
            metrics["num_faults_in_query"] = num_faults_in_query
            metrics["query_window_coverage"] = 1.0
            metrics["unresolved"] = False
            query_results.append(metrics)

        for target in unmatched_targets:
            metrics = _evaluate_prediction_target_pair(
                None, target, entity_ids,
                component_ranking=component_ranking,
                need_component=need_component, need_time=need_time,
            )
            metrics["num_faults_in_query"] = num_faults_in_query
            metrics["query_window_coverage"] = 1.0
            metrics["unresolved"] = False
            query_results.append(metrics)

        for _ in range(unresolved_count):
            metrics = _evaluate_prediction_target_pair(
                None, {"component_idx": -1, "timestamp": 0.0, "tolerance": 1},
                entity_ids, component_ranking=None,
                need_component=need_component, need_time=need_time,
            )
            metrics["num_faults_in_query"] = num_faults_in_query
            metrics["unresolved"] = True
            query_results.append(metrics)

        evaluated_query_count += 1

    num_root_causes = total_matched_rc + total_unmatched_rc + total_unresolved_rc
    print(f"  official_queries: {official_query_count} | "
          f"parsed: {parsed_query_count} | "
          f"evaluated: {evaluated_query_count} | "
          f"skipped: {skipped_query_count}")
    print(f"  official_rc: {total_official_rc} | "
          f"resolved: {total_resolved_rc} | "
          f"unresolved: {total_unresolved_rc} | "
          f"predicted: {total_predicted_rc} | "
          f"matched: {total_matched_rc} | "
          f"unmatched: {total_unmatched_rc}")

    resolved_results = [r for r in query_results if not r.get("unresolved", False)]
    agg_all = _aggregate_by_applicability(query_results)
    agg_resolved = _aggregate_by_applicability(resolved_results) if resolved_results else {"n": 0}

    agg = {
        "official_query_count": official_query_count,
        "parsed_query_count": parsed_query_count,
        "evaluated_query_count": evaluated_query_count,
        "skipped_query_count": skipped_query_count,
        "official_root_cause_count": total_official_rc,
        "resolved_root_cause_count": total_resolved_rc,
        "unresolved_root_cause_count": total_unresolved_rc,
        "predicted_root_cause_count": total_predicted_rc,
        "matched_root_cause_count": total_matched_rc,
        "unmatched_root_cause_count": total_unmatched_rc,
        "n": agg_all.get("n", 0),
        "n_resolved": agg_all.get("n_resolved", 0),
        "component_metric_count": agg_all.get("component_metric_count", 0),
        "time_metric_count": agg_all.get("time_metric_count", 0),
        "joint_metric_count": agg_all.get("joint_metric_count", 0),
        "component_top1": agg_all.get("component_top1", 0),
        "component_top3": agg_all.get("component_top3", 0),
        "mrr": agg_all.get("mrr", 0),
        "avg_rank": agg_all.get("avg_rank", 0),
        "time_hit_rate": agg_all.get("time_hit_rate", 0),
        "time_mae": agg_all.get("time_mae", 0),
        "time_mae_min": agg_all.get("time_mae_min", 0),
        "time_hit_5min": agg_all.get("time_hit_5min", 0),
        "time_hit_10min": agg_all.get("time_hit_10min", 0),
        "time_hit_15min": agg_all.get("time_hit_15min", 0),
        "joint_hit_rate": agg_all.get("joint_hit_rate", 0),
        "resolved_component_top1": agg_all.get("resolved_component_top1", 0),
        "resolved_component_top3": agg_all.get("resolved_component_top3", 0),
        "resolved_mrr": agg_all.get("resolved_mrr", 0),
    }
    return agg, skipped_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='predictions.json')
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market/cloudbed-1', 'Market/cloudbed-2', 'Telecom'])
    parser.add_argument('--resample_sec', type=int, default=120)
    parser.add_argument('--skipped_csv', type=str, default='skipped_queries.csv')
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions file not found: {args.predictions}")
        print("Run phaseA_run_inference.py first to generate predictions.json")
        return

    with open(args.predictions, 'r') as f:
        all_predictions = json.load(f)

    missing_system = [p for p in all_predictions if "system" not in p]
    if missing_system:
        print(f"ERROR: {len(missing_system)} prediction entries missing 'system' field. "
              f"Re-run phaseA_run_inference.py to regenerate predictions.json.")
        return

    predictions_by_key = {(p["system"], p["query_id"]): p for p in all_predictions}
    print(f"Loaded {len(predictions_by_key)} prediction entries from {args.predictions}")

    print("=" * 60)
    print("Phase 0.8 R5: Evaluation (real ranking, query mask, separated counts)")
    print("=" * 60)

    all_results = {}
    all_skipped = []
    for sys_name in args.systems:
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")
        agg, skipped_queries = evaluate_system(sys_name, predictions_by_key, args.resample_sec)
        all_results[sys_name] = agg
        for sq in skipped_queries:
            sq["system"] = sys_name
        all_skipped.extend(skipped_queries)

    if all_skipped:
        with open(args.skipped_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "system", "reason", "details"])
            writer.writeheader()
            writer.writerows(all_skipped)
        print(f"\nSkipped queries written to {args.skipped_csv}: {len(all_skipped)} rows")

    print(f"\n{'='*115}")
    print(f"RESULTS")
    print(f"{'='*115}")
    header = (f"{'System':<18s} {'T1':>6s} {'T3':>6s} {'MRR':>7s} "
              f"{'H@5m':>6s} {'H@10m':>6s} {'H@15m':>6s} "
              f"{'MAEmin':>7s} {'Jnt':>5s} "
              f"{'#oQ':>5s} {'#eQ':>5s} {'#sQ':>5s} "
              f"{'#oRC':>5s} {'#rRC':>5s} {'#uRC':>5s}")
    print(header)
    print("-" * 115)
    for sys_name in args.systems:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        if r.get("n", 0) == 0:
            print(f"{sys_name:<18s} {'--':>6s} {'--':>6s} {'--':>7s} "
                  f"{'--':>6s} {'--':>6s} {'--':>6s} {'--':>7s} {'--':>5s} "
                  f"{'0':>5s} {'0':>5s} {'0':>5s} {'0':>5s} {'0':>5s} {'0':>5s}")
            continue
        time_mae_min = r.get('time_mae_min', 0)
        if isinstance(time_mae_min, (np.ndarray,)):
            time_mae_min = float(time_mae_min)
        if isinstance(time_mae_min, float) and np.isnan(time_mae_min):
            time_mae_min = float('inf')
        print(f"{sys_name:<18s} "
              f"{r['component_top1']:>5.1%} {r['component_top3']:>5.1%} "
              f"{r['mrr']:>7.3f} "
              f"{r.get('time_hit_5min', 0):>5.1%} "
              f"{r.get('time_hit_10min', 0):>5.1%} "
              f"{r.get('time_hit_15min', 0):>5.1%} "
              f"{time_mae_min if np.isfinite(time_mae_min) else 0.0:>7.1f} "
              f"{r['joint_hit_rate']:>4.1%} "
              f"{r.get('official_query_count', 0):>5d} "
              f"{r.get('evaluated_query_count', 0):>5d} "
              f"{r.get('skipped_query_count', 0):>5d} "
              f"{r.get('official_root_cause_count', 0):>5d} "
              f"{r.get('resolved_root_cause_count', 0):>5d} "
              f"{r.get('unresolved_root_cause_count', 0):>5d}")
    print("-" * 115)
    print(f"Legend: T1=Top-1, T3=Top-3, H@5m=Hit@5min, Jnt=Joint Hit, "
          f"#oQ=official queries, #eQ=evaluated, #sQ=skipped, "
          f"#oRC=official root-causes, #rRC=resolved, #uRC=unresolved")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
