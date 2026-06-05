"""Phase 0.8 Round 4: Evaluation script — reads predictions.json + GT.

This script reads:
  - predictions.json (from phaseA_run_inference.py)
  - query.csv scoring_points column (via load_eval_targets)
  - record.csv (for label matching)

It evaluates multi-fault queries with one-to-one prediction/target matching.
"""

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

from foundation.evaluation.strict_eval import aggregate_metrics

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


def _match_eval_targets(eval_target, labels):
    """Match EvalTarget root causes to record.csv labels. EVALUATION ONLY."""
    matched = []
    used_label_indices = set()

    for comp, dt_str, reason, tolerance in eval_target.root_causes:
        found = None
        found_idx = None
        if dt_str:
            try:
                gt_dt = dt.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                gt_ts = gt_dt.replace(tzinfo=OPENRCA_TZ).timestamp()
                for lb_idx, lb in enumerate(labels):
                    if lb_idx in used_label_indices:
                        continue
                    if abs(lb["timestamp"] - gt_ts) < 300:
                        found = lb
                        found_idx = lb_idx
                        break
            except (ValueError, OSError):
                pass

        if found is None and comp:
            cl = comp.lower().replace("_", "").replace("-", "").replace(" ", "")
            for lb_idx, lb in enumerate(labels):
                if lb_idx in used_label_indices:
                    continue
                lb_comp = lb["component"].lower().replace("_", "").replace("-", "").replace(" ", "")
                if cl and lb_comp and (cl in lb_comp or lb_comp in cl):
                    found = lb
                    found_idx = lb_idx
                    break

        if found is not None:
            used_label_indices.add(found_idx)
            matched.append({
                "component": found["component"],
                "component_idx": found["component_idx"],
                "timestamp": found["timestamp"],
                "reason": found["reason"],
                "tolerance": tolerance,
                "target_component": comp,
                "target_datetime": dt_str,
                "target_reason": reason,
            })

    return matched


def _matching_cost(prediction, target, component_mismatch_penalty=10000.0):
    cost = 0.0
    if prediction.get("component_idx", -1) != target.get("component_idx", -1):
        cost += component_mismatch_penalty

    pred_ts = prediction.get("timestamp")
    target_ts = target.get("timestamp")
    if pred_ts is None or target_ts is None:
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


def _evaluate_prediction_target_pair(prediction, target, entity_ids):
    gt_c = target["component_idx"]
    gt_ts = target["timestamp"]
    tolerance = target["tolerance"]

    comp_hit = prediction is not None and prediction.get("component_idx", -1) == gt_c

    time_hit = False
    time_error_min = float('inf')
    if prediction is not None and prediction.get("timestamp") is not None:
        time_error_sec = abs(prediction["timestamp"] - gt_ts)
        time_error_min = time_error_sec / 60.0
        time_hit = time_error_sec <= tolerance * 60

    comp_rank = 1 if comp_hit else len(entity_ids)
    joint_hit = comp_hit and time_hit

    return {
        "component_rank": comp_rank,
        "component_top1": comp_rank == 1,
        "component_top3": comp_rank <= 3,
        "time_error": float('nan') if time_error_min == float('inf') else time_error_min,
        "time_error_min": time_error_min if time_error_min != float('inf') else float('nan'),
        "time_hit": time_hit,
        "time_hit_5min": bool(time_error_min <= 5.0),
        "time_hit_10min": bool(time_error_min <= 10.0),
        "time_hit_15min": bool(time_error_min <= 15.0),
        "joint_component_hit": comp_hit,
        "joint_time_hit": time_hit,
        "joint_hit": joint_hit,
        "reciprocal_rank": 1.0 / max(1, comp_rank),
    }


def evaluate_system(sys_name, predictions_by_id, resample_sec):
    """Evaluate predictions for one system with one-to-one prediction matching."""
    cfg = SYSTEM_CONFIGS[sys_name]
    data_dir = cfg["data_dir"]
    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return {"n": 0}

    query_path = os.path.join(data_dir, "query.csv")
    if not os.path.exists(query_path):
        print(f"  query.csv not found: {query_path}")
        return {"n": 0}

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

    labels = _extract_labels(data_dir, entity_ids, adapter)
    print(f"  Loaded {len(labels)} labels from record.csv")

    query_results = []
    processed = 0
    skipped_no_label = 0

    for qid, eval_tgt in eval_targets.items():
        pred_entry = predictions_by_id.get(qid, {"query_id": qid, "predictions": []})
        raw_predictions = pred_entry.get("predictions", [])

        targets = _match_eval_targets(eval_tgt, labels)
        if not targets:
            skipped_no_label += 1
            continue

        predictions = [_parse_prediction(prediction, entity_ids) for prediction in raw_predictions]
        matched_pairs, _, unmatched_targets = match_predictions_to_targets(predictions, targets)

        num_faults_in_query = len(targets)
        for pair in matched_pairs:
            metrics = _evaluate_prediction_target_pair(pair["prediction"], pair["target"], entity_ids)
            metrics["num_faults_in_query"] = num_faults_in_query
            metrics["query_window_coverage"] = 1.0
            query_results.append(metrics)

        for target in unmatched_targets:
            metrics = _evaluate_prediction_target_pair(None, target, entity_ids)
            metrics["num_faults_in_query"] = num_faults_in_query
            metrics["query_window_coverage"] = 1.0
            query_results.append(metrics)

        processed += 1

    num_root_causes = len(query_results)
    skipped = skipped_no_label
    print(f"  Evaluated queries: {processed} | "
          f"Total root-causes: {num_root_causes} | "
          f"Skipped: {skipped} (no_label: {skipped_no_label})")

    agg = aggregate_metrics(query_results)
    agg["total_official"] = len(eval_targets)
    agg["n_queries_evaluated"] = processed
    agg["n_queries_skipped"] = skipped
    agg["n_queries_skipped_no_pred"] = 0
    agg["n_queries_skipped_no_label"] = skipped_no_label
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='predictions.json')
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market/cloudbed-1', 'Market/cloudbed-2', 'Telecom'])
    parser.add_argument('--resample_sec', type=int, default=120)
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions file not found: {args.predictions}")
        print("Run phaseA_run_inference.py first to generate predictions.json")
        return

    with open(args.predictions, 'r') as f:
        all_predictions = json.load(f)

    predictions_by_id = {p["query_id"]: p for p in all_predictions}
    print(f"Loaded {len(predictions_by_id)} prediction entries from {args.predictions}")

    print("=" * 60)
    print("Phase 0.8 R4: Evaluation (one-to-one matching over predictions.json + GT)")
    print("=" * 60)

    all_results = {}
    for sys_name in args.systems:
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")
        agg = evaluate_system(sys_name, predictions_by_id, args.resample_sec)
        all_results[sys_name] = agg

    print(f"\n{'='*100}")
    print(f"RESULTS")
    print(f"{'='*100}")
    header = (f"{'System':<18s} {'T1':>6s} {'T3':>6s} {'MRR':>7s} "
              f"{'H@5m':>6s} {'H@10m':>6s} {'H@15m':>6s} "
              f"{'MAEmin':>7s} {'Jnt':>5s} {'#root':>6s} {'#q':>5s}")
    print(header)
    print("-" * 100)
    for sys_name in args.systems:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        if r.get("n", 0) == 0:
            print(f"{sys_name:<18s} {'--':>6s} {'--':>6s} {'--':>7s} "
                  f"{'--':>6s} {'--':>6s} {'--':>6s} {'--':>7s} {'--':>5s} "
                  f"{'0':>6s} {'0':>5s}")
            continue
        time_mae_min = r.get('time_mae_min', r.get('time_mae', 0) * args.resample_sec / 60.0)
        if isinstance(time_mae_min, (np.ndarray,)):
            time_mae_min = float(time_mae_min)
        total_q = r.get('total_official', 0)
        n_q_eval = r.get('n_queries_evaluated', r.get('n', 0))
        print(f"{sys_name:<18s} "
              f"{r['component_top1']:>5.1%} {r['component_top3']:>5.1%} "
              f"{r['mrr']:>7.3f} "
              f"{r.get('time_hit_5min', 0):>5.1%} "
              f"{r.get('time_hit_10min', 0):>5.1%} "
              f"{r.get('time_hit_15min', 0):>5.1%} "
              f"{time_mae_min:>7.1f} "
              f"{r['joint_hit_rate']:>4.1%} "
              f"{r['n']:>6d} "
              f"{total_q:>5d}")
    print("-" * 100)
    print(f"Legend: T1=Top-1, T3=Top-3, H@5m=Hit@5min, Jnt=Joint Hit, "
          f"#root=root-cause instances, #q=official queries")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
