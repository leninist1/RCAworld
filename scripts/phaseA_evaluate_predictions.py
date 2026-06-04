"""Phase 0.8 Round 1: Evaluation-only script — reads predictions.json + GT.

This script reads:
  - predictions.json (from phaseA_run_inference.py)
  - query.csv scoring_points column (via load_eval_targets)
  - record.csv (for label matching)

It supports single-fault evaluation only (matching the current strict_eval logic).
Do NOT process multi-fault matching in this round.
"""

import json
import os
import sys
import argparse

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


def _match_eval_label(eval_target, labels):
    """Match EvalTarget root causes to record.csv labels. EVALUATION ONLY."""
    matched = []
    for rc_idx, (comp, dt_str, reason, tolerance) in enumerate(eval_target.root_causes):
        found = None
        if dt_str:
            try:
                from datetime import datetime as dt
                gt_dt = dt.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                gt_ts = gt_dt.replace(tzinfo=OPENRCA_TZ).timestamp()
                for lb in labels:
                    if abs(lb["timestamp"] - gt_ts) < 300:
                        found = lb
                        break
            except (ValueError, OSError):
                pass

        if found is None and comp:
            cl = comp.lower().replace("_", "").replace("-", "").replace(" ", "")
            for lb in labels:
                lb_comp = lb["component"].lower().replace("_", "").replace("-", "").replace(" ", "")
                if cl and lb_comp and (cl in lb_comp or lb_comp in cl):
                    found = lb
                    break

        if found is not None:
            matched.append(found)

    return matched


def evaluate_system(sys_name, predictions_by_id, resample_sec):
    """Evaluate predictions for one system — single-fault only."""
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
    skipped_no_pred = 0
    skipped_no_label = 0

    for qid, eval_tgt in eval_targets.items():
        if qid not in predictions_by_id:
            skipped_no_pred += 1
            continue

        pred_entry = predictions_by_id[qid]
        predictions = pred_entry.get("predictions", [])

        matched_labels = _match_eval_label(eval_tgt, labels)
        if not matched_labels:
            skipped_no_label += 1
            continue

        gt_fault_count = len(eval_tgt.root_causes)
        if gt_fault_count > 1:
            skipped_no_label += 1
            continue

        # Single-fault evaluation
        for rc_idx, lb in enumerate(matched_labels):
            if rc_idx >= len(eval_tgt.root_causes):
                break
            _, _, _, tolerance = eval_tgt.root_causes[rc_idx]
            gt_c = lb["component_idx"]
            gt_ts = lb["timestamp"]

            # Determine predicted component and time
            if rc_idx < len(predictions):
                pred_component = predictions[rc_idx].get("component", "")
                pred_dt_str = predictions[rc_idx].get("datetime", "")
            else:
                pred_component = ""
                pred_dt_str = ""

            # Component match: check if predicted component matches GT
            comp_hit = False
            if pred_component and gt_c < len(entity_ids):
                comp_hit = pred_component == entity_ids[gt_c]
                if not comp_hit:
                    cl = pred_component.lower().replace("_", "").replace("-", "").replace(" ", "")
                    gl = entity_ids[gt_c].lower().replace("_", "").replace("-", "").replace(" ", "")
                    comp_hit = bool(cl and gl and (cl in gl or gl in cl))

            # Time match: check if predicted datetime is within tolerance
            time_hit = False
            time_error_min = float('inf')
            if pred_dt_str:
                try:
                    from datetime import datetime as dt
                    pred_dt = dt.strptime(pred_dt_str, "%Y-%m-%d %H:%M:%S")
                    pred_ts = pred_dt.replace(tzinfo=OPENRCA_TZ).timestamp()
                    time_error_sec = abs(pred_ts - gt_ts)
                    time_error_min = time_error_sec / 60.0
                    time_hit = time_error_sec <= tolerance * 60
                except (ValueError, OSError):
                    pass

            # Component rank: 1 if match else len(entities) worst case
            comp_rank = 1 if comp_hit else len(entity_ids)

            joint_hit = comp_hit and time_hit

            query_results.append({
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
                "num_faults_in_query": gt_fault_count,
                "query_window_coverage": 1.0,
            })
        processed += 1

    num_root_causes = len(query_results)
    skipped = skipped_no_pred + skipped_no_label
    print(f"  Evaluated queries: {processed} | "
          f"Total root-causes: {num_root_causes} | "
          f"Skipped: {skipped} (no_pred: {skipped_no_pred}, no_label: {skipped_no_label})")

    agg = aggregate_metrics(query_results)
    agg["total_official"] = len(eval_targets)
    agg["n_queries_evaluated"] = processed
    agg["n_queries_skipped"] = skipped
    agg["n_queries_skipped_no_pred"] = skipped_no_pred
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
    print("Phase 0.8 R1: Evaluation-only (reads predictions.json + scoring_points + record.csv)")
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
