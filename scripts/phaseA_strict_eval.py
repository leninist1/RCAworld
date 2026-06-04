"""Phase A-Hardening P0.6: GT-isolated Query Episode Protocol evaluation.

GT isolation (P0.1):
  - InferenceQuery: observation metadata ONLY. NEVER contains GT.
  - EvalTarget: ground truth ONLY, read from separate code path.
  - Episode builder NEVER touches EvalTarget.
  - Model inference NEVER touches EvalTarget or record.csv.

Fixes implemented vs P0.5:
  P0.2: Time Hit constrained to query window (onset_score respects qw_mask).
  P0.3: Multi-fault queries decoded via Non-Maximum Suppression.
  P0.4: Adapter loads explicit date range from query dates.
  P1.1: UTC+8 aware datetimes throughout.
  P1.4: Sliding window tail coverage (last window always included).
"""

import os, sys, time, argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import h5py
import jax, jax.numpy as jnp
from flax.training import train_state as flax_train_state
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

from foundation.models import RCAWorldFoundation
from foundation.evaluation.strict_eval import (
    compute_joint_scores, evaluate_joint, aggregate_metrics,
    JointScores,
)
from foundation.evaluation.query_parser import (
    parse_query_csv, InferenceQuery, EvalTarget,
    format_episode_window, OPENRCA_TZ,
)
from foundation.evaluation.episode_builder import (
    build_telemetry_tensor, KPI_TO_OB, assign_entity_type,
)
from foundation.evaluation.leakage_checks import (
    validate_labels_in_telemetry_range,
    validate_dataset_coverage,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phaseA", "best")
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"
MAX_D = 16
WS = 23
DET_DIM = 256

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
    """Create adapter with explicit date set (P0.4: not max_days, but dates)."""
    from foundation.adapters import (
        OpenRCABankAdapter, OpenRCATelecomAdapter, OpenRCAMarketAdapter,
    )
    cls_map = {
        "OpenRCABankAdapter": lambda: OpenRCABankAdapter(
            max_days=len(include_dates) if include_dates else 30,
            max_container_events=500000, max_container_rows=5000000),
        "OpenRCATelecomAdapter": lambda: OpenRCATelecomAdapter(
            max_days=len(include_dates) if include_dates else 30,
            max_container_timestamps=5000),
        "OpenRCAMarketAdapter": lambda: OpenRCAMarketAdapter(
            max_days=len(include_dates) if include_dates else 30,
            max_container_events=500000, max_container_rows=5000000),
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


def _match_eval_label(eval_target: EvalTarget, labels: list,
                       timestamps: np.ndarray, entity_ids: list) -> list:
    """Match EvalTarget root causes to record.csv labels. EVALUATION ONLY."""
    qw_idx = 0  # placeholder; actual query window from eval_target context
    matched = []
    for rc_idx, (comp, dt_str, reason, tolerance) in enumerate(eval_target.root_causes):
        # Strategy 1: match by datetime from scoring_points
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

        # Strategy 2: match by component name
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


def compute_residuals_and_latents(model, params, tensor_norm, type_idx,
                                  obs_mask, use_posterior=False, ws=WS):
    """Sliding-window inference with tail coverage (P1.4)."""
    T, N, D = tensor_norm.shape
    resid_acc = np.zeros((T, N), dtype=np.float32)
    resid_cnt = np.zeros((T, N), dtype=np.float32)
    h_acc = np.zeros((T, N, DET_DIM), dtype=np.float32)
    h_cnt = np.zeros((T, N), dtype=np.float32)
    coverage = np.zeros(T, dtype=bool)

    stride = max(1, ws // 2)
    starts = list(range(0, T - ws, stride))
    # P1.4: always include the last window
    last_start = max(0, T - ws - 1)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    windows, pos = [], []
    for start in starts:
        end = start + ws + 1
        if end > T:
            end = T
        windows.append(tensor_norm[start:end])
        pos.append((start, end))

    if not windows:
        return resid_acc, h_acc, coverage

    windows = np.stack(windows, axis=0)
    t_idx_j = jnp.array(type_idx)
    o_mask_j = np.array(obs_mask)
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), 8):
        bx = jnp.array(windows[i:i + 8])
        out = model.apply(
            {"params": params}, bx, t_idx_j, rng,
            obs_mask=o_mask_j, use_posterior=use_posterior)
        res_b = np.array(out["residual"])
        h_b = np.array(out["h_seq"])

        for j, (s, e) in enumerate(pos[i:i + 8]):
            ns = min(res_b.shape[1], e - s)
            for tt in range(ns):
                tg = s + tt + 1
                if tg < T:
                    resid_acc[tg] += res_b[j, tt]
                    resid_cnt[tg] += 1.0
                    coverage[tg] = True
            nh = min(h_b.shape[1], e - s)
            for tt in range(nh):
                tg = s + tt + 1
                if tg < T:
                    h_acc[tg] += h_b[j, tt]
                    h_cnt[tg] += 1.0

    for tt in range(T):
        for n in range(N):
            if resid_cnt[tt, n] > 0:
                resid_acc[tt, n] /= resid_cnt[tt, n]
            if h_cnt[tt, n] > 0:
                h_acc[tt, n] /= h_cnt[tt, n]

    return np.nan_to_num(resid_acc, nan=0.0), h_acc, coverage


def evaluate_system(sys_name, model, state, ob_mean, ob_std, use_posterior,
                    scoring_method, all_zero_type, burn_in_min, resample_sec):
    cfg = SYSTEM_CONFIGS[sys_name]
    data_dir = cfg["data_dir"]
    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return {"n": 0}

    query_path = os.path.join(data_dir, "query.csv")
    if not os.path.exists(query_path):
        print(f"  query.csv not found: {query_path}")
        return {"n": 0}

    # Parse query.csv → separated InferenceQuery + EvalTarget (P0.1)
    inference_queries, eval_targets = parse_query_csv(query_path)
    print(f"  Parsed {len(inference_queries)} queries from query.csv")

    total_official_queries = len(inference_queries)

    # Compute date range needed from queries
    all_dates = sorted(set(q.window_start.date() for q in inference_queries))
    min_date = all_dates[0]
    max_date = all_dates[-1]
    print(f"  Query date range: {min_date} — {max_date} ({len(all_dates)} unique dates)")

    # P0.4: compute exact date directories needed
    telemetry_dir = os.path.join(data_dir, "telemetry")
    available_dates = sorted([d for d in os.listdir(telemetry_dir)
                              if d.startswith("20")]) if os.path.exists(telemetry_dir) else []
    min_date_str = min_date.strftime("%Y_%m_%d")
    max_date_str = max_date.strftime("%Y_%m_%d")
    needed_dates = [d for d in available_dates
                    if min_date_str <= d <= max_date_str]
    print(f"  Available telemetry dates: {len(available_dates)}; "
          f"needed range: {len(needed_dates)}")

    # P0.4: pass needed_dates to adapter
    adapter = _make_adapter(cfg["adapter_cls"], needed_dates)
    all_entities = adapter.discover_entities(data_dir)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities
    entity_ids = [e.entity_id for e in cont_entities]
    N = len(entity_ids)
    print(f"  Entities: {N}")

    t0 = time.time()
    all_events = adapter.extract_events(data_dir, cont_entities)
    event_load_time = time.time() - t0
    print(f"  Loaded {len(all_events)} events in {event_load_time:.1f}s")

    if not all_events:
        print("  No events loaded!")
        return {"n": 0}

    t0 = time.time()
    global_tensor, global_obs_mask, global_ts = build_telemetry_tensor(
        events=all_events, entity_ids=entity_ids, kpi_to_ob=KPI_TO_OB,
        max_ob_features=8, resample_interval_sec=resample_sec,
    )
    T_global = global_tensor.shape[0]
    if T_global == 0:
        print("  Empty global tensor!")
        return {"n": 0}

    t_min_dt = datetime.fromtimestamp(global_ts[0], tz=OPENRCA_TZ)
    t_max_dt = datetime.fromtimestamp(global_ts[-1], tz=OPENRCA_TZ)
    print(f"  Global tensor: {global_tensor.shape} [{t_min_dt} — {t_max_dt}]")

    # Entity types
    type_str_to_idx = {"container": 0, "database": 1, "middleware": 2, "host": 0}
    type_strs = [assign_entity_type(eid) for eid in entity_ids] if not all_zero_type \
                else ["container"] * N
    type_indices = np.array(
        [type_str_to_idx.get(t, 0) if not all_zero_type else 0 for t in type_strs],
        dtype=np.int32,
    )

    global_obs_mask_2d = (global_obs_mask.sum(axis=0) > 0).astype(np.float32)
    global_obs_mask_model = np.pad(global_obs_mask_2d, ((0, 0), (0, MAX_D - 8)), mode='constant')

    global_tensor_norm = np.nan_to_num(
        (global_tensor - ob_mean) / (ob_std + 1e-6), nan=0.0)
    global_tensor_pad = np.pad(global_tensor_norm,
                               ((0, 0), (0, 0), (0, MAX_D - 8)), mode='constant')

    # Load labels from record.csv (EVALUATION ONLY)
    labels = extract_labels_from_record(data_dir, entity_ids, adapter)

    # Coverage report
    coverage = validate_dataset_coverage(labels, global_ts, entity_ids, sys_name)
    print(f"  Coverage: entity={coverage['entity_match']}/{coverage['total_labels']} "
          f"time={coverage['time_in_range']}/{coverage['total_labels']}")
    violations = validate_labels_in_telemetry_range(labels, global_ts, sys_name)
    if violations:
        for v in violations[:3]:
            print(f"  WARNING: {v}")

    # Process each query as independent Episode
    query_results = []
    total_time = 0.0
    processed = 0
    skipped_no_data = 0
    skipped_no_label = 0

    for qi, iq in enumerate(inference_queries):
        qid = iq.query_id
        if qid not in eval_targets:
            skipped_no_label += 1
            continue

        # Determine time range for this episode
        data_start_ts = (iq.window_start - timedelta(minutes=burn_in_min)).timestamp()
        data_end_ts = iq.window_end.timestamp()

        # Find global tensor indices
        idx_slice = np.where((global_ts >= data_start_ts) & (global_ts <= data_end_ts))[0]
        if len(idx_slice) < WS + 2:
            skipped_no_data += 1
            continue

        s_start = idx_slice[0]
        s_end = idx_slice[-1] + 1
        tensor_sub = global_tensor_pad[s_start:s_end]
        ts_sub = global_ts[s_start:s_end]
        T_sub = tensor_sub.shape[0]

        obs_mask_sub_2d = (global_obs_mask[s_start:s_end].sum(axis=0) > 0).astype(np.float32)
        obs_mask_sub_2d = np.pad(obs_mask_sub_2d, ((0, 0), (0, MAX_D - 8)), mode='constant')

        # Inference (prior-only)
        t_inf = time.time()
        residuals, h_states, coverage_mask = compute_residuals_and_latents(
            model, state.params, tensor_sub, type_indices,
            obs_mask_sub_2d, use_posterior=use_posterior, ws=WS)
        inference_time = time.time() - t_inf
        total_time += inference_time

        if residuals.shape[0] == 0:
            skipped_no_data += 1
            continue

        # Burn-in mask and query window mask
        qw_start_ts = iq.window_start.timestamp()
        qw_end_ts = iq.window_end.timestamp()
        burn_in_mask = ts_sub < qw_start_ts
        qw_mask = (ts_sub >= qw_start_ts) & (ts_sub <= qw_end_ts)

        # Query window coverage check
        qw_coverage = coverage_mask[qw_mask]
        if qw_coverage.any():
            qw_cov_pct = qw_coverage.mean()
        else:
            qw_cov_pct = 0.0

        # Compute joint scores
        joint = compute_joint_scores(
            residuals=residuals, latents=h_states,
            model_onset_scores=None, method=scoring_method,
            burn_in_mask=burn_in_mask,
            lambda_residual=1.0, lambda_shift=0.5, lambda_early=0.3,
            temporal_smooth_window=3,
        )
        joint.query_window_mask = qw_mask

        # Match EvalTarget to record.csv labels (EVALUATION ONLY)
        eval_tgt = eval_targets[qid]
        matched_labels = _match_eval_label(eval_tgt, labels, ts_sub, entity_ids)
        if not matched_labels:
            skipped_no_label += 1
            continue

        # Evaluate per root cause
        for rc_idx, lb in enumerate(matched_labels):
            if rc_idx >= len(eval_tgt.root_causes):
                break
            _, _, _, tolerance = eval_tgt.root_causes[rc_idx]
            tolerance_steps = max(1, (tolerance * 60) // resample_sec)
            metric = evaluate_joint(
                joint=joint,
                gt_component_idx=lb["component_idx"],
                gt_onset_ts=lb["timestamp"],
                timestamps=ts_sub,
                onset_tolerance_steps=tolerance_steps,
            )
            # Convert step error to minutes
            metric["time_error_min"] = metric["time_error"] * resample_sec / 60.0
            metric["query_window_coverage"] = qw_cov_pct
            query_results.append(metric)
        processed += 1

        if (qi + 1) % 30 == 0:
            print(f"    {qi + 1}/{len(inference_queries)} queries... "
                  f"(avg inf: {total_time/max(1,processed):.2f}s)")

    skipped = skipped_no_data + skipped_no_label
    print(f"  Official queries: {total_official_queries} | Evaluated: {processed} | "
          f"Skipped: {skipped} (no_data: {skipped_no_data}, no_label: {skipped_no_label})")
    if processed > 0:
        print(f"  Avg inference: {total_time/processed:.2f}s | "
              f"Time MAE in min: {resample_sec/60.0}s per step")

    agg = aggregate_metrics(query_results)
    agg["total_official"] = total_official_queries
    agg["skipped"] = skipped
    return agg


def extract_labels_from_record(data_dir, entity_ids, adapter):
    return _extract_labels(data_dir, entity_ids, adapter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_posterior', action='store_true')
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market/cloudbed-1', 'Market/cloudbed-2', 'Telecom'])
    parser.add_argument('--method', type=str, default='calibrated',
                        choices=['residual_only', 'calibrated', 'legacy_per_entity_max',
                                 'onset_head'])
    parser.add_argument('--all_zero_type', action='store_true')
    parser.add_argument('--burn_in_min', type=int, default=60)
    parser.add_argument('--resample_sec', type=int, default=120)
    args = parser.parse_args()

    use_posterior = args.ablation_posterior
    mode_str = "POSTERIOR (ablation)" if use_posterior else "PRIOR-ONLY"
    type_str = "all-zero" if args.all_zero_type else "heuristic"

    if args.method == "onset_head" and not use_posterior:
        print("NOTE: onset_head requires posterior; falling back to calibrated.")
        args.method = "calibrated"

    print("=" * 60)
    print(f"Phase A-Hardening P0.6: GT-isolated Query Episode Protocol ({mode_str})")
    print(f"Scoring: {args.method} | Type: {type_str} | "
          f"Resample: {args.resample_sec}s | Burn-in: {args.burn_in_min}min")
    print("=" * 60)

    model = RCAWorldFoundation(
        common_dim=128, max_obs_dim=MAX_D, num_entity_types=3,
        det_dim=DET_DIM, stoch_dim=32, stoch_classes=32,
        use_onset_head=True, use_component_head=True,
    )
    ex_x = jnp.zeros((1, 24, 10, MAX_D))
    rng = jax.random.PRNGKey(0)
    init_rng, _ = jax.random.split(rng)
    v = model.init(init_rng, ex_x, jnp.zeros(10, dtype=jnp.int32),
                   jax.random.PRNGKey(1))
    tx = __import__('optax').adam(1e-3)
    state = flax_train_state.TrainState.create(
        apply_fn=model.apply, params=v["params"], tx=tx)

    if os.path.exists(CKPT_PATH):
        state = state.replace(
            params=ocp.PyTreeCheckpointer().restore(CKPT_PATH)["params"])
        print(f"Loaded checkpoint: {CKPT_PATH}")

    with h5py.File(DATA_PATH, 'r') as f:
        ob_mean = f['stats/node_mean'][:].mean(axis=1).reshape(1, 1, 8)
        ob_std = f['stats/node_std'][:].mean(axis=1).reshape(1, 1, 8)

    all_results = {}
    for sys_name in args.systems:
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")
        t0 = time.time()
        agg = evaluate_system(
            sys_name=sys_name, model=model, state=state,
            ob_mean=ob_mean, ob_std=ob_std,
            use_posterior=use_posterior,
            scoring_method=args.method,
            all_zero_type=args.all_zero_type,
            burn_in_min=args.burn_in_min,
            resample_sec=args.resample_sec,
        )
        elapsed = time.time() - t0
        all_results[sys_name] = agg
        print(f"  Total time: {elapsed:.1f}s")

    print(f"\n{'='*80}")
    print(f"RESULTS ({mode_str}, {args.method}, {type_str} type)")
    print(f"{'='*80}")
    header = (f"{'System':<18s} {'Comp T1':>8s} {'Comp T3':>8s} "
              f"{'MRR':>8s} {'AvgR':>6s} {'TimeH5':>7s} "
              f"{'MAEmin':>7s} {'JntHit':>7s} {'Ev/N':>8s}")
    print(header)
    print("-" * 80)
    for sys_name in args.systems:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        if r.get("n", 0) == 0:
            print(f"{sys_name:<18s} {'--':>8s} {'--':>8s} "
                  f"{'--':>8s} {'--':>6s} {'--':>7s} {'--':>7s} {'--':>7s} "
                  f"{'0/0':>8s}")
            continue
        time_mae_min = r.get('time_mae', 0) * args.resample_sec / 60.0
        total = r.get('total_official', 0)
        skipped = r.get('skipped', 0)
        print(f"{sys_name:<18s} "
              f"{r['component_top1']:>7.1%} {r['component_top3']:>7.1%} "
              f"{r['mrr']:>8.3f} {r['avg_rank']:>6.2f} "
              f"{r['time_hit_rate']:>7.1%} {time_mae_min:>7.1f} "
              f"{r['joint_hit_rate']:>7.1%} "
              f"{r['n']:>4d}/{total:>4d}")
    print("-" * 80)
    print(f"Time MAE in minutes (resample={args.resample_sec}s). "
          f"TimeHit@5min for tolerance=3 steps.")
    print(f"\nDone. Mode: {mode_str} | Method: {args.method} | Type: {type_str}")


if __name__ == "__main__":
    main()
