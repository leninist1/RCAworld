"""Phase A-Hardening P0.5: Query Episode Protocol — strict zero-shot evaluation.

Rules:
  1. Each query.csv row -> independent Episode with its own observation window.
  2. Telemetry loaded once globally, then sliced per-query window + burn-in.
  3. GT labels ONLY used in metric computation, NEVER during inference.
  4. prior-only as primary; posterior as --ablation.
  5. OpenRCA data NEVER used for training.
  6. Real observation mask from actual KPI availability (counts > 0).
  7. Fixed-time-grid resampling (default 60s per step).
  8. Overlapping window residuals averaged, not overwritten.
  9. Cross-entity calibrated scoring (MAD-based z-scores).
  10. query.csv observation windows are the only legal data slice.

Reports:
  System | Comp Top-1 | Top-3 | MRR | Time MAE | Time Hit@5min | Joint Hit
"""

import os, sys, time, argparse
from datetime import datetime, timedelta
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
    parse_query_csv, ParsedQuery,
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
WS = 23  # RSSM window size
DET_DIM = 256


SYSTEM_CONFIGS = {
    "Bank": {
        "data_dir": os.path.join(OPENRCA, "Bank", "Bank"),
        "adapter_cls": "OpenRCABankAdapter",
    },
    "Market/cloudbed-1": {
        "data_dir": os.path.join(OPENRCA, "Market", "Market", "cloudbed-1"),
        "adapter_cls": "OpenRCAMarketAdapter",
    },
    "Market/cloudbed-2": {
        "data_dir": os.path.join(OPENRCA, "Market", "Market", "cloudbed-2"),
        "adapter_cls": "OpenRCAMarketAdapter",
    },
    "Telecom": {
        "data_dir": os.path.join(OPENRCA, "Telecom", "Telecom"),
        "adapter_cls": "OpenRCATelecomAdapter",
    },
}


def _make_adapter(adapter_cls_name, max_days):
    from foundation.adapters import (
        OpenRCABankAdapter, OpenRCATelecomAdapter, OpenRCAMarketAdapter,
    )
    cls_map = {
        "OpenRCABankAdapter": lambda: OpenRCABankAdapter(
            max_days=max_days, max_container_events=500000,
            max_container_rows=5000000),
        "OpenRCATelecomAdapter": lambda: OpenRCATelecomAdapter(
            max_days=max_days, max_container_timestamps=5000),
        "OpenRCAMarketAdapter": lambda: OpenRCAMarketAdapter(
            max_days=max_days, max_container_events=500000,
            max_container_rows=5000000),
    }
    return cls_map[adapter_cls_name]()


def _extract_labels(data_dir, entity_ids, adapter):
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


def compute_residuals_and_latents(model, params, tensor_norm, type_idx,
                                  obs_mask, use_posterior=False, ws=WS):
    """Sliding-window inference: residuals [T,N] and h_states [T,N,D].

    FIXED: overlapping window residuals averaged, not overwritten.
    """
    T, N, D = tensor_norm.shape
    resid_acc = np.zeros((T, N), dtype=np.float32)
    resid_cnt = np.zeros((T, N), dtype=np.float32)
    h_acc = np.zeros((T, N, DET_DIM), dtype=np.float32)
    h_cnt = np.zeros((T, N), dtype=np.float32)

    windows, pos = [], []
    for start in range(0, T - ws, max(1, ws // 2)):
        end = start + ws + 1
        if end > T:
            break
        windows.append(tensor_norm[start:end])
        pos.append((start, end))
    if not windows:
        return resid_acc, h_acc

    windows = np.stack(windows, axis=0)
    t_idx_j = jnp.array(type_idx)
    o_mask_j = jnp.array(obs_mask)
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

    return np.nan_to_num(resid_acc, nan=0.0), h_acc


def _find_matching_label(query, labels, timestamps, entity_ids):
    """Match a ParsedQuery to the corresponding GT label from record.csv.

    Strategy (priority order):
      1. Match by GT datetime from scoring_points (most precise).
      2. Match by GT component name AND query observation window.
      3. Match by GT component name only.
      4. Match any label within the query observation window (fallback for
         time-only and reason-only queries that lack component in scoring).
    """
    gt_comp = query.gt_component.lower().strip() if query.gt_component else ""
    gt_time_str = query.gt_datetime.strip() if query.gt_datetime else ""
    gt_reason = query.gt_reason.lower().strip() if query.gt_reason else ""
    qw_s = query.window_start.timestamp()
    qw_e = query.window_end.timestamp()

    # Strategy 1: exact datetime match from scoring_points
    if gt_time_str:
        try:
            gt_dt = datetime.strptime(gt_time_str, "%Y-%m-%d %H:%M:%S")
            gt_ts = gt_dt.timestamp()
            for lb in labels:
                if abs(lb["timestamp"] - gt_ts) < 300:
                    return lb
        except (ValueError, OSError):
            pass

    # Strategy 2: component match within query window
    if gt_comp:
        for lb in labels:
            if lb["timestamp"] < qw_s or lb["timestamp"] > qw_e:
                continue
            lb_comp = lb["component"].lower().replace("_", "").replace("-", "").replace(" ", "")
            gt_comp_norm = gt_comp.lower().replace("_", "").replace("-", "").replace(" ", "")
            if gt_comp_norm and lb_comp and (gt_comp_norm in lb_comp or lb_comp in gt_comp_norm):
                return lb

    # Strategy 3: component match (any time)
    if gt_comp:
        for lb in labels:
            lb_comp = lb["component"].lower().replace("_", "").replace("-", "").replace(" ", "")
            gt_comp_norm = gt_comp.lower().replace("_", "").replace("-", "").replace(" ", "")
            if gt_comp_norm and lb_comp and (gt_comp_norm in lb_comp or lb_comp in gt_comp_norm):
                return lb

    # Strategy 4: match any label within query observation window (fallback)
    # For time-only and reason-only queries without component in scoring_points
    for lb in labels:
        if qw_s <= lb["timestamp"] <= qw_e:
            return lb

    return None


def evaluate_system(sys_name, model, state, ob_mean, ob_std, use_posterior,
                    scoring_method, all_zero_type, burn_in_min, resample_sec,
                    max_days):
    cfg = SYSTEM_CONFIGS[sys_name]
    data_dir = cfg["data_dir"]
    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return {"n": 0}

    query_path = os.path.join(data_dir, "query.csv")
    if not os.path.exists(query_path):
        print(f"  query.csv not found: {query_path}")
        return {"n": 0}

    queries = parse_query_csv(query_path)
    print(f"  Parsed {len(queries)} queries from query.csv")

    # Determine covered date range from queries
    all_dates = sorted(set(q.window_start.date() for q in queries))
    min_date = all_dates[0]
    max_date = all_dates[-1]
    print(f"  Query date range: {min_date} — {max_date} ({len(all_dates)} unique dates)")

    # Auto-compute max_days: count available telemetry date directories
    # between the first and last query date (lexicographic order).
    # This avoids loading dates before/after the query range and keeps
    # the run time manageable for systems like Telecom (15+ dates, 2 needed).
    min_date_str = min_date.strftime("%Y_%m_%d")
    max_date_str = max_date.strftime("%Y_%m_%d")
    telemetry_dir = os.path.join(data_dir, "telemetry")
    if os.path.exists(telemetry_dir):
        available_dates = sorted([d for d in os.listdir(telemetry_dir)
                                  if d.startswith("20")])
        num_available = len(available_dates)
        needed_range = [d for d in available_dates
                        if min_date_str <= d <= max_date_str]
        auto_max_days = len(needed_range) + 2  # +2 for days at edges
        auto_max_days = min(auto_max_days, max_days, num_available)
        print(f"  Available telemetry dates: {num_available}; "
              f"needed range includes ~{len(needed_range)} directories "
              f"(loading {auto_max_days})")
    else:
        auto_max_days = max_days

    # Create adapter covering needed dates
    adapter = _make_adapter(cfg["adapter_cls"], auto_max_days)
    all_entities = adapter.discover_entities(data_dir)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities
    entity_ids = [e.entity_id for e in cont_entities]
    N = len(entity_ids)
    print(f"  Entities: {N}")

    # Load all events (covers all dates up to max_days)
    t0 = time.time()
    all_events = adapter.extract_events(data_dir, cont_entities)
    event_load_time = time.time() - t0
    print(f"  Loaded {len(all_events)} events in {event_load_time:.1f}s")

    if not all_events:
        print("  No events loaded!")
        return {"n": 0}

    # Build GLOBAL tensor from all loaded events
    t0 = time.time()
    global_tensor, global_obs_mask, global_ts = build_telemetry_tensor(
        events=all_events,
        entity_ids=entity_ids,
        kpi_to_ob=KPI_TO_OB,
        max_ob_features=8,
        resample_interval_sec=resample_sec,
        window_start_ts=None,
        window_end_ts=None,
    )
    tensor_build_time = time.time() - t0
    T_global = global_tensor.shape[0]
    if T_global == 0:
        print("  Empty global tensor!")
        return {"n": 0}

    t_min_dt = datetime.fromtimestamp(global_ts[0])
    t_max_dt = datetime.fromtimestamp(global_ts[-1])
    print(f"  Global tensor: {global_tensor.shape} [{t_min_dt} — {t_max_dt}]")
    print(f"  Tensor build: {tensor_build_time:.1f}s")

    # Entity types
    type_str_to_idx = {"container": 0, "database": 1, "middleware": 2, "host": 0}
    type_strs = [assign_entity_type(eid) for eid in entity_ids] if not all_zero_type \
                else ["container"] * N
    type_indices = np.array(
        [type_str_to_idx.get(t, 0) if not all_zero_type else 0 for t in type_strs],
        dtype=np.int32,
    )

    # Build global obs mask for model input
    global_obs_mask_2d = (global_obs_mask.sum(axis=0) > 0).astype(np.float32)
    global_obs_mask_model = np.pad(global_obs_mask_2d, ((0, 0), (0, MAX_D - 8)),
                                   mode='constant')

    # Normalize global tensor with OB stats
    global_tensor_norm = np.nan_to_num(
        (global_tensor - ob_mean) / (ob_std + 1e-6), nan=0.0)
    global_tensor_pad = np.pad(global_tensor_norm,
                               ((0, 0), (0, 0), (0, MAX_D - 8)), mode='constant')

    # Load labels (evaluation only)
    labels = extract_labels_from_record(data_dir, entity_ids, adapter)

    # Validate coverage
    coverage = validate_dataset_coverage(labels, global_ts, entity_ids, sys_name)
    print(f"  Coverage: entity={coverage['entity_match']}/{coverage['total_labels']} "
          f"time={coverage['time_in_range']}/{coverage['total_labels']}")

    violations = validate_labels_in_telemetry_range(labels, global_ts, sys_name)
    if violations:
        for v in violations[:5]:
            print(f"  WARNING: {v}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more")

    # Process each query as independent Episode
    query_results = []
    total_time = 0.0
    processed = 0
    skipped = 0
    skip_no_data = 0
    skip_no_label = 0

    for qi, query in enumerate(queries):
        # Determine time range for this episode
        data_start_ts = (query.window_start - timedelta(minutes=burn_in_min)).timestamp()
        data_end_ts = query.window_end.timestamp()

        # Find global tensor indices within [data_start_ts, data_end_ts]
        idx_slice = np.where((global_ts >= data_start_ts) & (global_ts <= data_end_ts))[0]
        if len(idx_slice) < WS + 2:
            skip_no_data += 1
            continue

        # Slice
        s_start = idx_slice[0]
        s_end = idx_slice[-1] + 1
        tensor_sub = global_tensor_pad[s_start:s_end]
        ts_sub = global_ts[s_start:s_end]
        T_sub = tensor_sub.shape[0]

        # Build time-varying obs_mask slice
        obs_mask_sub_3d = global_obs_mask[s_start:s_end]
        obs_mask_sub_2d = (obs_mask_sub_3d.sum(axis=0) > 0).astype(np.float32)
        obs_mask_sub_2d = np.pad(obs_mask_sub_2d, ((0, 0), (0, MAX_D - 8)), mode='constant')

        # Inference
        t_inf = time.time()
        residuals, h_states = compute_residuals_and_latents(
            model, state.params, tensor_sub, type_indices,
            obs_mask_sub_2d, use_posterior=use_posterior, ws=WS)
        inference_time = time.time() - t_inf
        total_time += inference_time

        if residuals.shape[0] == 0:
            skip_no_data += 1
            continue

        # Burn-in mask: everything before query window start inside this slice
        qw_start_in_slice = query.window_start.timestamp()
        burn_in_mask = ts_sub < qw_start_in_slice
        qw_mask = (ts_sub >= query.window_start.timestamp()) & \
                   (ts_sub <= query.window_end.timestamp())

        # Compute joint scores
        joint = compute_joint_scores(
            residuals=residuals,
            latents=h_states,
            model_onset_scores=None,
            method=scoring_method,
            burn_in_mask=burn_in_mask,
            lambda_residual=1.0,
            lambda_shift=0.5,
            lambda_early=0.3,
            lambda_onset=0.0,
            temporal_smooth_window=3,
        )
        joint.query_window_mask = qw_mask

        # Find matching GT label
        gt_label = _find_matching_label(query, labels, ts_sub, entity_ids)
        if gt_label is None:
            skip_no_label += 1
            continue

        # Evaluate
        tolerance_steps = max(1, (query.gt_tolerance_min * 60) // resample_sec)
        metric = evaluate_joint(
            joint=joint,
            gt_component_idx=gt_label["component_idx"],
            gt_onset_ts=gt_label["timestamp"],
            timestamps=ts_sub,
            onset_tolerance_steps=tolerance_steps,
        )
        query_results.append(metric)
        processed += 1

        if (qi + 1) % 30 == 0:
            print(f"    {qi + 1}/{len(queries)} queries... "
                  f"(avg inf: {total_time/max(1,processed):.2f}s)")

    print(f"  Evaluated: {processed} | skipped: {skipped} "
          f"(no_data: {skip_no_data}, no_label: {skip_no_label})")
    if processed > 0:
        print(f"  Avg inference time: {total_time/processed:.2f}s per query")

    return aggregate_metrics(query_results)


def extract_labels_from_record(data_dir, entity_ids, adapter):
    """Extract labels from record.csv — EVALUATION ONLY (never exposed to model)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_posterior', action='store_true',
                        help='Use posterior mode (default: prior-only)')
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market/cloudbed-1', 'Market/cloudbed-2', 'Telecom'])
    parser.add_argument('--method', type=str, default='calibrated',
                        choices=['residual_only', 'calibrated', 'legacy_per_entity_max',
                                 'onset_head'],
                        help='Scoring method (default: calibrated)')
    parser.add_argument('--all_zero_type', action='store_true',
                        help='Use all-zero type indices (ablation)')
    parser.add_argument('--burn_in_min', type=int, default=60,
                        help='Burn-in period before query window (minutes)')
    parser.add_argument('--resample_sec', type=int, default=60,
                        help='Time grid resampling interval (seconds)')
    parser.add_argument('--max_days', type=int, default=30,
                        help='Maximum number of telemetry days to load')
    args = parser.parse_args()

    use_posterior = args.ablation_posterior
    mode_str = "POSTERIOR (ablation)" if use_posterior else "PRIOR-ONLY"
    type_str = "all-zero" if args.all_zero_type else "heuristic"
    resample_sec = args.resample_sec

    if args.method == "onset_head" and not use_posterior:
        print("NOTE: onset_head requires posterior mode; falling back to calibrated.")
        args.method = "calibrated"

    print("=" * 60)
    print(f"Phase A-Hardening P0.5: Query Episode Protocol ({mode_str})")
    print(f"Scoring: {args.method} | Type: {type_str} | "
          f"Resample: {resample_sec}s | Burn-in: {args.burn_in_min}min")
    print("=" * 60)

    # Load model
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
            resample_sec=resample_sec,
            max_days=args.max_days,
        )
        elapsed = time.time() - t0
        all_results[sys_name] = agg
        print(f"  Total time: {elapsed:.1f}s")

    # Report
    print(f"\n{'='*80}")
    print(f"RESULTS ({mode_str}, {args.method}, {type_str} type)")
    print(f"{'='*80}")
    header = (f"{'System':<18s} {'Comp T1':>8s} {'Comp T3':>8s} "
              f"{'MRR':>8s} {'AvgR':>6s} {'TimeHit':>8s} "
              f"{'TimeMAE':>8s} {'JointHit':>8s} {'N':>5s}")
    print(header)
    print("-" * 80)
    for sys_name in args.systems:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        if r.get("n", 0) == 0:
            print(f"{sys_name:<18s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>6s} "
                  f"{'--':>8s} {'--':>8s} {'--':>8s} {'0':>5s}")
            continue
        print(f"{sys_name:<18s} "
              f"{r['component_top1']:>7.1%} {r['component_top3']:>7.1%} "
              f"{r['mrr']:>8.3f} {r['avg_rank']:>6.2f} "
              f"{r['time_hit_rate']:>7.1%} {r['time_mae']:>8.1f} "
              f"{r['joint_hit_rate']:>7.1%} {r['n']:>5d}")
    print("-" * 80)

    print(f"\nDone. Mode: {mode_str} | Method: {args.method} | Type: {type_str}")


if __name__ == "__main__":
    main()
