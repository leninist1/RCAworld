"""Phase 0.8 Round 1: Inference-only script — ZERO GT access at file level.

This script reads ONLY:
  - query.csv (task_index + instruction columns only, never scoring_points)
  - telemetry/
  - checkpoint
  - configuration parameters

It NEVER reads:
  - scoring_points column
  - record.csv
  - EvalTarget
  - gt_component, gt_time, gt_reason, gt_fault_count

Output: predictions.json with query_id, predictions list (datetime, component, score).
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax.training import train_state as flax_train_state
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

from foundation.models import RCAWorldFoundation
from foundation.evaluation.strict_eval import (
    compute_joint_scores,
    JointScores,
)
from foundation.evaluation.query_parser import (
    parse_inference_queries,
    InferenceQuery,
    format_episode_window,
    OPENRCA_TZ,
)
from foundation.evaluation.episode_builder import (
    build_telemetry_tensor,
    KPI_TO_OB,
    assign_entity_type,
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


def run_inference(sys_name, model, state, ob_mean, ob_std, use_posterior,
                  scoring_method, all_zero_type, burn_in_min, resample_sec):
    """Run inference for one system — ZERO GT access."""
    cfg = SYSTEM_CONFIGS[sys_name]
    data_dir = cfg["data_dir"]
    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return []

    query_path = os.path.join(data_dir, "query.csv")
    if not os.path.exists(query_path):
        print(f"  query.csv not found: {query_path}")
        return []

    inference_queries = parse_inference_queries(query_path)
    print(f"  Parsed {len(inference_queries)} queries from query.csv")

    burn_in_td = timedelta(minutes=burn_in_min)
    min_data_ts = min(q.window_start.timestamp() for q in inference_queries) - burn_in_td.total_seconds()
    max_data_ts = max(q.window_end.timestamp() for q in inference_queries)
    min_data_date = datetime.fromtimestamp(min_data_ts, tz=OPENRCA_TZ).date()
    max_data_date = datetime.fromtimestamp(max_data_ts, tz=OPENRCA_TZ).date()

    telemetry_dir = os.path.join(data_dir, "telemetry")
    available_dates = sorted([d for d in os.listdir(telemetry_dir)
                              if d.startswith("20")]) if os.path.exists(telemetry_dir) else []
    min_date_str = min_data_date.strftime("%Y_%m_%d")
    max_date_str = max_data_date.strftime("%Y_%m_%d")
    needed_dates_set = set(d for d in available_dates
                           if min_date_str <= d <= max_date_str)

    adapter = _make_adapter(cfg["adapter_cls"], needed_dates_set)
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
    print(f"  Loaded {len(all_events)} events in {time.time() - t0:.1f}s")

    if not all_events:
        return []

    t0 = time.time()
    global_tensor, global_obs_mask, global_ts = build_telemetry_tensor(
        events=all_events, entity_ids=entity_ids, kpi_to_ob=KPI_TO_OB,
        max_ob_features=8, resample_interval_sec=resample_sec,
    )
    T_global = global_tensor.shape[0]
    if T_global == 0:
        print("  Empty global tensor!")
        return []

    print(f"  Global tensor: {global_tensor.shape}")

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

    all_predictions = []
    processed = 0

    for qi, iq in enumerate(inference_queries):
        qid = iq.query_id

        data_start_ts = (iq.window_start - timedelta(minutes=burn_in_min)).timestamp()
        data_end_ts = iq.window_end.timestamp()

        idx_slice = np.where((global_ts >= data_start_ts) & (global_ts <= data_end_ts))[0]
        if len(idx_slice) < WS + 2:
            continue

        s_start = idx_slice[0]
        s_end = idx_slice[-1] + 1
        tensor_sub = global_tensor_pad[s_start:s_end]
        ts_sub = global_ts[s_start:s_end]

        obs_mask_sub_2d = (global_obs_mask[s_start:s_end].sum(axis=0) > 0).astype(np.float32)
        obs_mask_sub_2d = np.pad(obs_mask_sub_2d, ((0, 0), (0, MAX_D - 8)), mode='constant')

        residuals, h_states, coverage_mask = compute_residuals_and_latents(
            model, state.params, tensor_sub, type_indices,
            obs_mask_sub_2d, use_posterior=use_posterior, ws=WS)

        if residuals.shape[0] == 0:
            continue

        qw_start_ts = iq.window_start.timestamp()
        qw_end_ts = iq.window_end.timestamp()
        burn_in_mask = ts_sub < qw_start_ts
        qw_mask = (ts_sub >= qw_start_ts) & (ts_sub <= qw_end_ts)

        joint = compute_joint_scores(
            residuals=residuals, latents=h_states,
            model_onset_scores=None, method=scoring_method,
            burn_in_mask=burn_in_mask,
            lambda_residual=1.0, lambda_shift=0.5, lambda_early=0.3,
            temporal_smooth_window=3,
        )
        joint.query_window_mask = qw_mask

        requested_faults = iq.expected_fault_count
        min_time_dist = max(3, (5 * 60) // resample_sec)
        predictions = joint.decode_multiple_faults(
            max_faults=requested_faults,
            min_time_distance=min_time_dist,
        )

        pred_list = []
        for pt, pc, pscore in predictions:
            dt = datetime.fromtimestamp(ts_sub[pt], tz=OPENRCA_TZ).strftime("%Y-%m-%d %H:%M:%S")
            comp_name = entity_ids[pc] if pc < len(entity_ids) else f"entity_{pc}"
            pred_list.append({
                "datetime": dt,
                "component": comp_name,
                "score": round(float(pscore), 6),
            })

        comp_scores = joint.component_score  # [N]
        component_ranking = sorted(
            [(entity_ids[i], round(float(comp_scores[i]), 6)) for i in range(N)],
            key=lambda x: x[1], reverse=True,
        )

        all_predictions.append({
            "system": sys_name,
            "query_id": int(qid),
            "predictions": pred_list,
            "component_ranking": component_ranking,
        })
        processed += 1

        if (qi + 1) % 30 == 0:
            print(f"    {qi + 1}/{len(inference_queries)} queries...")

    print(f"  Inference complete: {processed}/{len(inference_queries)} queries predicted")
    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market/cloudbed-1', 'Market/cloudbed-2', 'Telecom'])
    parser.add_argument('--method', type=str, default='calibrated',
                        choices=['residual_only', 'calibrated', 'legacy_per_entity_max',
                                 'onset_head'])
    parser.add_argument('--all_zero_type', action='store_true')
    parser.add_argument('--burn_in_min', type=int, default=60)
    parser.add_argument('--resample_sec', type=int, default=120)
    parser.add_argument('--output', type=str, default='predictions.json')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 0.8 R1: Inference-only (ZERO GT access)")
    print(f"Scoring: {args.method} | Resample: {args.resample_sec}s | "
          f"Burn-in: {args.burn_in_min}min")
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

    all_predictions = []
    for sys_name in args.systems:
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")
        t0 = time.time()
        preds = run_inference(
            sys_name=sys_name, model=model, state=state,
            ob_mean=ob_mean, ob_std=ob_std,
            use_posterior=False, scoring_method=args.method,
            all_zero_type=args.all_zero_type,
            burn_in_min=args.burn_in_min,
            resample_sec=args.resample_sec,
        )
        elapsed = time.time() - t0
        all_predictions.extend(preds)
        print(f"  Time: {elapsed:.1f}s | Predictions: {len(preds)}")

    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)
    print(f"\nSaved {len(all_predictions)} query predictions to {output_path}")


if __name__ == "__main__":
    main()
