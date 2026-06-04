"""Phase A-Hardening: Strict zero-shot evaluation on Bank / Market / Telecom.

Rules:
  1. NO GT component read during candidate generation.
  2. NO GT onset time read during time localization.
  3. prior-only as primary; posterior as --ablation flag.
  4. OpenRCA data NEVER used for training.

Reports the main result table:
  System | Comp Top-1 | Top-3 | MRR | Time MAE | Time Hit | Joint Hit
"""
import os, sys, time, argparse
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phaseA", "best")
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"
MAX_D = 16
WS = 23  # RSSM window size


# ═══════════════════════════════════════════════════════════════════════
# Data: parse one OpenRCA system into OB 8-feature space
# ═══════════════════════════════════════════════════════════════════════

# Type assignments per entity (based on entity name heuristics)
def assign_entity_type(entity_id: str) -> str:
    """Heuristic entity type assignment."""
    eid = entity_id.lower()
    if any(k in eid for k in ['mysql', 'redis', 'db', 'postgres', 'mongo']):
        return 'database'
    if any(k in eid for k in ['tomcat', 'apache', 'nginx', 'java', 'jvm']):
        return 'middleware'
    if any(k in eid for k in ['docker', 'container', 'pod']):
        return 'container'
    if any(k in eid for k in ['host', 'node', 'server']):
        return 'host'
    return 'container'  # default for Bank/Market

# KPI → OB feature mapping (same as original clean eval)
KPI_TO_OB = {
    "cpu": 0, "jvm_cpu": 0,
    "mem": 1, "mem_usage": 1, "jvm_mem": 1,
    "net_rx": 2,
    "net_tx": 3,
    "disk_io": 4, "mysql_io": 4,
    "threads": 5, "sessions": 5,
    "fgc": 6,
}


def parse_system(system_name: str, data_dir: str, max_days: int = 1):
    """Parse an OpenRCA system into OB-mapped tensor + labels.

    Returns:
        tensor:   [T, N, 8]    mapped to OB feature space
        entities: list of entity dicts {id, type, type_idx}
        labels:   list of {component, component_idx, timestamp, reason}
        timestamps: [T]         original timestamps
    """
    from foundation.adapters import OpenRCABankAdapter, OpenRCATelecomAdapter

    if system_name == "Telecom":
        adapter = OpenRCATelecomAdapter(max_days=max_days,
                                        max_container_timestamps=5000)
    else:
        adapter = OpenRCABankAdapter(max_days=max_days,
                                     max_container_events=100000,
                                     max_container_rows=1000000)

    all_entities = adapter.discover_entities(data_dir)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities

    events = adapter.extract_events(data_dir, cont_entities)

    entity_ids = [e.entity_id for e in cont_entities]
    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    N = len(entity_ids)

    # Assign types — map to 3-type vocabulary (0=container, 1=database, 2=middleware)
    type_strs = [assign_entity_type(eid) for eid in entity_ids]
    type_str_to_idx = {"container": 0, "database": 1, "middleware": 2, "host": 0}
    type_indices = np.array([type_str_to_idx.get(t, 0) for t in type_strs], dtype=np.int32)

    # Build OB-mapped tensor
    all_ts = sorted(set(ev.timestamp for ev in events))
    ts_to_idx = {t: i for i, t in enumerate(all_ts)}
    T = len(all_ts)
    tensor = np.zeros((T, N, 8), dtype=np.float32)
    counts = np.zeros((T, N, 8), dtype=np.float32)

    for ev in events:
        if ev.entity_id not in eid_to_idx:
            continue
        fn = ev.feature_name
        if fn not in KPI_TO_OB:
            continue
        ob_idx = KPI_TO_OB[fn]
        t = ts_to_idx.get(ev.timestamp)
        n = eid_to_idx[ev.entity_id]
        if t is None:
            continue
        tensor[t, n, ob_idx] += ev.value
        counts[t, n, ob_idx] += 1

    # Average + forward fill
    for n_idx in range(N):
        for d in range(8):
            cnt = counts[:, n_idx, d]
            mask = cnt > 0
            if mask.any():
                tensor[mask, n_idx, d] /= cnt[mask]
            last = 0.0
            for tt in range(T):
                if counts[tt, n_idx, d] > 0:
                    last = tensor[tt, n_idx, d]
                else:
                    tensor[tt, n_idx, d] = last

    # Labels
    labels_raw = adapter.extract_labels(data_dir, cont_entities)
    labels = []
    for lb in labels_raw:
        matched = -1
        if lb.component in eid_to_idx:
            matched = eid_to_idx[lb.component]
        else:
            cl = lb.component.lower().replace("_", "").replace("-", "").replace(" ", "")
            for eid in entity_ids:
                el = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
                if cl in el or el in cl:
                    matched = eid_to_idx[eid]
                    break
        if matched >= 0:
            labels.append({
                "component": lb.component,
                "component_idx": matched,
                "timestamp": float(lb.occurrence_datetime) if lb.occurrence_datetime else 0.0,
                "reason": lb.reason,
            })

    entities_info = [{"id": eid, "type": t, "type_idx": ti}
                     for eid, t, ti in zip(entity_ids, type_strs, type_indices)]

    return tensor, entities_info, labels, np.array(all_ts, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Model inference
# ═══════════════════════════════════════════════════════════════════════

def compute_residuals_and_latents(model, params, tensor_norm, type_idx,
                                  obs_mask, use_posterior=False, ws=WS):
    """Sliding-window inference: returns residuals [T,N] and h_states [T,N,D]."""
    T, N, D = tensor_norm.shape
    residuals = np.full((T, N), np.nan, dtype=np.float32)
    # Collect h_seq from each window (use midpoint assignment)
    h_accum = np.zeros((T, N, 256), dtype=np.float32)  # det_dim=256
    h_counts = np.zeros((T, N), dtype=np.float32)

    windows, pos = [], []
    for start in range(0, T - ws, max(1, ws // 2)):
        end = start + ws + 1
        if end > T:
            break
        windows.append(tensor_norm[start:end])
        pos.append((start, end))

    if not windows:
        return residuals, h_accum

    windows = np.stack(windows, axis=0)
    t_idx_j = jnp.array(type_idx)
    o_mask_j = jnp.array(obs_mask)
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), 8):
        bx = jnp.array(windows[i:i + 8])
        out = model.apply(
            {"params": params}, bx, t_idx_j, rng,
            obs_mask=o_mask_j, use_posterior=use_posterior)
        res_b = np.array(out["residual"])       # [B, T_w, N]
        h_b = np.array(out["h_seq"])             # [B, T_w, N, 256]

        for j, (s, e) in enumerate(pos[i:i + 8]):
            ns = min(res_b.shape[1], e - s)
            residuals[s + 1:e] = res_b[j, :ns]  # shift to align

            nh = min(h_b.shape[1], e - s)
            # Assign h_seq[t] to timestep s+t+1 (align with residual)
            for tt in range(nh):
                t_global = s + tt + 1
                if t_global < T:
                    h_accum[t_global] += h_b[j, tt]
                    h_counts[t_global] += 1

    # Average overlapping h assignments
    for tt in range(T):
        for n in range(N):
            if h_counts[tt, n] > 0:
                h_accum[tt, n] /= h_counts[tt, n]

    return np.nan_to_num(residuals, nan=0.0), h_accum


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_posterior', action='store_true',
                        help='Use posterior mode (default: prior-only)')
    parser.add_argument('--systems', nargs='+',
                        default=['Bank', 'Market', 'Telecom'])
    args = parser.parse_args()

    use_posterior = args.ablation_posterior
    mode_str = "POSTERIOR (ablation)" if use_posterior else "PRIOR-ONLY"

    # ── Load model ──
    print("=" * 60)
    print(f"Phase A-Hardening: Strict Eval ({mode_str})")
    print("=" * 60)

    model = RCAWorldFoundation(
        common_dim=128, max_obs_dim=MAX_D, num_entity_types=3,
        det_dim=256, stoch_dim=32, stoch_classes=32,
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

    # Load OB normalization stats
    with h5py.File(DATA_PATH, 'r') as f:
        ob_mean = f['stats/node_mean'][:].mean(axis=1).reshape(1, 1, 8)
        ob_std = f['stats/node_std'][:].mean(axis=1).reshape(1, 1, 8)

    # ── Evaluate each system ──
    all_results = {}

    for sys_name in args.systems:
        print(f"\n{'='*60}")
        print(f"  {sys_name}")
        print(f"{'='*60}")

        if sys_name == "Bank":
            data_dir = os.path.join(OPENRCA, "Bank", "Bank")
        elif sys_name == "Market":
            data_dir = os.path.join(OPENRCA, "Market", "Market", "cloudbed-1")
        elif sys_name == "Telecom":
            data_dir = os.path.join(OPENRCA, "Telecom", "Telecom")
        else:
            print(f"  Unknown system: {sys_name}")
            continue

        # Parse
        t0 = time.time()
        tensor, entities_info, labels, timestamps = parse_system(sys_name, data_dir)
        N = len(entities_info)
        T = tensor.shape[0]
        type_indices = np.array([e["type_idx"] for e in entities_info], dtype=np.int32)
        print(f"  Entities: {N}, Timesteps: {T}, Labels: {len(labels)}")
        type_counts = {}
        for e in entities_info:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        print(f"  Types: {type_counts}")
        print(f"  Parse time: {time.time()-t0:.1f}s")

        # Normalize with OB stats
        tensor_norm = np.nan_to_num(
            (tensor - ob_mean) / (ob_std + 1e-6), nan=0.0)
        tensor_norm = np.pad(
            tensor_norm, ((0, 0), (0, 0), (0, MAX_D - 8)), mode='constant')
        obs_mask = np.zeros((N, MAX_D), dtype=np.float32)
        obs_mask[:, :8] = 1.0

        # Inference
        print("  Computing residuals + latents (prior-only)...")
        t0 = time.time()
        residuals, h_states = compute_residuals_and_latents(
            model, state.params, tensor_norm, type_indices, obs_mask,
            use_posterior=use_posterior, ws=WS)
        print(f"    {time.time()-t0:.1f}s")

        # ── Per-query evaluation (NO GT leak) ──
        query_results = []
        for li, label in enumerate(labels):
            rc = label["component_idx"]
            if rc < 0 or rc >= N:
                continue

            # Build joint scores WITHOUT reading GT
            joint = compute_joint_scores(
                residuals=residuals,
                latents=h_states,
                model_onset_scores=None,
                lambda_residual=1.0,
                lambda_shift=0.5,
                lambda_early=0.3,
                lambda_onset=0.0,
                temporal_smooth_window=3,
            )

            # Evaluate against GT (GT used ONLY in metric, not candidate gen)
            metric = evaluate_joint(
                joint=joint,
                gt_component_idx=rc,
                gt_onset_ts=label["timestamp"],
                timestamps=timestamps,
                onset_tolerance_steps=5,
            )
            query_results.append(metric)

        agg = aggregate_metrics(query_results)
        all_results[sys_name] = agg

    # ── Report ──
    print(f"\n{'='*80}")
    print(f"RESULTS ({mode_str})")
    print(f"{'='*80}")
    header = (f"{'System':<10s} {'Comp T1':>8s} {'Comp T3':>8s} "
              f"{'MRR':>8s} {'AvgR':>6s} {'TimeHit':>8s} "
              f"{'TimeMAE':>8s} {'JointHit':>8s} {'N':>5s}")
    print(header)
    print("-" * 80)
    for sys_name in args.systems:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        print(f"{sys_name:<10s} "
              f"{r['component_top1']:>7.1%} {r['component_top3']:>7.1%} "
              f"{r['mrr']:>8.3f} {r['avg_rank']:>6.2f} "
              f"{r['time_hit_rate']:>7.1%} {r['time_mae']:>8.1f} "
              f"{r['joint_hit_rate']:>7.1%} {r['n']:>5d}")
    print("-" * 80)

    # Comparison baseline
    print(f"\nReference (original clean eval, 16 faults):")
    print(f"  {'Bank':<10s} {'38%':>8s} {'50%':>8s} {'0.50':>8s}")

    print(f"\nDone. Mode: {mode_str}")


if __name__ == "__main__":
    main()
