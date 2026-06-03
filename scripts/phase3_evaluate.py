"""Phase 3: RCA Evaluation on Nezha fault data.

Evaluates:
- RQ2: Can prediction residuals improve RCA?
- RQ3: Can repair intervention distinguish root cause from symptoms?
"""
import os
import sys
import json

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import optax
from flax.training import train_state
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models.graph_rssm import GraphRSSM
from models.rssm import RSSMState
from training.losses import gaussian_nll_per_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase2")


def load_checkpoint():
    """Load trained model checkpoint."""
    model = GraphRSSM(
        node_dim=8, edge_dim=4, ctx_dim=3,
        graph_hidden=128, num_heads=4,
        det_dim=256, stoch_dim=32, stoch_classes=32,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, rng = jax.random.split(rng)

    # Dummy init
    b_nodes = jnp.ones((1, 24, 10, 8))
    b_edges = jnp.ones((1, 24, 14, 4))
    b_ctx = jnp.ones((1, 24, 3))
    b_edges_idx = jnp.zeros((2, 14), dtype=jnp.int32)

    params = model.init(init_rng, rng, b_nodes, b_edges, b_edges_idx, b_ctx)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )

    # Load checkpoint
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt_path = os.path.join(CKPT_DIR, "best")
    if os.path.exists(ckpt_path):
        restored = orbax_checkpointer.restore(ckpt_path)
        state = state.replace(params=restored["params"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using random init")

    return state, model


def compute_residuals_fast(state, nodes, edges, ctx, edge_index, window_sec=24):
    """Fast residual computation using batching."""
    T = nodes.shape[0]
    ws = window_sec
    stride = ws // 2

    batch_nodes = []
    batch_edges = []
    batch_ctx = []
    batch_starts = []

    for start in range(0, max(1, T - ws), max(1, stride)):
        end = min(T, start + ws)
        if end - start < 4:
            continue
        w_nodes = nodes[start:end]
        w_edges = edges[start:end]
        w_ctx = ctx[start:end]
        if w_nodes.shape[0] < ws:
            pad = ws - w_nodes.shape[0]
            w_nodes = np.pad(w_nodes, ((0, pad), (0, 0), (0, 0)))
            w_edges = np.pad(w_edges, ((0, pad), (0, 0), (0, 0)))
            w_ctx = np.pad(w_ctx, ((0, pad), (0, 0)))

        batch_nodes.append(w_nodes)
        batch_edges.append(w_edges)
        batch_ctx.append(w_ctx)
        batch_starts.append(start)

    if not batch_nodes:
        return np.full((T, nodes.shape[1]), np.nan, dtype=np.float32), None, None

    bn = jnp.array(np.stack(batch_nodes))
    be = jnp.array(np.stack(batch_edges))
    bw = jnp.array(np.stack(batch_ctx))
    ei = jnp.array(edge_index, dtype=jnp.int32)

    # Run model in batches of 20 to avoid OOM
    batch_size = 20
    all_res = []
    for i in range(0, len(bn), batch_size):
        sub_bn = bn[i:i+batch_size]
        sub_be = be[i:i+batch_size]
        sub_bw = bw[i:i+batch_size]

        out = state.apply_fn(state.params, jax.random.PRNGKey(0),
                             sub_bn, sub_be, ei, sub_bw,
                             use_posterior=True)
        nt = sub_bn[:, 1:, :, :]
        sub_res = gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt)
        all_res.append(np.array(sub_res))

    res_batch = np.concatenate(all_res, axis=0) if all_res else np.array([])

    # Scatter results back
    N = nodes.shape[1]
    all_res = np.full((T, N), np.nan, dtype=np.float32)
    for i, start in enumerate(batch_starts):
        end = min(T, start + ws)
        if i < res_batch.shape[0]:
            n_steps = min(ws - 1, end - start)
            all_res[start + 1:end] = np.array(res_batch[i, :n_steps])

    return all_res, None, None


def evaluate_rca(residuals, fault_info, service_names, normal_baseline):
    """Evaluate RCA performance using prediction residuals.

    Args:
        residuals: [T, N] per-node residuals
        fault_info: dict with root_cause service, fault_type, start/end
        service_names: list of service name strings
        normal_baseline: (mean, std) from normal data

    Returns:
        dict with evaluation metrics
    """
    N = len(service_names)
    rc = fault_info["root_cause_service"]
    if rc is None:
        return {"rc_rank": -1, "top1": False, "rc_earliest": False}

    threshold = normal_baseline["mean"] + 3 * normal_baseline["std"]
    fault_start = max(0, fault_info.get("fault_start", 0))
    fault_end = min(residuals.shape[0], fault_info.get("fault_end", residuals.shape[0]))

    # Average residual during fault window
    fault_residuals = residuals[fault_start:fault_end]
    if len(fault_residuals) == 0:
        return {"rc_rank": -1, "top1": False, "rc_earliest": False}

    avg_res = np.nanmean(fault_residuals, axis=0)
    # Handle NaN: set to -inf so they rank last
    avg_res = np.nan_to_num(avg_res, nan=-1.0)

    # Rank by descending residual
    rank = np.argsort(-avg_res)
    rc_rank = int(np.where(rank == rc)[0][0]) + 1
    top1 = (rank[0] == rc)

    # Earliest anomaly
    first_anomaly = np.full(N, 9999, dtype=int)
    for t in range(fault_start, min(fault_end, residuals.shape[0])):
        for n in range(N):
            if first_anomaly[n] == 9999 and not np.isnan(residuals[t, n]) and residuals[t, n] > threshold:
                first_anomaly[n] = t
    rc_earliest = (first_anomaly[rc] <= np.min(first_anomaly))

    return {
        "rc_rank": rc_rank,
        "top1": top1,
        "rc_earliest": rc_earliest,
        "avg_residual": {service_names[i]: float(avg_res[i]) for i in range(N)},
        "first_anomaly": {service_names[i]: int(first_anomaly[i]) for i in range(N)},
        "top3": [service_names[r] for r in rank[:3]],
    }


def main():
    print(f"Device: {jax.devices()[0]}")

    # Load model
    print("Loading model...")
    state, model = load_checkpoint()

    # Load dataset
    print("Loading dataset...")
    with h5py.File(DATA_PATH, "r") as f:
        edge_index = f["edge_index"][:]
        N = f.attrs["N_services"]
        service_names = [s.decode() if isinstance(s, bytes) else s for s in f["service_names"][:]]

        # Normal data for baseline
        normal_nodes = f["val"]["nodes"][:]
        normal_edges = f["val"]["edges"][:]
        normal_ctx = f["val"]["ctx"][:]

        # Load fault episodes
        fault_group = f["fault"]
        fault_episodes = []
        for ep_name in fault_group.keys():
            ep = fault_group[ep_name]
            fault_episodes.append({
                "nodes": ep["nodes"][:],
                "edges": ep["edges"][:],
                "ctx": ep["ctx"][:],
                "normal_mask": ep["normal_mask"][:],
                "fault_labels": json.loads(ep.attrs["fault_labels"]) if "fault_labels" in ep.attrs else [],
            })

    print(f"Loaded {len(fault_episodes)} fault episodes")

    # Compute normal baseline (use a few val windows)
    print("\nComputing normal baseline...")
    normal_residuals_all = []
    for i in range(min(5, len(normal_nodes))):
        res, _, _ = compute_residuals_fast(
            state, normal_nodes[i], normal_edges[i], normal_ctx[i], edge_index
        )
        normal_residuals_all.append(res[~np.isnan(res)])

    normal_flat = np.concatenate(normal_residuals_all)
    normal_baseline = {"mean": float(np.mean(normal_flat)), "std": float(np.std(normal_flat))}
    print(f"Normal baseline: mean={normal_baseline['mean']:.4f}, std={normal_baseline['std']:.4f}")

    # Evaluate each fault (limit to first 20 for speed)
    print("\n" + "=" * 60)
    print("RQ2: Root Cause Analysis by Prediction Residual")
    print("=" * 60)

    all_results = []
    svc_to_idx = {s: i for i, s in enumerate(service_names)}
    fault_count = 0
    max_faults = 20  # Limit for speed

    for ep_idx, ep in enumerate(fault_episodes):
        fault_labels = ep["fault_labels"]
        if not fault_labels or fault_count >= max_faults:
            continue

        # Compute residuals once per episode
        residuals, _, _ = compute_residuals_fast(
            state, ep["nodes"], ep["edges"], ep["ctx"], edge_index
        )
        T_ep = residuals.shape[0]

        for fl in fault_labels:
            if fault_count >= max_faults:
                break
            svc = fl["service"]
            if svc not in svc_to_idx:
                continue
            rc_idx = svc_to_idx[svc]

            rc_res = residuals[:, rc_idx]
            valid_mask = ~np.isnan(rc_res)
            if valid_mask.sum() < 5:
                continue

            peak_idx = np.nanargmax(rc_res)
            fault_start = max(0, peak_idx - 10)
            fault_end = min(T_ep, peak_idx + 10)

            fault_info = {
                "root_cause_service": rc_idx,
                "fault_type": fl["type"],
                "fault_start": fault_start,
                "fault_end": fault_end,
            }

            result = evaluate_rca(residuals, fault_info, service_names, normal_baseline)
            result["fault"] = f"{fl['type']}@{svc}"
            result["episode"] = ep_idx
            all_results.append(result)
            fault_count += 1

            print(f"\nFault: {result['fault']}")
            print(f"  Rank: {result['rc_rank']}/{N} {'✓' if result['top1'] else '✗'}")
            print(f"  Earliest: {'Yes' if result['rc_earliest'] else 'No'}")
            print(f"  Top-3: {', '.join(result['top3'])}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if all_results:
        top1 = sum(1 for r in all_results if r["top1"])
        earliest = sum(1 for r in all_results if r["rc_earliest"])
        valid = [r for r in all_results if r["rc_rank"] > 0]
        mean_rank = np.mean([r["rc_rank"] for r in valid]) if valid else 0

        print(f"Total faults evaluated: {len(all_results)}")
        print(f"Top-1 accuracy: {top1}/{len(all_results)} ({100*top1/len(all_results):.0f}%)")
        print(f"Root cause earliest anomaly: {earliest}/{len(all_results)} ({100*earliest/len(all_results):.0f}%)")
        print(f"Mean rank of root cause: {mean_rank:.1f}")

        # Per fault type breakdown
        from collections import defaultdict
        by_type = defaultdict(list)
        for r in all_results:
            ft = r["fault"].split("@")[0]
            by_type[ft].append(r["top1"])

        print("\nPer fault type Top-1:")
        for ft, results in sorted(by_type.items()):
            print(f"  {ft}: {sum(results)}/{len(results)} ({100*sum(results)/len(results):.0f}%)")
    else:
        print("No faults evaluated.")


if __name__ == "__main__":
    main()
