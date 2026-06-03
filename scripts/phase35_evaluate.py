"""Phase 3.5: RQ3 Repair Intervention + RQ4 Graph Ablation

RQ3: Does repair intervention distinguish root cause from symptoms?
RQ4: Does the dynamic graph really help over simpler architectures?
"""
import os, sys, json
import jax, jax.numpy as jnp
import numpy as np
import h5py
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models.graph_rssm import GraphRSSM
from models.rssm import RSSMState
from training.losses import gaussian_nll, gaussian_nll_per_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase2")


def load_model_and_state():
    """Load trained model."""
    model = GraphRSSM(
        node_dim=8, edge_dim=4, ctx_dim=3,
        graph_hidden=128, num_heads=4,
        det_dim=256, stoch_dim=32, stoch_classes=32,
    )
    rng = jax.random.PRNGKey(42)
    init_rng, _ = jax.random.split(rng)
    b = jnp.ones((1, 24, 10, 8))
    be = jnp.ones((1, 24, 14, 4))
    bw = jnp.ones((1, 24, 3))
    ei = jnp.zeros((2, 14), dtype=jnp.int32)
    params = model.init(init_rng, rng, b, be, ei, bw)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    ckpt_path = os.path.join(CKPT_DIR, "best")
    restored = orbax_checkpointer.restore(ckpt_path)
    state = state.replace(params=restored["params"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return state, model


def compute_residuals(state, nodes, edges, ctx, edge_index, window_sec=24, batch_max=20):
    """Compute per-node residuals over sliding windows."""
    T = nodes.shape[0]
    ws = window_sec
    stride = ws // 2
    N = nodes.shape[1]
    all_res = np.full((T, N), np.nan, dtype=np.float32)

    batch_nodes, batch_starts = [], []
    for start in range(0, max(1, T - ws), max(1, stride)):
        end = min(T, start + ws)
        if end - start < 4:
            continue
        w = nodes[start:end]
        we = edges[start:end]
        wc = ctx[start:end]
        if w.shape[0] < ws:
            pad = ws - w.shape[0]
            w = np.pad(w, ((0, pad), (0, 0), (0, 0)))
            we = np.pad(we, ((0, pad), (0, 0), (0, 0)))
            wc = np.pad(wc, ((0, pad), (0, 0)))
        batch_nodes.append((w, we, wc, start))

    ei = jnp.array(edge_index, dtype=jnp.int32)

    for i in range(0, len(batch_nodes), batch_max):
        chunk = batch_nodes[i:i+batch_max]
        bn = jnp.array(np.stack([x[0] for x in chunk]))
        be_b = jnp.array(np.stack([x[1] for x in chunk]))
        bw = jnp.array(np.stack([x[2] for x in chunk]))
        starts = [x[3] for x in chunk]

        out = state.apply_fn(state.params, jax.random.PRNGKey(0), bn, be_b, ei, bw, use_posterior=True)
        nt = bn[:, 1:, :, :]
        res = np.array(gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt))

        for j, start in enumerate(starts):
            end = min(T, start + ws)
            n_steps = min(ws - 1, end - start)
            all_res[start + 1:end] = res[j, :n_steps]

    return all_res


def compute_latent_deviation(state, nodes, edges, ctx, edge_index, N, window_sec=24):
    """Compute per-service latent deviation from prior (uniform)."""
    ws = min(window_sec, nodes.shape[0])
    if ws < 4:
        return np.ones(N)

    bn = jnp.array(nodes[-ws:][None])
    be = jnp.array(edges[-ws:][None])
    bw = jnp.array(ctx[-ws:][None])
    ei = jnp.array(edge_index, dtype=jnp.int32)

    out = state.apply_fn(state.params, jax.random.PRNGKey(0), bn, be, ei, bw, use_posterior=True)

    # z_seq: [1, T, N, D_stoch, C]
    z_seq = np.array(out["z_seq"])
    z_last = z_seq[:, -1, :, :, :]  # [1, N, D_stoch, C]
    C = z_last.shape[-1]

    # Convert one-hot to probabilities (use softmax as approximation)
    # z is one-hot with straight-through, so we examine the logits
    post_logits = np.array(out["post_logits"])[:, -1, :, :, :]  # [1, N, D_stoch, C]
    post_probs = np.exp(post_logits - post_logits.max(axis=-1, keepdims=True))
    post_probs = post_probs / (post_probs.sum(axis=-1, keepdims=True) + 1e-8)

    # Entropy: deviation from uniform (uniform entropy = log(C))
    uniform_entropy = np.log(C)
    entropy = -np.sum(post_probs * np.log(post_probs + 1e-8), axis=-1)  # [1, N, D_stoch]
    avg_entropy = entropy.mean(axis=-1)  # [1, N]
    deviation = np.abs(avg_entropy[0] - uniform_entropy)  # [N]
    return deviation


def compute_combined_rank(residuals, fault_start, fault_end, edge_index, N, deviation, svc_names):
    """Compute RCA ranking combining residual + temporal + topology + deviation.

    Evidence:
    - A(v): average residual during fault window
    - T(v): temporal precedence (earlier = higher)
    - P(v): topology consistency (has downstream services?)
    - D(v): latent deviation
    """
    fault_res = residuals[fault_start:fault_end]
    if len(fault_res) == 0:
        return np.arange(N)

    avg_res = np.nanmean(fault_res, axis=0)
    avg_res = np.nan_to_num(avg_res, nan=-1)

    # Temporal: first to cross threshold
    threshold = np.nanmean(residuals) + 2 * np.nanstd(residuals)
    first_dev = np.full(N, 9999, dtype=int)
    for t in range(fault_start, min(fault_start + 30, residuals.shape[0])):
        for n in range(N):
            if first_dev[n] == 9999 and not np.isnan(residuals[t, n]) and residuals[t, n] > threshold:
                first_dev[n] = t
    max_dev = first_dev.max()
    t_score = np.array([1.0 - d / (max_dev + 1) if d < 9999 else 0.0 for d in first_dev])

    # Topology: services with downstream edges get higher weight
    src, dst = edge_index[0], edge_index[1]
    n_downstream = np.array([len(np.unique(dst[src == n])) for n in range(N)])
    p_score = n_downstream / (n_downstream.max() + 1)  # [N]

    # Deviation score
    d_score = deviation / (deviation.max() + 1e-8)

    # Normalize residual
    res_range = avg_res.max() - avg_res.min()
    r_score = (avg_res - avg_res.min()) / (res_range + 1e-8) if res_range > 0 else np.ones(N) * 0.1

    # Combine: alpha * residual + beta * temporal + gamma * topology + delta * deviation
    combined = 0.40 * r_score + 0.15 * t_score + 0.15 * p_score + 0.30 * d_score

    rank = np.argsort(-combined)
    return rank, {
        "r_score": r_score, "t_score": t_score, "p_score": p_score, "d_score": d_score,
        "combined": combined, "first_dev": first_dev,
    }


def main():
    print(f"Device: {jax.devices()[0]}")
    state, model = load_model_and_state()

    with h5py.File(DATA_PATH, "r") as f:
        edge_index = f["edge_index"][:]
        N = f.attrs["N_services"]
        service_names = [s.decode() if isinstance(s, bytes) else s for s in f["service_names"][:]]
        svc_to_idx = {s: i for i, s in enumerate(service_names)}

        # Normal baseline
        val_nodes = f["val"]["nodes"][:]
        val_edges = f["val"]["edges"][:]
        val_ctx = f["val"]["ctx"][:]

        # Fault episodes
        fault_episodes = []
        for ep_name in f["fault"].keys():
            ep = f["fault"][ep_name]
            fault_episodes.append({
                "nodes": ep["nodes"][:], "edges": ep["edges"][:], "ctx": ep["ctx"][:],
                "normal_mask": ep["normal_mask"][:],
                "fault_labels": json.loads(ep.attrs.get("fault_labels", "[]")),
            })

    print(f"Services: {service_names}")
    print(f"Fault episodes: {len(fault_episodes)}")

    # Normal baseline
    normal_res_all = []
    for i in range(min(3, len(val_nodes))):
        res = compute_residuals(state, val_nodes[i], val_edges[i], val_ctx[i], edge_index)
        normal_res_all.append(res[~np.isnan(res)])
    normal_flat = np.concatenate(normal_res_all)
    normal_baseline = {"mean": float(np.mean(normal_flat)), "std": float(np.std(normal_flat))}
    threshold = normal_baseline["mean"] + 3 * normal_baseline["std"]
    print(f"Normal baseline: μ={normal_baseline['mean']:.4f}, σ={normal_baseline['std']:.4f}")
    print(f"Anomaly threshold: {threshold:.4f}")

    # ================================================================
    # RQ3: Evidence Fusion
    # ================================================================
    print("\n" + "=" * 60)
    print("RQ3: Evidence Fusion (Residual + Temporal + Topology + Latent Deviation)")
    print("=" * 60)

    results_rq3 = []
    for ep_idx, ep in enumerate(fault_episodes):
        labels = ep["fault_labels"]
        if not labels:
            continue

        residuals = compute_residuals(state, ep["nodes"], ep["edges"], ep["ctx"], edge_index)
        deviation = compute_latent_deviation(state, ep["nodes"], ep["edges"], ep["ctx"], edge_index, N)
        T = residuals.shape[0]

        for fl in labels[:10]:
            svc = fl["service"]
            if svc not in svc_to_idx:
                continue
            rc = svc_to_idx[svc]

            rc_res = residuals[:, rc]
            valid = ~np.isnan(rc_res)
            if valid.sum() < 5:
                continue
            peak = int(np.nanargmax(rc_res))
            fs = max(0, peak - 5)
            fe = min(T, peak + 10)

            # Residual-only ranking
            fault_res = residuals[fs:fe]
            avg_res = np.nanmean(fault_res, axis=0)
            avg_res = np.nan_to_num(avg_res, nan=-1)
            rank_residual = np.argsort(-avg_res)
            rc_rank_residual = int(np.where(rank_residual == rc)[0][0]) + 1

            # Combined ranking (residual + temporal + topology + deviation)
            rank_combined, details = compute_combined_rank(
                residuals, fs, fe, edge_index, N, deviation, service_names
            )
            rc_rank_combined = int(np.where(rank_combined == rc)[0][0]) + 1

            results_rq3.append({
                "fault": f"{fl['type']}@{svc}",
                "rc": rc,
                "rank_residual": rc_rank_residual,
                "rank_combined": rc_rank_combined,
                "top1_residual": rc_rank_residual == 1,
                "top1_combined": rc_rank_combined == 1,
                "deviation_rc": float(deviation[rc]),
                "deviation_max_svc": service_names[int(np.argmax(deviation))],
                "improved": rc_rank_combined < rc_rank_residual,
            })

            if len(results_rq3) <= 15:
                arrow = "↑" if rc_rank_combined < rc_rank_residual else ("↓" if rc_rank_combined > rc_rank_residual else "=")
                print(f"{fl['type']}@{svc}: res={rc_rank_residual} → combined={rc_rank_combined} {arrow}"
                      f" | dev={deviation[rc]:.3f}")

    # RQ3 Summary
    print("\n--- RQ3 Summary ---")
    if results_rq3:
        residual_top1 = sum(1 for r in results_rq3 if r["top1_residual"])
        combined_top1 = sum(1 for r in results_rq3 if r["top1_combined"])
        improved = sum(1 for r in results_rq3 if r["improved"])
        print(f"Residual-only Top-1: {residual_top1}/{len(results_rq3)} ({100*residual_top1/max(1,len(results_rq3)):.0f}%)")
        print(f"Combined Top-1: {combined_top1}/{len(results_rq3)} ({100*combined_top1/max(1,len(results_rq3)):.0f}%)")
        print(f"Cases improved by evidence fusion: {improved}/{len(results_rq3)} ({100*improved/max(1,len(results_rq3)):.0f}%)")
        print(f"Avg residual rank: {np.mean([r['rank_residual'] for r in results_rq3]):.2f}")
        print(f"Avg combined rank: {np.mean([r['rank_combined'] for r in results_rq3]):.2f}")
        print(f"Avg latent deviation of RC: {np.mean([r['deviation_rc'] for r in results_rq3]):.3f}")
    else:
        print("No results.")

    # ================================================================
    # RQ4: Graph Ablation
    # ================================================================
    print("\n" + "=" * 60)
    print("RQ4: Dynamic Graph Ablation")
    print("=" * 60)
    print("(Using residual-only ranking for comparison)")

    # The ablation comparison:
    # A) No graph (MLP): each service independently predicted
    # B) Static graph: same graph for all timesteps
    # C) Dynamic graph (our full model): graph features change per timestep

    # Since we only have one trained model (full Graph-RSSM),
    # we run ablation by modifying the graph input:
    # 1. Full model (dynamic): use actual edge features
    # 2. Static graph: use mean edge features for all timesteps
    # 3. No graph: zero out all edge features (equivalent to no message passing)

    print("\nRQ4: Comparing prediction quality with different graph inputs")
    eval_nodes = val_nodes[:5]
    eval_edges = val_edges[:5]
    eval_ctx = val_ctx[:5]
    ei = jnp.array(edge_index, dtype=jnp.int32)

    # Full dynamic graph
    scores_dynamic = []
    for i in range(len(eval_nodes)):
        out = state.apply_fn(state.params, jax.random.PRNGKey(0),
                             jnp.array(eval_nodes[i:i+1]), jnp.array(eval_edges[i:i+1]),
                             ei, jnp.array(eval_ctx[i:i+1]), use_posterior=True)
        nt = jnp.array(eval_nodes[i:i+1, 1:, :, :])
        loss = float(gaussian_nll(out["metrics_mu"], out["metrics_logsigma"], nt))
        scores_dynamic.append(loss)

    # Static graph (mean edges)
    mean_edges = eval_edges.mean(axis=0, keepdims=True)  # [1, T, E, D]
    mean_edges_batch = np.repeat(mean_edges, len(eval_nodes), axis=0)
    scores_static = []
    for i in range(len(eval_nodes)):
        out = state.apply_fn(state.params, jax.random.PRNGKey(0),
                             jnp.array(eval_nodes[i:i+1]), jnp.array(mean_edges_batch[i:i+1]),
                             ei, jnp.array(eval_ctx[i:i+1]), use_posterior=True)
        nt = jnp.array(eval_nodes[i:i+1, 1:, :, :])
        loss = float(gaussian_nll(out["metrics_mu"], out["metrics_logsigma"], nt))
        scores_static.append(loss)

    # No graph (zero edges)
    zero_edges = np.zeros_like(eval_edges[:1])
    zero_edges_batch = np.repeat(zero_edges, len(eval_nodes), axis=0)
    scores_nograph = []
    for i in range(len(eval_nodes)):
        out = state.apply_fn(state.params, jax.random.PRNGKey(0),
                             jnp.array(eval_nodes[i:i+1]), jnp.array(zero_edges_batch[i:i+1]),
                             ei, jnp.array(eval_ctx[i:i+1]), use_posterior=True)
        nt = jnp.array(eval_nodes[i:i+1, 1:, :, :])
        loss = float(gaussian_nll(out["metrics_mu"], out["metrics_logsigma"], nt))
        scores_nograph.append(loss)

    print(f"\n  Dynamic graph NLL:    {np.mean(scores_dynamic):.4f} ± {np.std(scores_dynamic):.4f}")
    print(f"  Static graph NLL:     {np.mean(scores_static):.4f} ± {np.std(scores_static):.4f}")
    print(f"  No graph (edges=0):   {np.mean(scores_nograph):.4f} ± {np.std(scores_nograph):.4f}")

    # Relative improvement
    base = np.mean(scores_dynamic)
    print(f"\n  Dynamic vs Static: {100*(np.mean(scores_static)-base)/abs(base):+.1f}%")
    print(f"  Dynamic vs NoGraph: {100*(np.mean(scores_nograph)-base)/abs(base):+.1f}%")


if __name__ == "__main__":
    main()
