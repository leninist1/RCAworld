"""Phase 4c: Test RCAWorld on OpenRCA datasets (Bank, Telecom, Market).

Tests cross-domain generalization: world model trained on Online Boutique
evaluated on banking/telecom/cloud systems.
"""
import sys, os, json
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp, optax, h5py
from flax.training import train_state
import orbax.checkpoint as ocp
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models.graph_rssm import GraphRSSM
from training.losses import gaussian_nll_per_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENRCA_ROOT = "/home/dell2/RCA513/yyx/OpenRCA"


def load_pretrained_model():
    """Load Online Boutique-trained Graph-RSSM."""
    model = GraphRSSM(node_dim=8, edge_dim=4, ctx_dim=3, graph_hidden=128, num_heads=4,
                      det_dim=256, stoch_dim=32, stoch_classes=32)
    rng = jax.random.PRNGKey(42)
    init_rng, _ = jax.random.split(rng)
    params = model.init(init_rng, rng,
        jnp.ones((1,24,10,8)), jnp.ones((1,24,14,4)),
        jnp.zeros((2,14),dtype=jnp.int32), jnp.ones((1,24,3)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "phase2", "best")
    state = state.replace(params=ocp.PyTreeCheckpointer().restore(ckpt_path)["params"])
    return state, model


def parse_openrca_metrics(system: str, data_dir: str, max_days: int = 5) -> dict:
    """Parse OpenRCA metric files into node features [T, N, D_node]."""
    telemetry_dir = os.path.join(data_dir, "telemetry")
    days = sorted([d for d in os.listdir(telemetry_dir) if d.startswith("20")])[:max_days]

    all_nodes = []
    all_ctx = []
    service_set = set()

    # First pass: discover all unique services
    for day in days:
        metric_dir = os.path.join(telemetry_dir, day, "metric")
        # Try different metric file patterns
        for fname in ["metric_service.csv", "metric_app.csv"]:
            mfile = os.path.join(metric_dir, fname)
            if not os.path.exists(mfile):
                continue
            df = pd.read_csv(mfile)
            if 'service' in df.columns:
                service_set.update(df['service'].unique())
            elif 'tc' in df.columns:
                service_set.update(df['tc'].unique())
            elif 'serviceName' in df.columns:
                service_set.update(df['serviceName'].unique())
            break

    services = sorted(service_set)
    svc_to_idx = {s: i for i, s in enumerate(services)}
    N = len(services)
    print(f"  System: {system}, Services: {N}, Days: {len(days)}")

    # Second pass: build time series
    for day in days:
        metric_dir = os.path.join(telemetry_dir, day, "metric")
        mfile = None
        for fname in ["metric_service.csv", "metric_app.csv"]:
            candidate = os.path.join(metric_dir, fname)
            if os.path.exists(candidate):
                mfile = candidate
                break
        if mfile is None:
            continue

        df = pd.read_csv(mfile)

        # Normalize timestamp column name
        ts_col = None
        for col in ['timestamp', 'startTime', 'time']:
            if col in df.columns:
                ts_col = col
                break
        if ts_col is None:
            continue
        if ts_col == 'startTime':
            df['timestamp'] = (df['startTime'].astype(float) / 1000).astype(int)
            ts_col = 'timestamp'

        df[ts_col] = df[ts_col].astype(int)
        svc_col = next((c for c in ['service', 'tc', 'serviceName'] if c in df.columns), None)
        if svc_col is None:
            continue

        timestamps = sorted(df[ts_col].unique())
        t_to_idx = {t: i for i, t in enumerate(timestamps)}
        T_day = len(timestamps)

        node_day = np.zeros((T_day, N, 8), dtype=np.float32)
        ctx_day = np.zeros((T_day, 3), dtype=np.float32)

        for _, row in df.iterrows():
            svc = row[svc_col]
            if svc not in svc_to_idx:
                continue
            t = t_to_idx.get(int(row[ts_col]))
            if t is None:
                continue
            n = svc_to_idx[svc]

            # Map features (system-agnostic feature names → our 8-d space)
            rr = float(row.get('rr', row.get('num', row.get('cnt', 0))))  # request rate / count
            sr = float(row.get('sr', row.get('succee_rate', row.get('success_rate', 100))))  # success rate
            mrt = float(row.get('mrt', row.get('avg_time', row.get('latency', 0))))  # latency

            node_day[t, n, 6] = rr   # workload
            node_day[t, n, 7] = sr   # success rate
            node_day[t, n, 4] = mrt  # latency p90
            node_day[t, n, 5] = mrt * 1.5  # p99 ≈ 1.5×

        node_day = np.nan_to_num(node_day, nan=0.0)

        ctx_day[:, 0] = node_day[:, :, 6].sum(axis=1)
        ctx_day[:, 1] = node_day[:, :, 7].mean(axis=1)
        ctx_day[:, 2] = (node_day[:, :, 6] > 0).sum(axis=1)

        all_nodes.append(node_day)
        all_ctx.append(ctx_day)

    if not all_nodes:
        return None

    nodes = np.concatenate(all_nodes, axis=0)
    ctx = np.concatenate(all_ctx, axis=0)
    return {
        "nodes": nodes, "ctx": ctx, "services": services,
        "svc_to_idx": svc_to_idx,
    }


def parse_openrca_faults(system: str, data_dir: str) -> list:
    """Parse record.csv into fault list with service mapping."""
    record_file = os.path.join(data_dir, "record.csv")
    if not os.path.exists(record_file):
        return []

    df = pd.read_csv(record_file)
    faults = []
    for _, row in df.iterrows():
        component = str(row.get('component', ''))
        faults.append({
            "component": component,
            "reason": str(row.get('reason', '')),
            "timestamp": float(row['timestamp']),
            "level": str(row.get('level', 'pod')),
        })
    return faults


def map_component_to_service(component: str, services: list, svc_to_idx: dict) -> int:
    """Fuzzy map a fault component (pod/docker) to a service index."""
    comp_lower = component.lower().replace("_", "").replace("-", "").replace(" ", "")

    # Direct match (e.g., "shippingservice-1" → "shippingservice")
    for svc in services:
        svc_lower = svc.lower().replace("_", "").replace("-", "").replace(" ", "")
        if comp_lower == svc_lower:
            return svc_to_idx[svc]

    # Substring: "shippingservice-1" contains "shippingservice" or vice versa
    for svc in services:
        svc_lower = svc.lower().replace("_", "").replace("-", "").replace(" ", "")
        if svc_lower in comp_lower or comp_lower in svc_lower:
            return svc_to_idx[svc]

    # For Bank: map infrastructure pods to first few service slots
    # (pods like Mysql02 aren't services - approximate mapping)
    infra_types = ["mysql", "redis", "tomcat", "apache", "mg", "docker", "java", "node"]
    for i, itype in enumerate(infra_types):
        if itype in comp_lower:
            return i % len(services)

    return -1


def compute_residuals_batch(state, nodes, edges, ctx, ei_arr, ws=24, bmax=10):
    """Batch residual computation."""
    T, N_ = nodes.shape[0], nodes.shape[1]
    all_res = np.full((T, N_), np.nan, dtype=np.float32)
    elems = []
    for start in range(0, max(1, T - ws), max(1, ws // 2)):
        end = min(T, start + ws)
        if end - start < 4:
            continue
        w = nodes[start:end]; we = edges[start:end]; wc = ctx[start:end]
        if w.shape[0] < ws:
            pad = ws - w.shape[0]
            w = np.pad(w, ((0,pad),(0,0),(0,0)))
            we = np.pad(we, ((0,pad),(0,0),(0,0)))
            wc = np.pad(wc, ((0,pad),(0,0)))
        elems.append((w, we, wc, start, end))
    if not elems:
        return all_res
    ei = jnp.array(ei_arr, dtype=jnp.int32)
    for i in range(0, len(elems), bmax):
        ck = elems[i:i+bmax]
        bn = jnp.array(np.stack([x[0] for x in ck]))
        be = jnp.array(np.stack([x[1] for x in ck]))
        bw = jnp.array(np.stack([x[2] for x in ck]))
        out = state.apply_fn(state.params, jax.random.PRNGKey(0), bn, be, ei, bw, use_posterior=True)
        nt = bn[:, 1:, :, :]
        res = np.array(gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt))
        for j, (_, _, _, start, end) in enumerate(ck):
            ns = min(res.shape[1], end - start)
            all_res[start + 1:end] = res[j, :ns]
    return all_res


def evaluate_system(system_name: str, data_dir: str, state, model, max_days: int = 5):
    """Evaluate RCA on one OpenRCA system."""
    print(f"\n{'='*60}")
    print(f"  {system_name}")
    print(f"{'='*60}")

    # Parse data
    data = parse_openrca_metrics(system_name, data_dir, max_days)
    if data is None:
        print("  No metric data found, skipping.")
        return None
    faults = parse_openrca_faults(system_name, data_dir)
    if not faults:
        print("  No fault labels found, skipping.")
        return None

    print(f"  Faults: {len(faults)}")

    # Normalize using simple stats
    nodes_raw = data["nodes"]
    n_mean = nodes_raw.mean(axis=(0, 1), keepdims=True) + 1e-6
    n_std = nodes_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    nodes_n = (nodes_raw - n_mean) / n_std
    nodes_n = np.nan_to_num(nodes_n, nan=0.0)

    T, N = nodes_n.shape[0], nodes_n.shape[1]

    # Edge features (dummy, no graph)
    EI = min(14, max(1, N * 2))  # use some edges
    edge_index_arr = np.zeros((2, EI), dtype=np.int32)
    for e in range(min(EI, N)):
        edge_index_arr[0, e] = e
        edge_index_arr[1, e] = (e + 1) % N
    edges_arr = np.zeros((T, EI, 4), dtype=np.float32)
    edges_arr[:, :, 2] = 1.0

    # Compute residuals
    print(f"  Computing residuals (T={T}, N={N})...")
    residuals = compute_residuals_batch(state, nodes_n, edges_arr, data["ctx"], edge_index_arr)
    print(f"  Residuals shape: {residuals.shape}")

    # Evaluate
    results = []
    for fl in faults:
        rc = map_component_to_service(fl["component"], data["services"], data["svc_to_idx"])
        if rc < 0 or rc >= N:
            continue
        rc_res = residuals[:, rc]
        valid = ~np.isnan(rc_res)
        if valid.sum() < 5:
            continue
        peak = int(np.nanargmax(rc_res))
        fs = max(0, peak - 3)
        fe = min(T, peak + 5)
        fault_res = residuals[fs:fe]
        avg_res = np.nan_to_num(np.nanmean(fault_res, axis=0), nan=-99)
        rank = np.argsort(-avg_res)
        rc_rank = int(np.where(rank == rc)[0][0]) + 1
        results.append({
            "fault": f"{fl['reason']}@{fl['component']}",
            "rank": rc_rank, "top1": rc_rank == 1,
            "top3": rc_rank <= 3, "mrr": 1.0 / rc_rank,
        })

    # Summary
    n = len(results)
    if n == 0:
        print("  No faults mapped, skipping.")
        return None
    t1 = sum(1 for r in results if r["top1"])
    t3 = sum(1 for r in results if r["top3"])
    mrr = np.mean([r["mrr"] for r in results])
    ar = np.mean([r["rank"] for r in results])

    print(f"  Evaluated: {n}/{len(faults)} faults mapped")
    print(f"  Top-1: {t1}/{n} ({100*t1/n:.0f}%)")
    print(f"  Top-3: {t3}/{n} ({100*t3/n:.0f}%)")
    print(f"  MRR: {mrr:.3f}")
    print(f"  Avg Rank: {ar:.2f} (chance: {N/2:.1f})")

    return {
        "system": system_name, "n_services": N, "n_faults": n,
        "top1": t1, "top3": t3, "mrr": mrr, "avg_rank": ar,
    }


def main():
    print(f"Device: {jax.devices()[0]}")
    print("Loading pretrained model (Online Boutique)...")
    state, model = load_pretrained_model()
    print("Model loaded.\n")

    results_all = []

    # Test Bank (banking system, 11 services, 136 faults)
    r_bank = evaluate_system(
        "Bank", os.path.join(OPENRCA_ROOT, "Bank", "Bank"), state, model
    )
    if r_bank:
        results_all.append(r_bank)

    # Test Telecom (telecom infrastructure)
    r_tel = evaluate_system(
        "Telecom", os.path.join(OPENRCA_ROOT, "Telecom", "Telecom"), state, model
    )
    if r_tel:
        results_all.append(r_tel)

    # Test Market cloudbed-1 (microservice-like)
    r_mkt1 = evaluate_system(
        "Market/cloudbed-1", os.path.join(OPENRCA_ROOT, "Market", "Market", "cloudbed-1"),
        state, model
    )
    if r_mkt1:
        results_all.append(r_mkt1)

    # Test Market cloudbed-2
    r_mkt2 = evaluate_system(
        "Market/cloudbed-2", os.path.join(OPENRCA_ROOT, "Market", "Market", "cloudbed-2"),
        state, model
    )
    if r_mkt2:
        results_all.append(r_mkt2)

    # Final comparison
    print("\n" + "=" * 60)
    print("Cross-Domain Generalization Summary")
    print("=" * 60)
    print(f"{'System':<20s} {'N_svc':>6s} {'N_faults':>8s} {'Top-1':>8s} {'Top-3':>8s} {'MRR':>8s} {'AvgR':>6s}")
    print("-" * 70)
    for r in results_all:
        print(f"{r['system']:<20s} {r['n_services']:>6d} {r['n_faults']:>8d} "
              f"{r['top1']:>7d} {r['top3']:>7d} {r['mrr']:>8.3f} {r['avg_rank']:>6.2f}")
    print("-" * 70)
    print(f"{'Online Boutique (src)':<20s} {'10':>6s} {'24':>8s} {'24':>8s} {'24':>8s} {'1.000':>8s} {'1.00':>6s}")
    print(f"{'OB→TrainTicket':<20s} {'10':>6s} {'14':>8s} {'11':>8s} {'14':>8s} {'0.893':>8s} {'1.21':>6s}")


if __name__ == "__main__":
    main()
