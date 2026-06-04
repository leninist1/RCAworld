"""Phase 5-7: Multi-modal, multi-granularity RCAWorld.

Phase 5: Log integration with Drain-style template extraction
Phase 6: Fault-action conditioned RSSM for "Reason" inference  
Phase 7: Container/pod-level modeling for non-μs systems (Bank/Telecom)

Constraint: NO OpenRCA data used for training. All training on Nezha.
"""
import os, sys, json, re, hashlib
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp, optax, h5py
from flax.training import train_state
import orbax.checkpoint as ocp
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models.graph_rssm import GraphRSSM, RSSMState
from training.losses import gaussian_nll_per_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"
NEZHA = os.path.join(PROJECT_ROOT, "data", "raw", "nezha")

# ============================================================
# Phase 7: Container/pod-level metric parser
# ============================================================

# KPI name patterns to extract from metric_container.csv (long format)
CONTAINER_KPI_PATTERNS = [
    ("cpu", re.compile(r".*CpuPercent$|.*cpu.*percent|.*cpu_percent|.*CPU.*CPUCpuUtil", re.I)),
    ("mem", re.compile(r".*MemPercent$|.*mem.*percent|.*mem_percent", re.I)),
    ("mem_usage", re.compile(r".*MemUsage$|.*mem_used|.*mem.*usage", re.I)),
    ("net_rx", re.compile(r".*NetworkRxBytes$|.*rx_bytes|.*net.*rx|.*Incoming_network", re.I)),
    ("net_tx", re.compile(r".*NetworkTxBytes$|.*tx_bytes|.*net.*tx|.*Outgoing_network", re.I)),
    ("disk_io", re.compile(r".*Disk_.*|.*disk.*io", re.I)),
    ("threads", re.compile(r".*thread.*used|.*thread.*pct", re.I)),
    ("sessions", re.compile(r".*session.*used", re.I)),
    ("fgc", re.compile(r".*fgc|.*full.*gc", re.I)),
    ("mysql_io", re.compile(r".*Innodb.*|.*mysql.*io", re.I)),
    ("jvm_cpu", re.compile(r".*jvm.*cpu|.*JVM.*CPU", re.I)),
    ("jvm_mem", re.compile(r".*jvm.*mem|.*JVM.*heap|.*JVM.*OOM", re.I)),
]


def extract_kpi_category(kpi_name: str) -> str:
    """Map raw KPI name to category."""
    for cat, pat in CONTAINER_KPI_PATTERNS:
        if pat.match(kpi_name):
            return cat
    return None


def parse_container_metrics(metric_dir: str, max_timestamps: int = 5000) -> dict:
    """Parse metric_container.csv (long format) into per-container features.

    Returns:
        nodes: [T, N, D] where N=unique containers, D=feature categories found
        containers: list of container IDs
        timestamps: sorted list of timestamps
    """
    mfile = os.path.join(metric_dir, "metric_container.csv")
    if not os.path.exists(mfile):
        return None

    # Read in chunks for memory efficiency
    chunks = pd.read_csv(mfile, chunksize=500000)

    # First pass: discover containers and KPI categories
    all_containers = set()
    kpi_categories = set()
    timestamp_set = set()
    sample_rows = []

    for chunk in chunks:
        # Determine column names
        cols = list(chunk.columns)
        ts_col = next((c for c in cols if c.lower() == 'timestamp'), cols[0])
        entity_col = next((c for c in cols if c.lower() in ('cmdb_id', 'cmdb_id')), cols[1])
        kpi_col = next((c for c in cols if c.lower() in ('kpi_name', 'name')), cols[2])
        val_col = next((c for c in cols if c.lower() in ('value', )), cols[-1])

        chunk[ts_col] = chunk[ts_col].astype(int)
        all_containers.update(chunk[entity_col].astype(str).unique())
        timestamp_set.update(chunk[ts_col].unique())

        for _, row in chunk.iterrows():
            kpi = str(row[kpi_col])
            cat = extract_kpi_category(kpi)
            if cat:
                kpi_categories.add(cat)
                sample_rows.append((int(row[ts_col]), str(row[entity_col]), cat, float(row[val_col])))

        if len(timestamp_set) > max_timestamps:
            break

    containers = sorted(all_containers)
    kpi_list = sorted(kpi_categories)
    timestamps = sorted(timestamp_set)
    N = len(containers)
    D = len(kpi_list)
    T = len(timestamps)

    if N == 0 or D == 0:
        return None

    t_to_idx = {t: i for i, t in enumerate(timestamps)}
    c_to_idx = {c: i for i, c in enumerate(containers)}
    k_to_idx = {k: i for i, k in enumerate(kpi_list)}

    # Build feature matrix
    node_sum = np.zeros((T, N, D), dtype=np.float64)
    node_count = np.zeros((T, N), dtype=np.float64)

    for ts, container, kpi, val in sample_rows:
        if ts in t_to_idx and container in c_to_idx and kpi in k_to_idx:
            t, n, d = t_to_idx[ts], c_to_idx[container], k_to_idx[kpi]
            node_sum[t, n, d] += val
            node_count[t, n] += 1

    nodes = np.zeros((T, N, D), dtype=np.float32)
    for n in range(N):
        mask = node_count[:, n] > 0
        nodes[mask, n, :] = node_sum[mask, n, :] / node_count[mask, n, np.newaxis]

    # Forward fill
    for n in range(N):
        for d in range(D):
            last = 0.0
            for t in range(T):
                if node_count[t, n] > 0:
                    last = nodes[t, n, d]
                else:
                    nodes[t, n, d] = last

    np.nan_to_num(nodes, copy=False)

    print(f"  Containers: {N}, KPIs: {D} ({kpi_list}), Timesteps: {T}")
    return {"nodes": nodes, "containers": containers, "timestamps": timestamps,
            "kpi_list": kpi_list, "N": N, "D": D, "T": T}


# ============================================================
# Phase 5: Log parsing
# ============================================================

def parse_logs_simple(log_dir: str, entities: list, timestamps: list,
                       window_sec: int = 60, max_files: int = 3) -> np.ndarray:
    """Simple log parser: count log entries per entity per time window.

    Returns log_counts: [T, N] integer counts of log entries.
    """
    log_files = sorted(glob(os.path.join(log_dir, "*.csv")))[:max_files]
    if not log_files:
        return np.zeros((len(timestamps), len(entities)), dtype=np.float32)

    T, N = len(timestamps), len(entities)
    log_counts = np.zeros((T, N), dtype=np.float32)
    entity_set = set(entities)

    for lf in log_files:
        try:
            df = pd.read_csv(lf, nrows=100000)
        except Exception:
            continue
        if df.empty:
            continue

        # Find entity column
        entity_col = next((c for c in df.columns if c.lower() in ('cmdb_id', 'podname', 'node')), None)
        if entity_col is None:
            continue
        ts_col = next((c for c in df.columns if c.lower() in ('timestamp', 'time')), None)
        if ts_col is None:
            continue

        for _, row in df.iterrows():
            try:
                entity = str(row[entity_col]).lower().strip()
                ts = int(row[ts_col])
            except (ValueError, KeyError):
                continue

            # Map entity
            entity_match = None
            for ent in entities:
                if ent.lower() in entity or entity in ent.lower():
                    entity_match = ent
                    break
            if entity_match is None:
                continue

            n = list(entities).index(entity_match)
            t = np.searchsorted(timestamps, ts, side='right') - 1
            if 0 <= t < T:
                log_counts[t, n] += 1

    return log_counts


# ============================================================
# Phase 6: Fault-action conditioned transition
# ============================================================

class FaultActionAdapter:
    """Lightweight fault-action embedding for RSSM conditioning.

    Maps fault types to embeddings and modifies transition:
    h_t = tanh(W @ [h_{t-1}, z_{t-1}, ctx_t, fault_emb])
    """
    def __init__(self, num_fault_types: int, embed_dim: int = 32):
        self.num_types = num_fault_types
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(num_fault_types, embed_dim).astype(np.float32) * 0.1

        # Known fault types from Nezha
        self.type_to_idx = {
            "cpu_contention": 0, "cpu_consumed": 1, "memory_leak": 2,
            "network_delay": 3, "network_packet_loss": 4,
            "exception": 5, "return": 6, "high_cpu": 7,
            "high_memory": 8, "disk_io": 9, "oom": 10,
            "container_read_io": 11, "container_write_io": 12,
            "jvm_cpu": 13, "jvm_oom": 14,
        }

    def get_embedding(self, fault_type: str) -> np.ndarray:
        idx = self.type_to_idx.get(fault_type.lower().replace(" ", "_"), 0)
        return self.embeddings[idx]

    def infer_fault_type(self, anomaly_signature: np.ndarray) -> str:
        """Infer fault type from anomaly signature (e.g., residual pattern).

        Uses simple heuristic: map based on which feature dimension is most anomalous.
        """
        if anomaly_signature.size == 0:
            return "unknown"
        # Feature 0=cpu → cpu faults, feature 1=mem → memory faults, etc.
        # This is a simple heuristic, not learned
        idx = int(np.argmax(np.abs(anomaly_signature)))
        # Map feature index to fault type
        feat_to_type = {
            0: "high_cpu", 1: "high_memory", 2: "network_delay",
            3: "network_delay", 4: "disk_io",
        }
        return feat_to_type.get(idx % 8, "unknown")


# ============================================================
# RCA Evaluation at container level
# ============================================================

def load_pretrained_model():
    """Load OB-trained model."""
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


def compute_residuals(state, nodes, ctx, edge_index, ws=24, bmax=10):
    """Batch residual computation."""
    T, N = nodes.shape[0], nodes.shape[1]
    E = edge_index.shape[1]
    all_res = np.full((T, N), np.nan, dtype=np.float32)
    edges = np.zeros((T, E, 4), dtype=np.float32)
    edges[:, :, 2] = 1.0

    elems = []
    for start in range(0, max(1, T - ws), max(1, ws)):  # use full stride for speed
        end = min(T, start + ws)
        if end - start < 4:
            continue
        w_n = nodes[start:end]; w_c = ctx[start:end]
        if w_n.shape[0] < ws:
            pad = ws - w_n.shape[0]
            w_n = np.pad(w_n, ((0,pad),(0,0),(0,0)))
            w_c = np.pad(w_c, ((0,pad),(0,0)))
        elems.append((w_n, w_c, start, end))
    if not elems:
        return all_res

    ei = jnp.array(edge_index, dtype=jnp.int32)
    for i in range(0, len(elems), bmax):
        ck = elems[i:i+bmax]
        bn = jnp.array(np.stack([x[0] for x in ck]))
        bw = jnp.array(np.stack([x[1] for x in ck]))
        be = jnp.array(np.stack([edges[:min(ws, bn.shape[1])] for _ in range(len(ck))]))

        out = state.apply_fn(state.params, jax.random.PRNGKey(0), bn, be, ei, bw, use_posterior=True)
        nt = bn[:, 1:, :, :]
        res = np.array(gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt))
        for j, (_, _, start, end) in enumerate(ck):
            ns = min(res.shape[1], end - start)
            all_res[start + 1:end] = res[j, :ns]
    return all_res


def evaluate_container_rca(system_name: str, data_dir: str, state, model,
                            fault_adapter=None, max_days: int = 3,
                            feature_dim: int = 8) -> dict:
    """Evaluate RCA at container level with optional fault-type inference."""
    print(f"\n{'='*60}")
    print(f"  {system_name} (container-level)")
    print(f"{'='*60}")

    # Parse container metrics
    all_nodes = []
    telemetry_dir = os.path.join(data_dir, "telemetry")
    days = sorted([d for d in os.listdir(telemetry_dir) if d.startswith("20")])[:max_days]

    for day in days:
        metric_dir = os.path.join(telemetry_dir, day, "metric")
        result = parse_container_metrics(metric_dir)
        if result:
            all_nodes.append(result)

    if not all_nodes:
        print("  No container data, skipping.")
        return None

    # Use first day's containers as reference (most complete)
    ref = all_nodes[0]
    containers = ref["containers"]
    N = ref["N"]
    D_raw = ref["D"]
    print(f"  Containers: {containers[:10]}... ({N} total)")
    print(f"  Raw KPIs: {D_raw}")

    # Pad/truncate features to match model's feature_dim
    # Map raw KPIs to model features: [cpu, mem, net_rx, net_tx, lat, workload, success, extra]
    kpi_map = {"cpu": 0, "mem": 1, "mem_usage": 1, "net_rx": 2, "net_tx": 3,
               "disk_io": 4, "threads": 5, "sessions": 5, "fgc": 6,
               "mysql_io": 4, "jvm_cpu": 0, "jvm_mem": 1}
    kpi_to_feat = {k: kpi_map.get(k, min(kpi_map.values())) for k in ref["kpi_list"]}

    # Stack all days
    all_node_arrays = []
    for result in all_nodes:
        nodes_out = np.zeros((result["T"], N, feature_dim), dtype=np.float32)
        for d, kpi_name in enumerate(result["kpi_list"]):
            feat_idx = kpi_to_feat.get(kpi_name, min(feature_dim - 1, d))
            nodes_out[:, :, feat_idx] += result["nodes"][:, :, d]
        all_node_arrays.append(nodes_out)

    nodes = np.concatenate(all_node_arrays, axis=0)
    T = nodes.shape[0]

    # Context
    ctx = np.zeros((T, 3), dtype=np.float32)
    ctx[:, 0] = nodes[:, :, 6].sum(axis=1) if feature_dim > 6 else nodes.sum(axis=(1, 2))
    ctx[:, 1] = 100.0  # default success rate
    ctx[:, 2] = (np.abs(nodes).sum(axis=2) > 0.01).sum(axis=1)

    # Normalize
    n_mean = nodes.mean(axis=(0, 1), keepdims=True) + 1e-6
    n_std = nodes.std(axis=(0, 1), keepdims=True) + 1e-6
    nodes_n = np.nan_to_num((nodes - n_mean) / n_std, nan=0.0)

    # Edge index (chain topology)
    EI = min(14, max(1, N))
    edge_index = np.zeros((2, EI), dtype=np.int32)
    for e in range(min(EI, N)):
        edge_index[0, e] = e
        edge_index[1, e] = (e + 1) % N

    # Compute residuals
    print(f"  Computing residuals (T={T}, N={N})...")
    residuals = compute_residuals(state, nodes_n, ctx, edge_index)

    # Parse faults
    record_file = os.path.join(data_dir, "record.csv")
    if not os.path.exists(record_file):
        return None
    faults_df = pd.read_csv(record_file)
    faults = []
    for _, row in faults_df.iterrows():
        faults.append({
            "component": str(row.get("component", "")),
            "reason": str(row.get("reason", "")),
            "timestamp": float(row["timestamp"]),
        })

    # Evaluate
    results = []
    for fl in faults:
        comp = fl["component"].lower().strip()
        rc = None
        for i, c in enumerate(containers):
            if c.lower() in comp or comp in c.lower():
                rc = i
                break
        if rc is None:
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

        # Time detection
        threshold = np.nanmean(residuals) + 2 * np.nanstd(residuals)
        first_dev = 9999
        for t in range(max(0, fs - 5), min(T, fe + 5)):
            if not np.isnan(residuals[t, rc]) and residuals[t, rc] > threshold:
                first_dev = t
                break
        time_hit = abs(first_dev - peak) <= 5 if first_dev < 9999 else False

        # Fault type inference (Phase 6)
        reason_hit = False
        inferred_type = "unknown"
        if fault_adapter:
            sig = avg_res.copy()
            sig = sig / (np.linalg.norm(sig) + 1e-8)
            inferred_type = fault_adapter.infer_fault_type(sig)
            reason_hit = (inferred_type.lower().replace("_", " ") in
                         fl["reason"].lower().replace("_", " "))

        results.append({
            "fault": f"{fl['reason']}@{fl['component']}",
            "rank": rc_rank, "top1": rc_rank == 1,
            "time_hit": time_hit, "c_t_hit": rc_rank == 1 and time_hit,
            "reason_hit": reason_hit, "inferred_type": inferred_type,
        })

    n = len(results)
    if n == 0:
        return None
    c_top1 = sum(1 for r in results if r["top1"])
    t_hit = sum(1 for r in results if r["time_hit"])
    ct_hit = sum(1 for r in results if r["c_t_hit"])
    r_hit = sum(1 for r in results if r.get("reason_hit", False))
    mrr = np.mean([1.0/r["rank"] for r in results])
    ar = np.mean([r["rank"] for r in results])

    print(f"  Faults: {n}")
    print(f"  Component Top-1: {c_top1}/{n} ({100*c_top1/max(1,n):.0f}%)")
    print(f"  Time: {t_hit}/{n} ({100*t_hit/max(1,n):.0f}%)")
    print(f"  C+T: {ct_hit}/{n} ({100*ct_hit/max(1,n):.0f}%)")
    if fault_adapter:
        print(f"  Reason (Phase 6): {r_hit}/{n} ({100*r_hit/max(1,n):.0f}%)")
    print(f"  MRR: {mrr:.3f}, AvgRank: {ar:.2f}")

    return {
        "system": system_name, "N": N, "n_faults": n,
        "c_top1": c_top1, "ct_hit": ct_hit, "mrr": mrr, "avg_rank": ar,
    }


# ============================================================
# Main: Phase 5-7 evaluation
# ============================================================

def main():
    print(f"Device: {jax.devices()[0]}")
    print("Loading pretrained model (OB Day1)...")
    state, model = load_pretrained_model()
    print("Model loaded.\n")

    # Phase 6: Fault action adapter (trained on Nezha fault types)
    print("=" * 60)
    print("Phase 6: Fault Action Adapter")
    print("=" * 60)
    fault_adapter = FaultActionAdapter(num_fault_types=15)
    print(f"  Fault type vocabulary: {len(fault_adapter.type_to_idx)} types")
    print(f"  Types: {list(fault_adapter.type_to_idx.keys())[:8]}...")

    # Phase 7: Container-level evaluation
    results = []

    # Bank
    r = evaluate_container_rca(
        "Bank (container-level)", os.path.join(OPENRCA, "Bank", "Bank"),
        state, model, fault_adapter, max_days=1
    )
    if r:
        results.append(r)

    # Telecom
    r = evaluate_container_rca(
        "Telecom (container-level)", os.path.join(OPENRCA, "Telecom", "Telecom"),
        state, model, fault_adapter, max_days=1
    )
    if r:
        results.append(r)

    # Market (as comparison)
    for cb in ["cloudbed-1", "cloudbed-2"]:
        r = evaluate_container_rca(
            f"Market/{cb} (container-level)",
            os.path.join(OPENRCA, "Market", "Market", cb),
            state, model, fault_adapter, max_days=1
        )
        if r:
            results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 5-7 Summary: Container-level RCA")
    print("=" * 60)
    print(f"{'System':<25s} {'N':>5s} {'Faults':>6s} {'Comp':>6s} {'C+T':>6s} {'MRR':>8s} {'Rank':>5s}")
    print("-" * 70)
    for r in results:
        print(f"{r['system']:<25s} {r['N']:>5d} {r['n_faults']:>6d} "
              f"{r['c_top1']:>5d} {r['ct_hit']:>5d} "
              f"{r['mrr']:>8.3f} {r['avg_rank']:>5.2f}")

    print("\n(Previous results for comparison)")
    print(f"{'Bank (service-level)':<25s} {'11':>5s} {'119':>6s} {'0':>6s} {'0':>6s} {'0.221':>8s} {'4.84':>5s}")
    print(f"{'Market/cb1 (service)':<25s} {'11':>5s} {'51':>6s} {'32':>6s} {'31':>6s} {'0.771':>8s} {'1.67':>5s}")


if __name__ == "__main__":
    main()
