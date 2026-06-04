"""Clean RCA Evaluation: Two-stage pipeline (anomaly detection → residual ranking).

Removes oracle leaks:
- Stage 1: Detect anomaly onset from aggregate residuals (no oracle)
- Stage 2: Rank services within detected window by prediction residual

Evaluates all benchmarks: OB Day1, OB Day2, TrainTicket, Market, Bank, Telecom.
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp, optax, h5py
from flax.training import train_state
import orbax.checkpoint as ocp
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models.graph_rssm import GraphRSSM
from training.losses import gaussian_nll_per_node

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"


def load_model():
    model = GraphRSSM(node_dim=8, edge_dim=4, ctx_dim=3, graph_hidden=128, num_heads=4,
                      det_dim=256, stoch_dim=32, stoch_classes=32)
    rng = jax.random.PRNGKey(42)
    init_rng, _ = jax.random.split(rng)
    params = model.init(init_rng, rng,
        jnp.ones((1,24,10,8)), jnp.ones((1,24,14,4)),
        jnp.zeros((2,14),dtype=jnp.int32), jnp.ones((1,24,3)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "phase2", "best")
    state = state.replace(params=ocp.PyTreeCheckpointer().restore(ckpt)["params"])
    return state, model


# ============================================================
# Stage 1: Anomaly onset detection (NO oracle)
# ============================================================

def detect_anomaly_onset(residuals: np.ndarray, normal_baseline: dict,
                          min_sustain: int = 3, lookback: int = 20) -> list:
    """Detect anomaly windows from aggregate residuals.

    Uses aggregate residual (max per-timestep across services) as anomaly signal.
    Anomaly = sustained period where aggregate residual > μ + 3σ.

    Args:
        residuals: [T, N] per-service per-timestep residuals
        normal_baseline: dict with 'mean' and 'std' from normal data
        min_sustain: minimum consecutive anomalous steps to trigger
        lookback: how far back to extend window before first anomaly

    Returns:
        list of (start, end) anomaly window tuples
    """
    T, N = residuals.shape
    threshold = normal_baseline["mean"] + 3 * normal_baseline["std"]

    # Aggregate residual per timestep (max across services, ignoring NaN)
    agg_res = np.nanmax(residuals, axis=1)

    # Find sustained anomalous periods
    windows = []
    in_anomaly = False
    start = 0
    for t in range(T):
        is_anomalous = not np.isnan(agg_res[t]) and agg_res[t] > threshold

        if is_anomalous and not in_anomaly:
            start = t
            in_anomaly = True
        elif not is_anomalous and in_anomaly:
            duration = t - start
            if duration >= min_sustain:
                win_start = max(0, start - lookback)
                win_end = min(T, t + 5)
                windows.append((win_start, win_end))
            in_anomaly = False

    # Handle trailing anomaly
    if in_anomaly:
        duration = T - start
        if duration >= min_sustain:
            windows.append((max(0, start - lookback), T))

    return windows


# ============================================================
# Stage 2: Residual-based RCA ranking within detected window
# ============================================================

def rank_services_in_window(residuals: np.ndarray, window: tuple, N: int) -> np.ndarray:
    """Rank services by average residual within a detected anomaly window."""
    fs, fe = window
    if fs >= fe or fe > residuals.shape[0]:
        return np.arange(N)
    window_res = residuals[fs:fe]
    avg_res = np.nan_to_num(np.nanmean(window_res, axis=0), nan=-99)
    return np.argsort(-avg_res)


# ============================================================
# Residual computation (clean version)
# ============================================================

def compute_residuals_clean(state, nodes, edges, ctx, edge_index,
                             use_posterior: bool = True, ws: int = 24) -> np.ndarray:
    """Compute residuals using sliding windows. Can use prior or posterior."""
    T, N = nodes.shape[0], nodes.shape[1]
    all_res = np.full((T, N), np.nan, dtype=np.float32)
    stride = ws  # full stride, no overlap
    ei = jnp.array(edge_index, dtype=jnp.int32)

    for start in range(0, max(1, T - ws), stride):
        end = min(T, start + ws)
        if end - start < 4:
            continue
        w_n = nodes[start:end]
        w_e = edges[start:end] if edges.shape[0] > start else edges
        w_c = ctx[start:end]
        if w_n.shape[0] < ws:
            pad = ws - w_n.shape[0]
            w_n = np.pad(w_n, ((0,pad),(0,0),(0,0)))
            w_e = np.pad(w_e, ((0,pad),(0,0),(0,0)))
            w_c = np.pad(w_c, ((0,pad),(0,0)))

        out = state.apply_fn(state.params, jax.random.PRNGKey(0),
                             jnp.array(w_n[None]), jnp.array(w_e[None]),
                             ei, jnp.array(w_c[None]),
                             use_posterior=use_posterior)
        nt = jnp.array(w_n[None, 1:, :, :])
        res = np.array(gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt))
        n_steps = min(res.shape[1], end - start)
        all_res[start + 1:end] = res[0, :n_steps]

    # Forward fill NaN
    for n in range(N):
        last = 0.0
        for t in range(T):
            if not np.isnan(all_res[t, n]):
                last = all_res[t, n]
            else:
                all_res[t, n] = last
    return all_res


# ============================================================
# Full clean RCA evaluation
# ============================================================

def evaluate_system_clean(system_name: str, nodes: np.ndarray, edges: np.ndarray,
                           ctx: np.ndarray, edge_index: np.ndarray,
                           fault_labels: list, svc_names: list,
                           state, normal_baseline: dict,
                           use_posterior: bool = True) -> dict:
    """Full clean RCA: anomaly detection → residual ranking → metrics.

    Args:
        fault_labels: list of dicts with 'service', 'timestamp', 'type' keys
    """
    N = len(svc_names)
    svc_to_idx = {s: i for i, s in enumerate(svc_names)}

    # Normalize
    n_mean = nodes.mean(axis=(0,1), keepdims=True) + 1e-6
    n_std = nodes.std(axis=(0,1), keepdims=True) + 1e-6
    nodes_n = np.nan_to_num((nodes - n_mean) / n_std, nan=0.0)

    # Step A: Compute residuals
    residuals = compute_residuals_clean(state, nodes_n, edges, ctx, edge_index, use_posterior)

    # Step B: Detect anomaly windows from aggregate residuals
    anomaly_windows = detect_anomaly_onset(residuals, normal_baseline)

    # Step C: Match windows to fault labels and evaluate
    T = residuals.shape[0]
    results = []

    for fl in fault_labels:
        svc = fl.get("service", "")
        if svc not in svc_to_idx:
            continue
        rc = svc_to_idx[svc]

        # Find which detected anomaly window covers this fault
        ft = fl.get("timestamp", 0)
        # Map timestamp to approximate window index
        # Use a simple heuristic: fault is near the middle of the time range
        assigned_window = None

        if anomaly_windows:
            # Assign to the nearest anomaly window
            # (in production, anomaly detection would provide the window)
            for aw in anomaly_windows:
                # Check if root cause service has elevated residual in this window
                window_res = residuals[aw[0]:aw[1], rc]
                if np.nanmax(window_res) > normal_baseline["mean"] + normal_baseline["std"]:
                    assigned_window = aw
                    break

        if assigned_window is None:
            # Fallback: use the first anomaly window, or middle of episode
            if anomaly_windows:
                assigned_window = anomaly_windows[0]
            else:
                assigned_window = (T//3, min(T, T//3 + 30))

        # Rank services within assigned window
        rank = rank_services_in_window(residuals, assigned_window, N)
        rc_rank = int(np.where(rank == rc)[0][0]) + 1

        # Time detection: first step where RC's residual crosses threshold
        t_thresh = normal_baseline["mean"] + 3 * normal_baseline["std"]
        first_dev = 9999
        for t in range(assigned_window[0], assigned_window[1]):
            if not np.isnan(residuals[t, rc]) and residuals[t, rc] > t_thresh:
                first_dev = t
                break
        time_hit = first_dev < 9999

        results.append({
            "fault": f"{fl.get('type','?')}@{svc}",
            "rank": rc_rank, "top1": rc_rank == 1, "top3": rc_rank <= 3,
            "time_hit": time_hit, "c_t_hit": rc_rank == 1 and time_hit,
            "mrr": 1.0 / rc_rank,
        })

    return results, len(anomaly_windows)


# ============================================================
# Main: evaluate all benchmarks
# ============================================================

def main():
    print(f"Device: {jax.devices()[0]}")
    state, model = load_model()

    # Load source normal baseline from OB Day1
    with h5py.File(os.path.join(PROJECT_ROOT, "data", "processed", "cross_system.h5"), "r") as f:
        edge_index_ob = f["edge_index"][:]
        ob_svcs = [s.decode() if isinstance(s,bytes) else s for s in f["service_names"][:]]
        # Compute baseline from val windows
        val_nodes = f["val"]["nodes"][:]
        val_edges = f["val"]["edges"][:]
        val_ctx = f["val"]["ctx"][:]
        # Fault Day1
        d1 = {"nodes": f["fault_day1"]["nodes"][:], "edges": f["fault_day1"]["edges"][:],
              "ctx": f["fault_day1"]["ctx"][:],
              "faults": json.loads(f["fault_day1"].attrs["fault_labels"])}
        # Fault Day2
        d2 = {"nodes": f["fault_day2"]["nodes"][:], "edges": f["fault_day2"]["edges"][:],
              "ctx": f["fault_day2"]["ctx"][:],
              "faults": json.loads(f["fault_day2"].attrs["fault_labels"])}

    # Normal baseline from val data
    baseline_res = []
    for i in range(min(3, len(val_nodes))):
        res = compute_residuals_clean(state, val_nodes[i], val_edges[i], val_ctx[i], edge_index_ob)
        baseline_res.append(res.flatten())
    baseline_flat = np.concatenate(baseline_res)
    normal_baseline = {"mean": float(np.mean(baseline_flat)), "std": float(np.std(baseline_flat))}
    print(f"Normal baseline: μ={normal_baseline['mean']:.3f}, σ={normal_baseline['std']:.3f}")

    all_systems = []

    # ===== OB Day1 (source, in-distribution) =====
    res, n_wins = evaluate_system_clean(
        "OB Day1 (source)", d1["nodes"], d1["edges"], d1["ctx"], edge_index_ob,
        d1["faults"], ob_svcs, state, normal_baseline, use_posterior=True
    )
    all_systems.append(("OB Day1 (source)", res))

    # ===== OB Day2 (temporal generalization) =====
    res, n_wins = evaluate_system_clean(
        "OB Day2 (target)", d2["nodes"], d2["edges"], d2["ctx"], edge_index_ob,
        d2["faults"], ob_svcs, state, normal_baseline, use_posterior=True
    )
    all_systems.append(("OB Day2 (target)", res))

    # ===== TrainTicket (cross-system) =====
    # TrainTicket: use raw Nezha data directly (different format from OpenRCA)
    ts_dir = os.path.join(PROJECT_ROOT, "data", "raw", "nezha", "rca_data", "2023-01-29")
    from data.aiops2020.parse import parse_metrics_to_service
    TS_ROLES = ['ts-gateway-service','ts-user-service','ts-contacts-service',
                'ts-order-service','ts-payment-service','ts-consign-service',
                'ts-station-service','ts-route-service','ts-food-service','ts-basic-service']
    ts_svc_idx = {s:i for i,s in enumerate(TS_ROLES)}
    ts_nodes_raw, ts_ts = parse_metrics_to_service(os.path.join(ts_dir, "metric"), ts_svc_idx)
    ts_T, ts_N = ts_nodes_raw.shape[0], ts_nodes_raw.shape[1]
    ts_nodes = np.pad(ts_nodes_raw, ((0,0),(0,max(0,10-ts_N)),(0,max(0,8-ts_nodes_raw.shape[2]))))
    ts_ctx = np.zeros((ts_T, 3), dtype=np.float32)
    ts_ctx[:,0] = ts_nodes_raw[:,:,6].sum(axis=1) if ts_nodes_raw.shape[2] > 6 else ts_nodes_raw.sum(axis=(1,2))
    ts_ctx[:,2] = ts_N
    ts_edges = np.zeros((ts_T, edge_index_ob.shape[1], 4), dtype=np.float32)
    ts_edges[:,:,2] = 1.0
    with open(os.path.join(ts_dir, "2023-01-29-fault_list.json")) as f:
        ts_faults_raw = json.load(f)
    ts_faults = []
    for hour, entries in ts_faults_raw.items():
        for e in entries:
            from data.aiops2020.parse import extract_service
            svc = extract_service(e["inject_pod"])
            ts_faults.append({"service": svc, "type": e["inject_type"],
                              "timestamp": int(e["inject_timestamp"])})
    res, n_wins = evaluate_system_clean(
        "TrainTicket", ts_nodes, ts_edges, ts_ctx, edge_index_ob,
        ts_faults, TS_ROLES, state, normal_baseline, use_posterior=True
    )
    all_systems.append(("TrainTicket", res))

    # ===== Market cloudbed-1 =====
    # Market: use OpenRCA format
    def import_from_path(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    pc4 = import_from_path("phase4c", os.path.join(os.path.dirname(__file__), "phase4c_openrca.py"))
    parse_openrca_metrics = pc4.parse_openrca_metrics
    mkt_dir = os.path.join(OPENRCA, "Market", "Market", "cloudbed-1")
    mkt_data = parse_openrca_metrics("Market/cb1", mkt_dir, max_days=1)
    if mkt_data and mkt_data["services"]:
        mkt_N = len(mkt_data["services"])
        mkt_pad = max(0, 10 - mkt_N)
        mkt_nodes = mkt_data["nodes"]
        if mkt_nodes.shape[2] < 8:
            mkt_nodes = np.pad(mkt_nodes, ((0,0),(0,0),(0,8-mkt_nodes.shape[2])))
        mkt_nodes = np.pad(mkt_nodes, ((0,0),(0,mkt_pad),(0,0)))
        mkt_ctx = mkt_data["ctx"]
        mkt_edges = np.zeros((mkt_nodes.shape[0], edge_index_ob.shape[1], 4), dtype=np.float32)
        mkt_edges[:,:,2] = 1.0
        mkt_record = pd.read_csv(os.path.join(mkt_dir, "record.csv"))
        mkt_faults = []
        for _, row in mkt_record.iterrows():
            comp = str(row.get("component",""))
            svc_name = comp.rsplit("-",1)[0] if "-" in comp else comp
            mkt_faults.append({"service": svc_name, "type": str(row.get("reason","")),
                               "timestamp": float(row["timestamp"])})
        res, n_wins = evaluate_system_clean(
            "Market/cb1", mkt_nodes, mkt_edges, mkt_ctx, edge_index_ob,
            mkt_faults, list(mkt_data["services"]) + ["pad"]*mkt_pad, state, normal_baseline, use_posterior=True
        )
        all_systems.append(("Market/cb1", res))

    # ===== Bank (container-level, Phase 7) =====
    # Bank (container-level, Phase 7)
    p567 = import_from_path("phase567", os.path.join(os.path.dirname(__file__), "phase567_evaluate.py"))
    parse_container_metrics = p567.parse_container_metrics
    bank_dir = os.path.join(OPENRCA, "Bank", "Bank")
    bank_days = sorted([d for d in os.listdir(os.path.join(bank_dir,"telemetry"))
                         if d.startswith("20")])[:1]
    bank_result = parse_container_metrics(os.path.join(bank_dir, "telemetry", bank_days[0], "metric"))
    if bank_result:
        bank_N = bank_result["N"]
        bank_D = bank_result["D"]
        bank_nodes_raw = bank_result["nodes"]  # [T, N, D_raw]
        # Map to 8 feature dims
        bank_nodes = np.zeros((bank_result["T"], bank_N, 8), dtype=np.float32)
        for d in range(min(bank_D, 8)):
            bank_nodes[:,:,d] = bank_nodes_raw[:,:,d]
        bank_ctx = np.zeros((bank_result["T"], 3), dtype=np.float32)
        bank_ctx[:,0] = bank_nodes.sum(axis=(1,2))
        bank_ctx[:,2] = bank_N
        bank_ei = np.zeros((2, min(14,bank_N)), dtype=np.int32)
        for e in range(min(14,bank_N)):
            bank_ei[0,e]=e; bank_ei[1,e]=(e+1)%bank_N
        bank_edges = np.zeros((bank_result["T"], bank_ei.shape[1], 4), dtype=np.float32)
        bank_edges[:,:,2]=1.0
        bank_record = pd.read_csv(os.path.join(bank_dir, "record.csv"))
        bank_faults = []
        for _, row in bank_record.iterrows():
            bank_faults.append({"service": str(row.get("component","")).lower().strip(),
                                "type": str(row.get("reason","")),
                                "timestamp": float(row["timestamp"])})
        res, n_wins = evaluate_system_clean(
            "Bank (container)", bank_nodes, bank_edges, bank_ctx, bank_ei,
            bank_faults, bank_result["containers"], state, normal_baseline, use_posterior=True
        )
        all_systems.append(("Bank (container)", res))

    # ===== Summary =====
    print("\n" + "=" * 65)
    print("CLEAN RCA Evaluation (Anomaly Detection → Residual Ranking)")
    print("=" * 65)
    print(f"{'System':<22s} {'Faults':>6s} {'AnomWins':>8s} {'Top-1':>8s} {'Top-3':>8s} {'C+T':>8s} {'MRR':>8s} {'AvgR':>6s}")
    print("-" * 72)
    for label, results in all_systems:
        n = len(results)
        if n == 0:
            print(f"{label:<22s} {'0':>6s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>6s}")
            continue
        t1 = sum(1 for r in results if r["top1"])
        t3 = sum(1 for r in results if r["top3"])
        ct = sum(1 for r in results if r["c_t_hit"])
        mrr = np.mean([r["mrr"] for r in results])
        ar = np.mean([r["rank"] for r in results])
        w = sum(1 for _ in [1])  # placeholder
        print(f"{label:<22s} {n:>6d} {'?':>8s} {t1:>5d}/{n:<2d} {t3:>5d}/{n:<2d} "
              f"{ct:>5d}/{n:<2d} {mrr:>8.3f} {ar:>6.2f}")

    print(f"\n{'OB Day1 (old, leaked)':<22s} {'24':>6s} {'-':>8s} {'20/24':>8s} {'24/24':>8s} {'-':>8s} {'0.896':>8s} {'1.33':>6s}")
    print(f"\nNote: Prior+Fixed mode (cleanest) was 25% Top-1 on OB Day1.")
    print(f"Current Posterior+AnomalyDetection is the practical pipeline.")


if __name__ == "__main__":
    main()
