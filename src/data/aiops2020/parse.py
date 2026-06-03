"""Parse Nezha dataset into unified format for RCAWorld training.

Handles Online Boutique (hipster) and TrainTicket (ts).
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ParsedDataset:
    system_name: str
    node_features: np.ndarray       # [T, N, D_node]
    edge_features: np.ndarray       # [T, E, D_edge]
    edge_index: np.ndarray          # [2, E]
    workload_context: np.ndarray    # [T, D_ctx]
    timestamps: np.ndarray          # [T]
    service_names: List[str]
    node_feature_names: List[str]
    edge_feature_names: List[str]
    fault_labels: List[Dict]        # per-fault annotation
    normal_mask: np.ndarray         # [T] bool: True = no fault in window


# Selected features for the world model
NODE_FEATURE_NAMES = [
    "cpu_rate",          # CpuUsageRate(%)
    "memory_rate",       # MemoryUsageRate(%)
    "net_rx_bytes",      # NetworkReceiveBytes (log-scale)
    "net_tx_bytes",      # NetworkTransmitBytes (log-scale)
    "latency_p90_ms",    # PodClientLatencyP90(s) * 1000
    "latency_p99_ms",    # PodClientLatencyP99(s) * 1000
    "workload_ops",      # PodWorkload(Ops)
    "success_rate",      # PodSuccessRate(%)
]
N_FEATURES = len(NODE_FEATURE_NAMES)

EDGE_FEATURE_NAMES = [
    "call_count",
    "error_rate",
    "avg_latency_ms",
    "p99_latency_ms",
]

CONTEXT_NAMES = [
    "total_qps",
    "avg_success_rate",
    "active_services",
]


def extract_service(pod_name: str) -> str:
    """'frontend-579b9bff58-t2dbm' -> 'frontend'"""
    # Known service prefixes
    prefixes = [
        "frontend", "cartservice", "checkoutservice",
        "currencyservice", "emailservice", "paymentservice",
        "productcatalogservice", "recommendationservice",
        "shippingservice", "adservice",
    ]
    for p in prefixes:
        if pod_name.lower().startswith(p):
            return p
    return pod_name.rsplit('-', 2)[0].lower()


def parse_dependency_graph(dep_file: str) -> tuple:
    """Parse dependency.csv → services list, edge_index, edge_names."""
    df = pd.read_csv(dep_file)
    svc_set = {}
    edges = []

    for _, row in df.iterrows():
        src = str(row.iloc[0]).lower().strip()
        tgt = str(row.iloc[1]).lower().strip()
        # Filter IP edges
        if tgt[0].isdigit() or tgt.count('.') >= 2:
            continue
        if src not in svc_set:
            svc_set[src] = len(svc_set)
        if tgt not in svc_set:
            svc_set[tgt] = len(svc_set)
        edges.append((svc_set[src], svc_set[tgt]))

    services = [k for k, _ in sorted(svc_set.items(), key=lambda x: x[1])]
    edge_index = np.array([[s for s, _ in edges], [t for _, t in edges]], dtype=np.int32)
    edge_names = [(services[s], services[t]) for s, t in zip(edge_index[0], edge_index[1])]
    return services, edge_index, edge_names


def parse_metrics_to_service(
    metric_dir: str,
    service_to_idx: Dict[str, int],
    window_sec: int = 60,
) -> tuple:
    """Parse per-pod metric CSVs → service-level time series.

    Returns:
        node_features: [T, N, N_FEATURES]
        timestamps: [T]
    """
    files = sorted(glob.glob(os.path.join(metric_dir, "*_metric.csv")))
    if not files:
        raise FileNotFoundError(f"No metric files in {metric_dir}")

    N = len(service_to_idx)
    all_timestamps = set()

    # First pass: collect all timestamps
    sample_data = []
    for fp in files:
        chunks = pd.read_csv(fp, chunksize=5000)
        for chunk in chunks:
            for _, row in chunk.iterrows():
                try:
                    ts = int(row.iloc[1])  # TimeStamp column
                    pod = str(row.iloc[2]).lower()  # PodName
                    svc = extract_service(pod)
                    if svc not in service_to_idx:
                        continue
                    all_timestamps.add(ts)
                    sample_data.append((ts, svc, fp))
                except (ValueError, IndexError):
                    continue
            break  # Only first chunk for timestamp collection

    if not all_timestamps:
        raise ValueError("No valid timestamps found")

    timestamps = sorted(all_timestamps)
    t_to_idx = {t: i for i, t in enumerate(timestamps)}
    T = len(timestamps)
    print(f"  {T} timestamps, range: {timestamps[0]} - {timestamps[-1]}")

    # Second pass: accumulate metrics
    node_sum = np.zeros((T, N, N_FEATURES), dtype=np.float64)
    node_count = np.zeros((T, N), dtype=np.float64)

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        for _, row in df.iterrows():
            try:
                ts = int(row.iloc[1])
                pod = str(row.iloc[2]).lower()
            except (ValueError, IndexError):
                continue

            svc = extract_service(pod)
            if svc not in service_to_idx:
                continue
            if ts not in t_to_idx:
                continue

            t = t_to_idx[ts]
            n = service_to_idx[svc]
            node_count[t, n] += 1

            # Extract features (columns 3-20 are metrics)
            try:
                cpu_rate = float(row.iloc[4])   # CpuUsageRate(%)
                mem_rate = float(row.iloc[6])   # MemoryUsageRate(%)
                net_rx = float(row.iloc[9])     # NetworkReceiveBytes
                net_tx = float(row.iloc[10])    # NetworkTransmitBytes
                p90_lat = float(row.iloc[11])   # PodClientLatencyP90(s)
                p99_lat = float(row.iloc[15])   # PodClientLatencyP99(s)
                ops = float(row.iloc[17])       # PodWorkload(Ops)
                succ = float(row.iloc[18])      # PodSuccessRate(%)
            except (ValueError, IndexError, KeyError):
                continue

            node_sum[t, n, 0] += cpu_rate
            node_sum[t, n, 1] += mem_rate
            node_sum[t, n, 2] += np.log1p(max(net_rx, 0))  # log scale
            node_sum[t, n, 3] += np.log1p(max(net_tx, 0))
            node_sum[t, n, 4] += p90_lat * 1000  # to ms
            node_sum[t, n, 5] += p99_lat * 1000
            node_sum[t, n, 6] += ops
            node_sum[t, n, 7] += succ

    # Average
    node_features = np.zeros((T, N, N_FEATURES), dtype=np.float32)
    for n in range(N):
        mask = node_count[:, n] > 0
        node_features[mask, n, :] = (node_sum[mask, n, :] /
                                      node_count[mask, n, np.newaxis])

    # Fill missing with forward fill
    for n in range(N):
        for f in range(N_FEATURES):
            last_val = 0.0
            for t in range(T):
                if node_count[t, n] > 0:
                    last_val = node_features[t, n, f]
                else:
                    node_features[t, n, f] = last_val

    np.nan_to_num(node_features, copy=False)
    return node_features, np.array(timestamps)


def build_edge_features_from_traces(
    trace_dir: str,
    edge_names: List[tuple],
    service_to_idx: Dict[str, int],
    timestamps: np.ndarray,
    window_sec: int = 60,
    max_files: int = 20,
) -> np.ndarray:
    """Parse trace files → edge features [T, E, 4].

    Limits to max_files trace files for speed (traces are sampled).
    """
    T = len(timestamps)
    E = len(edge_names)
    t_min, t_max = timestamps[0], timestamps[-1]

    # Edge name lookup
    edge_name_to_idx = {e: i for i, e in enumerate(edge_names)}
    idx_to_svc = {i: s for s, i in service_to_idx.items()}

    # Accumulators
    call_counts = np.zeros((T, E), dtype=np.float32)
    error_counts = np.zeros((T, E), dtype=np.float32)
    lat_sum = np.zeros((T, E), dtype=np.float64)
    lat_max = np.zeros((T, E), dtype=np.float32)

    trace_files = sorted(glob.glob(os.path.join(trace_dir, "*_trace.csv")))[:max_files]
    if not trace_files:
        return np.zeros((T, E, 4), dtype=np.float32)

    for fpath in trace_files:
        try:
            df = pd.read_csv(fpath, nrows=100000)
        except Exception:
            continue
        if df.empty:
            continue

        for _, row in df.iterrows():
            try:
                ts = int(row["StartTimeUnixNano"]) // 1_000_000_000
                dur = int(row.get("Duration", 0))
            except (ValueError, KeyError):
                continue

            if ts < t_min or ts > t_max:
                continue
            w = np.searchsorted(timestamps, ts, side='right') - 1
            if w < 0 or w >= T:
                continue

            pod = str(row.get("PodName", "")).lower()
            src = extract_service(pod)
            op_val = row.get("OperationName", "")
            if pd.isna(op_val) or not isinstance(op_val, str):
                continue
            op = op_val.lower()

            # Determine destination from operation name
            dst = None
            for dst_name in idx_to_svc.values():
                if dst_name != src and dst_name in op:
                    dst = dst_name
                    break

            # If no dst from operation, try parent-based inference
            if dst is None:
                parent_id = str(row.get("ParentID", ""))
                if parent_id == "root" and src != "frontend":
                    dst = "frontend"
                elif parent_id != "root" and parent_id != "":
                    # Could look up parent span's service, but skip for speed
                    pass

            if dst is None:
                continue

            edge_key = (src, dst)
            if edge_key not in edge_name_to_idx:
                continue
            e = edge_name_to_idx[edge_key]

            call_counts[w, e] += 1
            lat_ms = dur / 1_000_000
            lat_sum[w, e] += lat_ms
            lat_max[w, e] = max(lat_max[w, e], lat_ms)

            if "error" in op or "fail" in op or "exception" in op:
                error_counts[w, e] += 1

    # Build output
    edge_features = np.zeros((T, E, 4), dtype=np.float32)
    edge_features[:, :, 0] = call_counts
    edge_features[:, :, 2] = np.divide(lat_sum, np.maximum(call_counts, 1),
                                        out=np.zeros_like(lat_sum),
                                        where=call_counts > 0)
    edge_features[:, :, 3] = lat_max
    total_calls = np.maximum(call_counts, 1)
    edge_features[:, :, 1] = error_counts / total_calls
    np.nan_to_num(edge_features, copy=False)
    return edge_features


def parse_faults(fault_file: str, window_sec: int = 60) -> List[Dict]:
    """Parse fault_list.json."""
    with open(fault_file) as f:
        data = json.load(f)
    faults = []
    for hour_key, entries in data.items():
        for entry in entries:
            faults.append({
                "inject_time": entry["inject_time"],
                "timestamp": int(entry["inject_timestamp"]),
                "pod": entry["inject_pod"],
                "type": entry["inject_type"],
                "service": extract_service(entry["inject_pod"]),
            })
    return faults


def build_normal_mask(
    T: int,
    timestamps: np.ndarray,
    faults: List[Dict],
    fault_duration: int = 600,
) -> np.ndarray:
    """Build boolean mask: True for windows without active fault."""
    mask = np.ones(T, dtype=bool)
    for ft in faults:
        t0 = ft["timestamp"]
        t1 = t0 + fault_duration
        w0 = np.searchsorted(timestamps, t0)
        w1 = np.searchsorted(timestamps, t1)
        if w0 < T:
            mask[w0:min(w1, T)] = False
    return mask


def build_workload_context(node_features: np.ndarray) -> np.ndarray:
    """Build workload context [T, D_ctx] from node features.

    ctx[0] = total_qps (sum of workload_ops across services)
    ctx[1] = avg_success_rate (mean across services)
    ctx[2] = active_services (count of services with ops > 0)
    """
    T = node_features.shape[0]
    ctx = np.zeros((T, 3), dtype=np.float32)
    ctx[:, 0] = node_features[:, :, 6].sum(axis=1)  # total ops
    ctx[:, 1] = node_features[:, :, 7].mean(axis=1)  # avg success rate
    ctx[:, 2] = (node_features[:, :, 6] > 0.1).sum(axis=1)  # active services
    np.nan_to_num(ctx, copy=False)
    return ctx


def parse_full_dataset(
    metric_dir: str,
    trace_dir: str,
    dep_file: str,
    fault_file: Optional[str] = None,
    window_sec: int = 60,
    system_name: str = "hipster",
) -> ParsedDataset:
    """Main parsing entry point."""

    # Dependency graph
    services, edge_index, edge_names = parse_dependency_graph(dep_file)
    svc_to_idx = {s: i for i, s in enumerate(services)}
    N, E = len(services), edge_index.shape[1]
    print(f"System: {system_name}, Services: {N}, Edges: {E}")

    # Metrics
    node_features, timestamps = parse_metrics_to_service(
        metric_dir, svc_to_idx, window_sec
    )
    T = len(timestamps)
    print(f"Windows: {T}")

    # Edge features from traces
    edge_features = build_edge_features_from_traces(
        trace_dir, edge_names, svc_to_idx, timestamps, window_sec, max_files=50
    )

    # Workload context
    workload_context = build_workload_context(node_features)

    # Fault labels
    fault_labels = parse_faults(fault_file, window_sec) if fault_file else []
    normal_mask = build_normal_mask(T, timestamps, fault_labels) if fault_labels else np.ones(T, dtype=bool)

    if fault_labels:
        print(f"Faults: {len(fault_labels)}, Normal windows: {normal_mask.sum()}/{T}")

    return ParsedDataset(
        system_name=system_name,
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        workload_context=workload_context,
        timestamps=timestamps,
        service_names=services,
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        fault_labels=fault_labels,
        normal_mask=normal_mask,
    )


if __name__ == "__main__":
    nezha = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "data", "raw", "nezha"
    )

    # Test: Online Boutique normal (2022-08-22)
    base = os.path.join(nezha, "construct_data", "2022-08-22")
    print("=== Normal data (2022-08-22) ===")
    ds = parse_full_dataset(
        os.path.join(base, "metric"),
        os.path.join(base, "trace"),
        os.path.join(base, "metric", "dependency.csv"),
        system_name="hipster",
    )
    print(f"  nodes: {ds.node_features.shape}, edges: {ds.edge_features.shape}")
    print(f"  Services: {ds.service_names}")

    # Test: Online Boutique fault (2022-08-22)
    base = os.path.join(nezha, "rca_data", "2022-08-22")
    print("\n=== Fault data (2022-08-22) ===")
    ds = parse_full_dataset(
        os.path.join(base, "metric"),
        os.path.join(base, "trace"),
        os.path.join(base, "metric", "dependency.csv"),
        os.path.join(base, "2022-08-22-fault_list.json"),
        system_name="hipster",
    )
    print(f"  nodes: {ds.node_features.shape}")
    print(f"  Normal windows: {ds.normal_mask.sum()}/{len(ds.normal_mask)}")
    for fl in ds.fault_labels[:3]:
        print(f"  Fault: {fl['inject_time']} {fl['service']} {fl['type']}")
