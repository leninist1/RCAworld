"""Build the full training/evaluation dataset from Nezha data.

Combines all data sources, normalizes, splits by episode, saves as HDF5.
"""
import os
import sys
import json
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# Insert project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data.aiops2020.parse import (
    parse_full_dataset, ParsedDataset, NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES, N_FEATURES,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "nezha")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Dataset configuration: (system, date, has_faults)
DATASETS = [
    # Online Boutique (hipster) - normal data
    ("hipster", "2022-08-22", False, "construct_data"),
    ("hipster", "2022-08-23", False, "construct_data"),
    # Online Boutique (hipster) - fault data
    ("hipster", "2022-08-22", True, "rca_data"),
    ("hipster", "2022-08-23", True, "rca_data"),
    # TrainTicket (ts) - normal data
    ("ts", "2023-01-29", False, "construct_data"),
    ("ts", "2023-01-30", False, "construct_data"),
    # TrainTicket (ts) - fault data
    ("ts", "2023-01-29", True, "rca_data"),
    ("ts", "2023-01-30", True, "rca_data"),
]


def load_raw_dataset(system: str, date: str, has_faults: bool, data_type: str) -> ParsedDataset:
    """Load and parse one raw dataset."""
    base = os.path.join(RAW_DIR, data_type, date)
    metric_dir = os.path.join(base, "metric")
    trace_dir = os.path.join(base, "trace")
    dep_file = os.path.join(base, "metric", "dependency.csv")

    # Fault file for rca_data
    fault_file = None
    if has_faults:
        fault_file = os.path.join(base, f"{date}-fault_list.json")
        if not os.path.exists(fault_file):
            print(f"  Warning: fault file not found: {fault_file}")
            fault_file = None

    return parse_full_dataset(
        metric_dir=metric_dir,
        trace_dir=trace_dir,
        dep_file=dep_file,
        fault_file=fault_file,
        system_name=system,
    )


def normalize_dataset(
    node_features: np.ndarray,
    edge_features: np.ndarray,
    workload_context: np.ndarray,
    normal_mask: np.ndarray,
) -> dict:
    """Compute normalization stats on normal windows only.

    Returns dict with mean, std for each modality.
    """
    normal_idx = np.where(normal_mask)[0]

    if len(normal_idx) == 0:
        return {
            "node_mean": node_features.mean(axis=0, keepdims=True),
            "node_std": node_features.std(axis=0, keepdims=True) + 1e-6,
            "edge_mean": edge_features.mean(axis=0, keepdims=True),
            "edge_std": edge_features.std(axis=0, keepdims=True) + 1e-6,
            "ctx_mean": workload_context.mean(axis=0, keepdims=True),
            "ctx_std": workload_context.std(axis=0, keepdims=True) + 1e-6,
        }

    normal_nodes = node_features[normal_idx]
    normal_edges = edge_features[normal_idx]
    normal_ctx = workload_context[normal_idx]

    return {
        "node_mean": normal_nodes.mean(axis=0, keepdims=True),
        "node_std": normal_nodes.std(axis=0, keepdims=True) + 1e-6,
        "edge_mean": normal_edges.mean(axis=0, keepdims=True),
        "edge_std": normal_edges.std(axis=0, keepdims=True) + 1e-6,
        "ctx_mean": normal_ctx.mean(axis=0, keepdims=True),
        "ctx_std": normal_ctx.std(axis=0, keepdims=True) + 1e-6,
    }


def apply_normalization(dataset: ParsedDataset, stats: dict) -> dict:
    """Normalize dataset using precomputed stats."""
    nodes = (dataset.node_features - stats["node_mean"]) / stats["node_std"]
    edges = (dataset.edge_features - stats["edge_mean"]) / stats["edge_std"]
    ctx = (dataset.workload_context - stats["ctx_mean"]) / stats["ctx_std"]

    np.nan_to_num(nodes, copy=False)
    np.nan_to_num(edges, copy=False)
    np.nan_to_num(ctx, copy=False)

    return {
        "nodes": nodes.astype(np.float32),
        "edges": edges.astype(np.float32),
        "ctx": ctx.astype(np.float32),
    }


def create_sliding_windows(
    nodes: np.ndarray,
    edges: np.ndarray,
    ctx: np.ndarray,
    normal_mask: np.ndarray,
    fault_labels: list,
    window_size: int = 24,
    stride: int = 6,
) -> dict:
    """Create sliding windows for training.

    Returns dict with training (normal-only) and fault windows.
    """
    T = nodes.shape[0]
    train_windows = []
    val_windows = []
    fault_windows = []

    for start in range(0, T - window_size, stride):
        end = start + window_size
        if end > T:
            break

        win_normal = normal_mask[start:end].all()
        win = {
            "nodes": nodes[start:end],
            "edges": edges[start:end],
            "ctx": ctx[start:end],
        }

        if win_normal:
            # Train/val split: 80/20 based on start position parity
            if (start // window_size) % 5 == 0:
                val_windows.append(win)
            else:
                train_windows.append(win)
        else:
            # Fault window - record which faults are active
            active_faults = []
            for fl in fault_labels:
                ft = fl.get("timestamp", 0)
                fdur = 600  # 10 min
                win_start_ts = 0  # We don't have exact timestamps per window here
                if ft and start <= ft // 60:  # approximate
                    active_faults.append(fl)

            fault_windows.append({
                **win,
                "active_faults": active_faults,
            })

    return {
        "train": train_windows,
        "val": val_windows,
        "fault": fault_windows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=24, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=6, help="Window stride")
    parser.add_argument("--output", type=str, default=None, help="Output HDF5 path")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Collect all normal data for normalization
    print("=" * 60)
    print("Step 1: Loading all datasets")
    print("=" * 60)

    all_hipster = []
    all_ts = []

    for system, date, has_faults, dtype in DATASETS:
        print(f"\nLoading {system}/{date} ({'fault' if has_faults else 'normal'})...")
        try:
            ds = load_raw_dataset(system, date, has_faults, dtype)
            if system == "hipster":
                all_hipster.append(ds)
            else:
                all_ts.append(ds)
        except Exception as e:
            print(f"  Skipped: {e}")
            continue

    if not all_hipster:
        print("No hipster data loaded. Exiting.")
        return

    # Focus on Online Boutique (hipster) for Phase 1
    # Combine normal (construct_data) datasets for training
    normal_datasets = [d for d in all_hipster if not d.fault_labels]

    # Combine into one big normal dataset
    # Assume same services and edges
    print("\n" + "=" * 60)
    print("Step 2: Building normal training set")
    print("=" * 60)

    ref = normal_datasets[0]
    N = len(ref.service_names)
    E = ref.edge_index.shape[1]

    # Stack all normal data
    all_nodes = []
    all_edges = []
    all_ctx = []
    all_normal = []

    for ds in normal_datasets:
        all_nodes.append(ds.node_features)
        all_edges.append(ds.edge_features)
        all_ctx.append(ds.workload_context)
        all_normal.append(ds.normal_mask)

    # Concatenate along time axis
    train_nodes_full = np.concatenate(all_nodes, axis=0)
    train_edges_full = np.concatenate(all_edges, axis=0)
    train_ctx_full = np.concatenate(all_ctx, axis=0)
    train_normal_full = np.concatenate(all_normal, axis=0)

    print(f"Training data: {train_nodes_full.shape[0]} windows")
    print(f"Normal windows: {train_normal_full.sum()}")

    # Step 3: Normalize
    print("\n" + "=" * 60)
    print("Step 3: Normalization")
    print("=" * 60)

    stats = normalize_dataset(
        train_nodes_full, train_edges_full, train_ctx_full, train_normal_full
    )
    for k, v in stats.items():
        print(f"  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")

    # Apply normalization
    train_data = {
        "nodes": (train_nodes_full - stats["node_mean"]) / stats["node_std"],
        "edges": (train_edges_full - stats["edge_mean"]) / stats["edge_std"],
        "ctx": (train_ctx_full - stats["ctx_mean"]) / stats["ctx_std"],
    }
    np.nan_to_num(train_data["nodes"], copy=False)
    np.nan_to_num(train_data["edges"], copy=False)
    np.nan_to_num(train_data["ctx"], copy=False)

    # Step 4: Create sliding windows
    print("\n" + "=" * 60)
    print("Step 4: Sliding windows")
    print("=" * 60)

    windows = create_sliding_windows(
        train_data["nodes"].astype(np.float32),
        train_data["edges"].astype(np.float32),
        train_data["ctx"].astype(np.float32),
        train_normal_full,
        [],  # no fault labels in normal data
        window_size=args.window_size,
        stride=args.stride,
    )

    print(f"Train windows: {len(windows['train'])}")
    print(f"Val windows: {len(windows['val'])}")

    # Step 5: Save
    print("\n" + "=" * 60)
    print("Step 5: Saving")
    print("=" * 60)

    output_path = args.output or os.path.join(PROCESSED_DIR, "hipster_dataset.h5")
    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs["system"] = "hipster"
        f.attrs["N_services"] = N
        f.attrs["N_edges"] = E
        f.attrs["N_node_features"] = N_FEATURES
        f.attrs["N_edge_features"] = 4
        f.attrs["N_context_features"] = 3
        f.attrs["window_size"] = args.window_size
        f.attrs["stride"] = args.stride

        # Store service info
        dt = h5py.string_dtype()
        f.create_dataset("service_names", data=np.array(ref.service_names, dtype=object))
        f.create_dataset("node_feature_names", data=np.array(NODE_FEATURE_NAMES, dtype=object))
        f.create_dataset("edge_feature_names", data=np.array(EDGE_FEATURE_NAMES, dtype=object))
        f.create_dataset("edge_index", data=ref.edge_index)

        # Store normalization stats
        for k, v in stats.items():
            f.create_dataset(f"stats/{k}", data=v)

        # Store train windows
        if windows["train"]:
            train_group = f.create_group("train")
            n_train = len(windows["train"])
            train_group.create_dataset("nodes", (n_train, args.window_size, N, N_FEATURES), dtype=np.float32)
            train_group.create_dataset("edges", (n_train, args.window_size, E, 4), dtype=np.float32)
            train_group.create_dataset("ctx", (n_train, args.window_size, 3), dtype=np.float32)
            for i, w in enumerate(tqdm(windows["train"], desc="Saving train")):
                train_group["nodes"][i] = w["nodes"]
                train_group["edges"][i] = w["edges"]
                train_group["ctx"][i] = w["ctx"]

        # Store val windows
        if windows["val"]:
            val_group = f.create_group("val")
            n_val = len(windows["val"])
            val_group.create_dataset("nodes", (n_val, args.window_size, N, N_FEATURES), dtype=np.float32)
            val_group.create_dataset("edges", (n_val, args.window_size, E, 4), dtype=np.float32)
            val_group.create_dataset("ctx", (n_val, args.window_size, 3), dtype=np.float32)
            for i, w in enumerate(tqdm(windows["val"], desc="Saving val")):
                val_group["nodes"][i] = w["nodes"]
                val_group["edges"][i] = w["edges"]
                val_group["ctx"][i] = w["ctx"]

        # Store fault data separately (for evaluation)
        fault_ds_list = [d for d in all_hipster if d.fault_labels]
        if fault_ds_list:
            fault_group = f.create_group("fault")
            for fd_idx, fd in enumerate(fault_ds_list):
                fg = fault_group.create_group(f"episode_{fd_idx}")
                norm = apply_normalization(fd, stats)
                fg.create_dataset("nodes", data=norm["nodes"])
                fg.create_dataset("edges", data=norm["edges"])
                fg.create_dataset("ctx", data=norm["ctx"])
                fg.create_dataset("normal_mask", data=fd.normal_mask)

                # Save fault labels as JSON
                fault_json = json.dumps(fd.fault_labels)
                fg.attrs["fault_labels"] = fault_json

    print(f"\nSaved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
