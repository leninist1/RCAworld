"""Episode builder for query-level OpenRCA evaluation.

For each InferenceQuery, builds an independent Episode tensor with:
  - Only telemetry within the query's observation window (+ burn-in).
  - Real observation mask from actual KPI availability.
  - Fixed-time-grid resampling (optional, default 60s).
  - Entity type assignments (heuristic or learned).

CRITICAL: InferenceQuery contains ZERO ground truth. GT labels are NEVER
exposed to episode building, telemetry loading, or model inference.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np

from ..schema.entity import Entity
from ..schema.event import ObservationEvent
from .query_parser import InferenceQuery


def assign_entity_type(entity_id: str) -> str:
    """Heuristic entity type assignment (same as original clean eval)."""
    eid = entity_id.lower()
    if any(k in eid for k in ['mysql', 'redis', 'db', 'postgres', 'mongo']):
        return 'database'
    if any(k in eid for k in ['tomcat', 'apache', 'nginx', 'java', 'jvm']):
        return 'middleware'
    if any(k in eid for k in ['docker', 'container', 'pod']):
        return 'container'
    if any(k in eid for k in ['host', 'node', 'server']):
        return 'host'
    return 'container'


# KPI -> OB feature index mapping (8-feature Online Boutique space)
KPI_TO_OB: Dict[str, int] = {
    "cpu": 0, "jvm_cpu": 0,
    "mem": 1, "mem_usage": 1, "jvm_mem": 1,
    "net_rx": 2,
    "net_tx": 3,
    "disk_io": 4, "mysql_io": 4,
    "threads": 5, "sessions": 5,
    "fgc": 6,
    # Additional features beyond 8 can use index 7 for residual
    "signal": 2,             # maps to net_rx slot (closest semantic)
    "packet_loss": 5,        # maps to threads/sessions slot
    "bandwidth": 3,          # maps to net_tx slot
}


def build_telemetry_tensor(
    events: List[ObservationEvent],
    entity_ids: List[str],
    kpi_to_ob: Dict[str, int],
    max_ob_features: int = 8,
    resample_interval_sec: int = 60,
    window_start_ts: Optional[float] = None,
    window_end_ts: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense [T, N, D] tensor + obs_mask from flat event list.

    Args:
        events: All observation events (pre-filtered to date range).
        entity_ids: Ordered list of entity IDs (determines N).
        kpi_to_ob: Mapping from KPI name to OB feature index.
        max_ob_features: Maximum number of OB feature dimensions (default 8).
        resample_interval_sec: Resampling grid interval in seconds.
        window_start_ts: Optional UNIX timestamp filter start.
        window_end_ts: Optional UNIX timestamp filter end.

    Returns:
        tensor: [T, N, D] float32 observation values.
        obs_mask: [T, N, D] boolean mask of observed data points.
        timestamps: [T] float64 timestamps (UNIX seconds).
    """
    N = len(entity_ids)
    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    # Collect all feature names that map to valid OB indices
    D = max_ob_features

    # Determine time grid
    # Sanity filter: timestamps must be in valid UNIX range (2000-2100)
    MIN_TS = 946684800   # 2000-01-01
    MAX_TS = 4102444800  # 2100-01-01
    all_ts = sorted(set(ev.timestamp for ev in events
                        if ev.entity_id in eid_to_idx
                        and ev.feature_name in kpi_to_ob
                        and MIN_TS <= ev.timestamp <= MAX_TS))
    if not all_ts:
        return np.zeros((0, N, D), dtype=np.float32), \
               np.zeros((0, N, D), dtype=np.float32), \
               np.zeros((0,), dtype=np.float64)

    # Build fixed-interval time grid
    # Snap to the nearest resample_interval_sec boundary
    t_min = all_ts[0]
    t_max = all_ts[-1]

    # Apply window filter if provided
    if window_start_ts is not None and window_end_ts is not None:
        t_min = max(t_min, window_start_ts)
        t_max = min(t_max, window_end_ts)

    # Snap to grid
    t_grid_min = (t_min // resample_interval_sec) * resample_interval_sec
    t_grid_max = ((t_max + resample_interval_sec - 1) // resample_interval_sec) * resample_interval_sec
    grid_ts = np.arange(t_grid_min, t_grid_max + 1, resample_interval_sec, dtype=np.float64)
    if len(grid_ts) == 0:
        return np.zeros((0, N, D), dtype=np.float32), \
               np.zeros((0, N, D), dtype=np.float32), \
               np.zeros((0,), dtype=np.float64)

    T = len(grid_ts)
    ts_to_grid = {ts: i for i, ts in enumerate(grid_ts)}

    tensor = np.zeros((T, N, D), dtype=np.float32)
    counts = np.zeros((T, N, D), dtype=np.float32)

    for ev in events:
        if ev.entity_id not in eid_to_idx:
            continue
        fn = ev.feature_name
        if fn not in kpi_to_ob:
            continue
        if ev.timestamp < MIN_TS or ev.timestamp > MAX_TS:
            continue
        ob_idx = kpi_to_ob[fn]
        if ob_idx >= D:
            continue

        # Snap event timestamp to nearest grid point
        grid_ts_val = (ev.timestamp // resample_interval_sec) * resample_interval_sec
        t = ts_to_grid.get(grid_ts_val)
        if t is None:
            continue
        n = eid_to_idx[ev.entity_id]
        tensor[t, n, ob_idx] += ev.value
        counts[t, n, ob_idx] += 1

    # Average at each grid point
    mask_bool = counts > 0
    for n_idx in range(N):
        for d_idx in range(D):
            m = counts[:, n_idx, d_idx] > 0
            if m.any():
                tensor[m, n_idx, d_idx] /= counts[m, n_idx, d_idx]

    # Forward fill missing values
    for n_idx in range(N):
        for d_idx in range(D):
            last = 0.0
            for t in range(T):
                if counts[t, n_idx, d_idx] > 0:
                    last = tensor[t, n_idx, d_idx]
                else:
                    tensor[t, n_idx, d_idx] = last

    obs_mask = mask_bool.astype(np.float32)

    return tensor, obs_mask, grid_ts


def build_episode(
    events: List[ObservationEvent],
    all_entities: List[Entity],
    query: InferenceQuery,  # NO GT — only observation metadata
    burn_in_min: int = 60,
    resample_interval_sec: int = 60,
    ob_mean: Optional[np.ndarray] = None,
    ob_std: Optional[np.ndarray] = None,
    max_ob_features: int = 8,
    type_str_to_idx: Optional[Dict[str, int]] = None,
    all_zero_type: bool = False,
) -> Dict:
    """Build a per-query Episode for strict evaluation.

    Args:
        events: Pre-loaded observation events for the system.
        all_entities: All discovered entities for the system.
        query: Parsed query with observation window.
        burn_in_min: Burn-in period before window_start (minutes).
        resample_interval_sec: Time grid interval in seconds.
        ob_mean: [1, 1, D] Online Boutique feature means for normalization.
        ob_std: [1, 1, D] Online Boutique feature stds for normalization.
        max_ob_features: Number of OB feature dimensions.
        type_str_to_idx: Mapping from type string to integer index.
        all_zero_type: If True, all entities get type_idx=0.

    Returns:
        Dict with keys:
            tensor: [T, N, D] normalized observation tensor.
            obs_mask: [T, N, D] boolean observation mask.
            timestamps: [T] timestamp grid (UNIX seconds).
            entity_ids: [N] list of entity ID strings.
            entity_types: [N] list of entity type strings.
            type_idx: [N] integer type indices.
            window_start_ts: float UNIX timestamp of window start.
            window_end_ts: float UNIX timestamp of window end.
            query: InferenceQuery (reference, NO GT).
    """
    if type_str_to_idx is None:
        type_str_to_idx = {"container": 0, "database": 1, "middleware": 2, "host": 0}

    # Use container/service entities only (matching original parse_system logic)
    cont_entities = [e for e in all_entities
                     if e.entity_type.value in ("container", "service")]
    if not cont_entities:
        cont_entities = all_entities

    entity_ids = [e.entity_id for e in cont_entities]
    N = len(entity_ids)
    type_strs = [assign_entity_type(eid) for eid in entity_ids] if not all_zero_type \
                else ["container"] * N
    type_indices = np.array(
        [type_str_to_idx.get(t, 0) if not all_zero_type else 0 for t in type_strs],
        dtype=np.int32,
    )

    # Convert query window to UNIX timestamps
    window_start_ts = query.window_start.timestamp()
    window_end_ts = query.window_end.timestamp()

    # Include burn-in period before window
    burn_in_start = query.window_start - timedelta(minutes=burn_in_min)
    data_start_ts = burn_in_start.timestamp()
    data_end_ts = window_end_ts

    # Build tensor from events within the time range
    tensor, obs_mask, timestamps = build_telemetry_tensor(
        events=events,
        entity_ids=entity_ids,
        kpi_to_ob=KPI_TO_OB,
        max_ob_features=max_ob_features,
        resample_interval_sec=resample_interval_sec,
        window_start_ts=data_start_ts,
        window_end_ts=data_end_ts,
    )

    if tensor.shape[0] == 0:
        # Empty episode
        return {
            "tensor": np.zeros((0, N, max_ob_features), dtype=np.float32),
            "obs_mask": np.zeros((0, N, max_ob_features), dtype=np.float32),
            "timestamps": np.array([], dtype=np.float64),
            "entity_ids": entity_ids,
            "entity_types": type_strs,
            "type_idx": type_indices,
            "window_start_ts": window_start_ts,
            "window_end_ts": window_end_ts,
            "query": query,
        }

    # Normalize with OB stats if provided
    tensor_norm = tensor.copy()
    if ob_mean is not None and ob_std is not None:
        tensor_norm = np.nan_to_num(
            (tensor - ob_mean) / (ob_std + 1e-6), nan=0.0)

    return {
        "tensor": tensor_norm,
        "obs_mask": obs_mask,
        "timestamps": timestamps,
        "entity_ids": entity_ids,
        "entity_types": type_strs,
        "type_idx": type_indices,
        "window_start_ts": window_start_ts,
        "window_end_ts": window_end_ts,
        "query": query,
    }


def build_episodes_for_queries(
    events: List[ObservationEvent],
    all_entities: List[Entity],
    queries: List[InferenceQuery],
    burn_in_min: int = 60,
    resample_interval_sec: int = 60,
    ob_mean: Optional[np.ndarray] = None,
    ob_std: Optional[np.ndarray] = None,
    max_ob_features: int = 8,
    all_zero_type: bool = False,
) -> List[Dict]:
    """Build episodes for a list of queries.

    This is a convenience wrapper that builds episodes one by one.
    For production use, consider batching queries that share date ranges.
    """
    episodes = []
    type_str_to_idx = {"container": 0, "database": 1, "middleware": 2, "host": 0}

    for query in queries:
        ep = build_episode(
            events=events,
            all_entities=all_entities,
            query=query,
            burn_in_min=burn_in_min,
            resample_interval_sec=resample_interval_sec,
            ob_mean=ob_mean,
            ob_std=ob_std,
            max_ob_features=max_ob_features,
            type_str_to_idx=type_str_to_idx,
            all_zero_type=all_zero_type,
        )
        episodes.append(ep)

    return episodes
