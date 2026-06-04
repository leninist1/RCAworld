"""Observation event schema for multi-modal telemetry events."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np


class Modality(str, Enum):
    """Telemetry modality types."""
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"
    ALERT = "alert"
    CHANGE = "change"


@dataclass
class ObservationEvent:
    """A single observation at a point in time for one entity.

    Unified representation for metrics, logs, traces, alerts, and changes.

    Attributes:
        timestamp: Unix timestamp (seconds or milliseconds, system-consistent).
        entity_id: Which entity this observation belongs to.
        modality: METRIC, LOG, TRACE, ALERT, or CHANGE.
        feature_name: Name of the feature (e.g. 'cpu_usage', 'ERROR_LOCK_WAIT').
        value: Scalar value for metrics, or event count/indicator for logs/traces.
        embedding: Optional pre-computed embedding (for log templates, trace spans).
        quality_mask: 1.0 = valid, 0.0 = missing/corrupted.
        metadata: Any additional fields (e.g. log severity, trace span_id).
    """

    timestamp: float
    entity_id: str
    modality: Modality = Modality.METRIC
    feature_name: str = ""
    value: float = 0.0
    embedding: Optional[np.ndarray] = None
    quality_mask: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventBatch:
    """A time-ordered batch of observation events.

    Efficient batched representation for training. Events are sorted by
    timestamp, then entity_id, then modality.

    Attributes:
        timestamps: [T] sorted unique timestamps.
        entity_ids: [N] entity identifiers.
        event_tensor: [T, N, D_max] dense feature matrix.
        event_mask: [T, N, D_max] validity mask (1.0 = valid).
        modality_mask: [T, N, D_max] one-hot modality indicator.
        feature_names: [D_max] ordered feature names.
        entity_types: [N] entity type for each entity.
        relation_graph: (edge_index, edge_type, edge_features) or None.
    """

    timestamps: np.ndarray          # [T]
    entity_ids: List[str]           # [N]
    event_tensor: np.ndarray        # [T, N, D]
    event_mask: np.ndarray          # [T, N, D]
    modality_mask: np.ndarray       # [T, N, D, num_modalities]
    feature_names: List[str]
    entity_types: List[str]
    relation_graph: Optional[Dict] = None
