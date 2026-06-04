"""Base adapter protocol for RCAWorld-Foundation.

All dataset adapters must implement this interface. The adapter is responsible
for transforming raw telemetry data (metrics, logs, traces, alerts) into
SystemEpisode objects with typed entities, relations, and observation events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np

from ..schema.entity import Entity, EntityType
from ..schema.relation import Relation, RelationType
from ..schema.event import ObservationEvent, Modality, EventBatch
from ..schema.episode import SystemEpisode, QueryMask, RootCauseLabel


class AdapterProtocol(ABC):
    """Protocol for dataset adapters.

    Each adapter:
    1. Discovers entities and their types from raw data.
    2. Extracts observation features per entity type.
    3. Builds SystemEpisode objects for training or evaluation.
    4. Provides event batching for efficient model consumption.
    """

    @abstractmethod
    def discover_entities(self, data_dir: str) -> List[Entity]:
        """Discover all entities in the dataset and their types."""
        ...

    @abstractmethod
    def extract_events(self, data_dir: str,
                       entities: List[Entity]) -> List[ObservationEvent]:
        """Extract all observation events from raw telemetry."""
        ...

    @abstractmethod
    def extract_relations(self, data_dir: str,
                          entities: List[Entity]) -> List[Relation]:
        """Extract explicit relations between entities."""
        ...

    @abstractmethod
    def extract_labels(self, data_dir: str,
                       entities: List[Entity]) -> List[RootCauseLabel]:
        """Extract root cause labels (optional, for evaluation)."""
        ...

    def build_episode(self, data_dir: str,
                      system_id: str,
                      filter_normal_only: bool = False,
                      query: Optional[str] = None,
                      query_mask: Optional[QueryMask] = None,
                      ) -> SystemEpisode:
        """Full pipeline: build a SystemEpisode from raw data.

        Args:
            data_dir: Path to the dataset directory.
            system_id: Unique identifier for this system.
            filter_normal_only: If True, exclude fault windows from events.
            query: Natural language query string.
            query_mask: Structured query mask.

        Returns:
            SystemEpisode with entities, events, relations, and optional labels.
        """
        entities = self.discover_entities(data_dir)
        events = self.extract_events(data_dir, entities)
        relations = self.extract_relations(data_dir, entities)
        labels = self.extract_labels(data_dir, entities)

        # Build normal mask
        if labels:
            fault_ts = set()
            for lb in labels:
                try:
                    fault_ts.add(float(lb.occurrence_datetime))
                except (ValueError, TypeError):
                    pass
            # Mark a window around each fault timestamp as anomalous
            normal_mask = np.ones(len(events), dtype=bool)
            # TODO: refine with actual timestamp matching
        else:
            normal_mask = None

        if filter_normal_only and labels:
            # Filter out events during fault windows
            pass  # deferred to batch construction

        return SystemEpisode(
            system_id=system_id,
            entities=entities,
            relations=relations,
            events=events,
            normal_mask=normal_mask,
            query=query,
            query_mask=query_mask,
            labels=labels,
        )

    @abstractmethod
    def events_to_batch(self, events: List[ObservationEvent],
                        entities: List[Entity],
                        window_size: int = 24,
                        stride: int = 12) -> EventBatch:
        """Convert flat event list to dense batched tensor representation."""
        ...


class BaseAdapter(AdapterProtocol):
    """Base adapter with common utilities for CSV/Parquet datasets."""

    system_name: str = "generic"

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _normalize_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """Convert to relative seconds from first timestamp."""
        if len(timestamps) == 0:
            return timestamps
        t0 = timestamps[0]
        return timestamps - t0

    def _build_normal_mask(self,
                           timestamps: np.ndarray,
                           labels: List[RootCauseLabel],
                           window_sec: float = 600.0) -> np.ndarray:
        """Build boolean mask: True for timestamps NOT near any fault."""
        mask = np.ones(len(timestamps), dtype=bool)
        if not labels:
            return mask
        for lb in labels:
            try:
                ft = float(lb.occurrence_datetime)
            except (ValueError, TypeError):
                continue
            # Apply pre/post fault window
            pre_s = ft - window_sec
            post_s = ft + window_sec
            mask[(timestamps >= pre_s) & (timestamps <= post_s)] = False
        return mask
