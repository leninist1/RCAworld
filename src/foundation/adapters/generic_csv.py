"""Generic CSV adapter for arbitrary datasets.

Handles any CSV-based dataset with minimal configuration.
Users specify entity_id column, timestamp column, and feature columns.
Entities are assigned GENERIC type by default.
"""

import os
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema.entity import Entity, EntityType
from ..schema.relation import Relation
from ..schema.event import ObservationEvent, Modality, EventBatch
from ..schema.episode import RootCauseLabel


class GenericCSVAdapter(BaseAdapter):
    """Adapter for generic CSV-based telemetry datasets.

    Configuration:
        metric_files: List of CSV file paths (or glob patterns).
        entity_col: Column name for entity identifier.
        timestamp_col: Column name for timestamp.
        feature_cols: List of column names to use as features.
        label_file: Optional CSV file with root cause labels.

    Example:
        adapter = GenericCSVAdapter(
            metric_files=["data/system_a/metrics.csv"],
            entity_col="hostname",
            timestamp_col="time",
            feature_cols=["cpu", "mem", "disk_io", "net_rx", "net_tx"],
            label_file="data/system_a/faults.csv",
            label_entity_col="node",
            label_ts_col="start_time",
            label_reason_col="fault_type",
        )
    """

    system_name = "generic_csv"

    def __init__(self,
                 metric_files: List[str],
                 entity_col: str,
                 timestamp_col: str,
                 feature_cols: List[str],
                 entity_type: EntityType = EntityType.GENERIC,
                 label_file: Optional[str] = None,
                 label_entity_col: Optional[str] = None,
                 label_ts_col: Optional[str] = None,
                 label_reason_col: Optional[str] = None,
                 label_level_col: Optional[str] = None,
                 ):
        self.metric_files = metric_files
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.feature_cols = feature_cols
        self.entity_type = entity_type
        self.label_file = label_file
        self.label_entity_col = label_entity_col or entity_col
        self.label_ts_col = label_ts_col or timestamp_col
        self.label_reason_col = label_reason_col or "reason"
        self.label_level_col = label_level_col or "level"

        self._entities: Optional[List[Entity]] = None

    def discover_entities(self, data_dir: str = "") -> List[Entity]:
        if self._entities is not None:
            return self._entities

        entity_ids: set = set()
        for mfile in self.metric_files:
            if not os.path.exists(mfile):
                continue
            try:
                df = pd.read_csv(mfile, nrows=50000)
                if self.entity_col in df.columns:
                    entity_ids.update(
                        df[self.entity_col].dropna().astype(str).unique())
            except Exception:
                continue

        entities = [
            Entity(entity_id=eid, entity_type=self.entity_type)
            for eid in sorted(entity_ids)
        ]
        self._entities = entities
        return entities

    def extract_events(self, data_dir: str = "",
                       entities: List[Entity] = None) -> List[ObservationEvent]:
        if entities is None:
            entities = self._entities or self.discover_entities()
        entity_ids = {e.entity_id for e in entities}
        events: List[ObservationEvent] = []

        for mfile in self.metric_files:
            if not os.path.exists(mfile):
                continue
            try:
                df = pd.read_csv(mfile, nrows=200000)
            except Exception:
                continue

            if self.entity_col not in df.columns or self.timestamp_col not in df.columns:
                continue

            df[self.timestamp_col] = df[self.timestamp_col].astype(int)

            for _, row in df.iterrows():
                entity = str(row[self.entity_col])
                if entity not in entity_ids:
                    continue
                ts = self._safe_int(row[self.timestamp_col])

                for feat in self.feature_cols:
                    if feat not in df.columns:
                        continue
                    val = self._safe_float(row[feat])
                    events.append(ObservationEvent(
                        timestamp=ts,
                        entity_id=entity,
                        modality=Modality.METRIC,
                        feature_name=feat,
                        value=val,
                    ))

        return events

    def extract_relations(self, data_dir: str = "",
                          entities: List[Entity] = None) -> List[Relation]:
        return []

    def extract_labels(self, data_dir: str = "",
                       entities: List[Entity] = None) -> List[RootCauseLabel]:
        if self.label_file is None or not os.path.exists(self.label_file):
            return []

        if entities is None:
            entities = self._entities or self.discover_entities()
        entity_ids = {e.entity_id for e in entities}
        labels: List[RootCauseLabel] = []

        try:
            df = pd.read_csv(self.label_file)
        except Exception:
            return labels

        for _, row in df.iterrows():
            ts = str(row.get(self.label_ts_col, ""))
            component = str(row.get(self.label_entity_col, ""))
            reason = str(row.get(self.label_reason_col, ""))
            level = str(row.get(self.label_level_col, ""))

            # Direct or fuzzy match to entity
            if component in entity_ids:
                pass
            else:
                for eid in entity_ids:
                    comp_lower = component.lower().replace("_", "").replace("-", "").replace(" ", "")
                    eid_lower = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
                    if comp_lower in eid_lower or eid_lower in comp_lower:
                        component = eid
                        break
                else:
                    continue

            labels.append(RootCauseLabel(
                occurrence_datetime=ts,
                component=component,
                reason=reason,
                level=level,
            ))

        return labels

    def events_to_batch(self, events: List[ObservationEvent],
                        entities: List[Entity],
                        window_size: int = 24,
                        stride: int = 12) -> EventBatch:
        if not events:
            raise ValueError("No events to batch")

        entity_ids = [e.entity_id for e in entities]
        eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        N = len(entities)

        feature_set: set = set()
        for ev in events:
            feature_set.add(ev.feature_name)
        feature_names = sorted(feature_set)
        feat_to_idx = {f: i for i, f in enumerate(feature_names)}
        D = len(feature_names)

        all_ts = sorted(set(ev.timestamp for ev in events))
        ts_to_idx = {t: i for i, t in enumerate(all_ts)}
        T = len(all_ts)

        event_tensor = np.zeros((T, N, D), dtype=np.float32)
        event_mask = np.zeros((T, N, D), dtype=np.float32)

        for ev in events:
            if ev.entity_id not in eid_to_idx:
                continue
            if ev.feature_name not in feat_to_idx:
                continue
            t = ts_to_idx.get(ev.timestamp)
            if t is None:
                continue
            n = eid_to_idx[ev.entity_id]
            d = feat_to_idx[ev.feature_name]
            event_tensor[t, n, d] = ev.value
            event_mask[t, n, d] = ev.quality_mask

        for n in range(N):
            for d in range(D):
                last_val = 0.0
                for t in range(T):
                    if event_mask[t, n, d] > 0:
                        last_val = event_tensor[t, n, d]
                    else:
                        event_tensor[t, n, d] = last_val
                        event_mask[t, n, d] = 1.0

        entity_types = [e.entity_type.value for e in entities]

        return EventBatch(
            timestamps=np.array(all_ts, dtype=np.float64),
            entity_ids=entity_ids,
            event_tensor=event_tensor,
            event_mask=event_mask,
            modality_mask=np.ones((T, N, D, 1), dtype=np.float32),
            feature_names=feature_names,
            entity_types=entity_types,
        )
