"""OpenRCA Bank adapter.

Bank is a banking system with 11 services and up to 100+ container-level
entities. The data includes metric_container.csv (container KPIs),
metric_service.csv (service KPIs), log CSV files, and record.csv (labels).

Entity model:
- SERVICE entities from metric_service.csv columns
- CONTAINER entities from metric_container.csv cmdb_id
- Relations: service -> container (DEPLOYED_ON) via fuzzy name matching
"""

import os
import re
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set
from glob import glob

import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema.entity import Entity, EntityType
from ..schema.relation import Relation, RelationType
from ..schema.event import ObservationEvent, Modality, EventBatch
from ..schema.episode import RootCauseLabel


class OpenRCABankAdapter(BaseAdapter):
    """Adapter for OpenRCA Bank dataset.

    Directory structure:
        Bank/Bank/
        ├── record.csv
        ├── SUPPORT.md
        └── telemetry/
            ├── 2021-03-04/
            │   ├── metric/
            │   │   ├── metric_service.csv
            │   │   └── metric_container.csv
            │   ├── log/
            │   │   └── ...
            │   └── trace/
            │       └── ...
            └── ...
    """

    system_name = "bank"

    # KPI patterns for container metrics (from existing parser)
    CONTAINER_KPI_PATTERNS = [
        ("cpu", re.compile(
            r".*CpuPercent$|.*cpu.*percent|.*cpu_percent|.*CPU.*CPUCpuUtil", re.I)),
        ("mem", re.compile(
            r".*MemPercent$|.*mem.*percent|.*mem_percent", re.I)),
        ("mem_usage", re.compile(r".*MemUsage$|.*mem_used|.*mem.*usage", re.I)),
        ("net_rx", re.compile(
            r".*NetworkRxBytes$|.*rx_bytes|.*net.*rx|.*Incoming_network", re.I)),
        ("net_tx", re.compile(
            r".*NetworkTxBytes$|.*tx_bytes|.*net.*tx|.*Outgoing_network", re.I)),
        ("disk_io", re.compile(r".*Disk_.*|.*disk.*io", re.I)),
        ("threads", re.compile(r".*thread.*used|.*thread.*pct", re.I)),
        ("sessions", re.compile(r".*session.*used", re.I)),
        ("fgc", re.compile(r".*fgc|.*full.*gc", re.I)),
        ("mysql_io", re.compile(r".*Innodb.*|.*mysql.*io", re.I)),
        ("jvm_cpu", re.compile(r".*jvm.*cpu|.*JVM.*CPU", re.I)),
        ("jvm_mem", re.compile(r".*jvm.*mem|.*JVM.*heap|.*JVM.*OOM", re.I)),
    ]

    # Service metric column mapping
    SERVICE_METRIC_COLS = {
        "rr": "request_rate",
        "request_rate": "request_rate",
        "sr": "success_rate",
        "success_rate": "success_rate",
        "succee_rate": "success_rate",
        "mrt": "latency_avg",
        "avg_time": "latency_avg",
        "latency": "latency_avg",
        "num": "request_rate",
        "cnt": "request_rate",
    }

    def __init__(self, max_days: int = 5, max_container_timestamps: int = 5000,
                 max_container_events: int = 50000, max_container_rows: int = 300000,
                 include_dates: Optional[Set[str]] = None):
        self.max_days = max_days
        self.max_container_timestamps = max_container_timestamps
        self.max_container_events = max_container_events
        self.max_container_rows = max_container_rows
        self.include_dates = include_dates  # if set, only load these date dirs

    def _get_days(self, data_dir: str) -> List[str]:
        telemetry_dir = os.path.join(data_dir, "telemetry")
        if not os.path.exists(telemetry_dir):
            return []
        days = sorted([d for d in os.listdir(telemetry_dir)
                        if d.startswith("20")])
        if self.include_dates is not None:
            days = [d for d in days if d in self.include_dates]
        return days[:self.max_days]

    def _map_kpi_to_category(self, kpi_name: str) -> Optional[str]:
        for cat, pat in self.CONTAINER_KPI_PATTERNS:
            if pat.match(str(kpi_name)):
                return cat
        return None

    # ---- Entity Discovery ----

    def discover_entities(self, data_dir: str) -> List[Entity]:
        entities: List[Entity] = []
        seen_ids: set = set()

        # 1. Service entities from metric_service.csv
        services = self._discover_services(data_dir)
        for svc in services:
            if svc not in seen_ids:
                seen_ids.add(svc)
                entities.append(Entity(
                    entity_id=svc,
                    entity_type=EntityType.SERVICE,
                ))

        # 2. Container entities from metric_container.csv
        containers = self._discover_containers(data_dir)
        for cont in containers:
            if cont not in seen_ids:
                seen_ids.add(cont)
                entities.append(Entity(
                    entity_id=cont,
                    entity_type=EntityType.CONTAINER,
                ))

        return entities

    def _discover_services(self, data_dir: str) -> List[str]:
        days = self._get_days(data_dir)
        services: set = set()
        for day in days:
            metric_dir = os.path.join(data_dir, "telemetry", day, "metric")
            # Try metric_service.csv first, then metric_app.csv (Bank uses metric_app)
            for fname in ["metric_service.csv", "metric_app.csv"]:
                mfile = os.path.join(metric_dir, fname)
                if not os.path.exists(mfile):
                    continue
                try:
                    df = pd.read_csv(mfile, nrows=10000)
                    if df.empty:
                        continue
                    svc_col = next(
                        (c for c in ["service", "serviceName", "tc"] if c in df.columns), None)
                    if svc_col:
                        services.update(df[svc_col].dropna().astype(str).unique())
                    break
                except Exception:
                    continue
        return sorted(services)

    def _discover_containers(self, data_dir: str) -> List[str]:
        days = self._get_days(data_dir)
        containers: set = set()
        for day in days:
            mfile = os.path.join(data_dir, "telemetry", day,
                                 "metric", "metric_container.csv")
            if not os.path.exists(mfile):
                continue
            try:
                for chunk in pd.read_csv(mfile, chunksize=500000):
                    cols = list(chunk.columns)
                    entity_col = next(
                        (c for c in cols if c.lower() in ("cmdb_id", "cmdb_id")), cols[1])
                    containers.update(
                        chunk[entity_col].dropna().astype(str).unique())
                    if len(containers) > 500:
                        break
            except Exception:
                continue
        return sorted(containers)

    # ---- Event Extraction ----

    def extract_events(self, data_dir: str,
                       entities: List[Entity]) -> List[ObservationEvent]:
        entity_map = {e.entity_id: e for e in entities}
        events: List[ObservationEvent] = []

        service_entities = [e for e in entities if e.entity_type == EntityType.SERVICE]
        container_entities = [e for e in entities if e.entity_type == EntityType.CONTAINER]

        days = self._get_days(data_dir)
        for day in days:
            metric_dir = os.path.join(data_dir, "telemetry", day, "metric")

            # Service metrics
            if service_entities:
                events.extend(self._extract_service_events(
                    metric_dir, service_entities))

            # Container metrics
            if container_entities:
                events.extend(self._extract_container_events(
                    metric_dir, container_entities))

        return events

    def _extract_service_events(self, metric_dir: str,
                                entities: List[Entity]) -> List[ObservationEvent]:
        events: List[ObservationEvent] = []
        entity_ids = {e.entity_id for e in entities}

        # Try both metric_service.csv and metric_app.csv
        for fname in ["metric_service.csv", "metric_app.csv"]:
            mfile = os.path.join(metric_dir, fname)
            if not os.path.exists(mfile):
                continue
            try:
                df = pd.read_csv(mfile, nrows=200000)
            except Exception:
                continue
            if df.empty:
                continue

            ts_col = next(
                (c for c in ["timestamp", "startTime", "time"] if c in df.columns), None)
            if ts_col is None:
                continue

            svc_col = next(
                (c for c in ["service", "serviceName", "tc"] if c in df.columns), None)
            if svc_col is None:
                continue

            # Normalize timestamps
            if ts_col == "startTime":
                df["_ts"] = (df[ts_col].astype(float) / 1000).astype(int)
            else:
                df["_ts"] = df[ts_col].astype(int)

            for _, row in df.iterrows():
                svc = str(row[svc_col])
                if svc not in entity_ids:
                    continue
                ts = self._safe_int(row["_ts"])

                for raw_col, feat_name in self.SERVICE_METRIC_COLS.items():
                    if raw_col in df.columns:
                        val = self._safe_float(row[raw_col])
                        events.append(ObservationEvent(
                            timestamp=ts,
                            entity_id=svc,
                            modality=Modality.METRIC,
                            feature_name=feat_name,
                            value=val,
                        ))
            break  # stop after first successful file
        return events

    def _extract_container_events(self, metric_dir: str,
                                  entities: List[Entity]) -> List[ObservationEvent]:
        mfile = os.path.join(metric_dir, "metric_container.csv")
        if not os.path.exists(mfile):
            return []

        events: List[ObservationEvent] = []
        entity_ids = {e.entity_id for e in entities}
        total_rows = 0

        try:
            for chunk in pd.read_csv(mfile, chunksize=100000):
                total_rows += len(chunk)
                cols = list(chunk.columns)
                ts_col = next(
                    (c for c in cols if c.lower() == "timestamp"), cols[0])
                entity_col = next(
                    (c for c in cols if c.lower() in ("cmdb_id", "cmdb_id")),
                    cols[1] if len(cols) > 1 else cols[0])
                kpi_col = next(
                    (c for c in cols if c.lower() in ("kpi_name", "name")),
                    cols[2] if len(cols) > 2 else cols[0])
                val_col = next(
                    (c for c in cols if c.lower() in ("value",)), cols[-1])

                chunk[ts_col] = chunk[ts_col].astype(int)

                for _, row in chunk.iterrows():
                    entity = str(row[entity_col])
                    if entity not in entity_ids:
                        continue
                    kpi = str(row[kpi_col])
                    cat = self._map_kpi_to_category(kpi)
                    if cat is None:
                        continue
                    ts = self._safe_int(row[ts_col])
                    val = self._safe_float(row[val_col])
                    events.append(ObservationEvent(
                        timestamp=ts,
                        entity_id=entity,
                        modality=Modality.METRIC,
                        feature_name=cat,
                        value=val,
                    ))
                    if len(events) >= self.max_container_events:
                        break
                if len(events) >= self.max_container_events or total_rows >= self.max_container_rows:
                    break
        except Exception:
            pass

        return events

    # ---- Relations ----

    def extract_relations(self, data_dir: str,
                          entities: List[Entity]) -> List[Relation]:
        relations: List[Relation] = []
        services = [e for e in entities if e.entity_type == EntityType.SERVICE]
        containers = [e for e in entities if e.entity_type == EntityType.CONTAINER]

        # Map containers to services via name matching
        for cont in containers:
            cont_lower = cont.lower().replace("_", "").replace("-", "").replace(" ", "")
            for svc in services:
                svc_lower = svc.lower().replace("_", "").replace("-", "").replace(" ", "")
                if svc_lower in cont_lower or cont_lower in svc_lower:
                    relations.append(Relation(
                        source_id=svc.entity_id,
                        target_id=cont.entity_id,
                        relation_type=RelationType.DEPLOYED_ON,
                    ))
                    break

        return relations

    # ---- Labels ----

    def extract_labels(self, data_dir: str,
                       entities: List[Entity]) -> List[RootCauseLabel]:
        record_file = os.path.join(data_dir, "record.csv")
        if not os.path.exists(record_file):
            return []

        entity_ids = {e.entity_id for e in entities}
        labels: List[RootCauseLabel] = []

        try:
            df = pd.read_csv(record_file)
        except Exception:
            return labels

        for _, row in df.iterrows():
            ts = str(row.get("timestamp", ""))
            component = str(row.get("component", ""))
            reason = str(row.get("reason", ""))
            level = str(row.get("level", "pod"))

            # Map component to entity id
            mapped_id = self._map_component(component, entity_ids)
            if mapped_id is None:
                continue

            labels.append(RootCauseLabel(
                occurrence_datetime=ts,
                component=mapped_id,
                reason=reason,
                level=level,
            ))

        return labels

    def _map_component(self, component: str,
                       entity_ids: set) -> Optional[str]:
        """Fuzzy match a fault component to a discovered entity ID."""
        comp_lower = component.lower().replace("_", "").replace("-", "").replace(" ", "")

        for eid in entity_ids:
            eid_lower = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
            if comp_lower == eid_lower:
                return eid
            if comp_lower in eid_lower or eid_lower in comp_lower:
                return eid

        return None

    # ---- Event Batching ----

    def events_to_batch(self, events: List[ObservationEvent],
                        entities: List[Entity],
                        window_size: int = 24,
                        stride: int = 12) -> EventBatch:
        if not events:
            raise ValueError("No events to batch")

        entity_ids = [e.entity_id for e in entities]
        eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        N = len(entities)

        # Collect all feature names across all events
        feature_set: set = set()
        for ev in events:
            feature_set.add(ev.feature_name)
        feature_names = sorted(feature_set)
        feat_to_idx = {f: i for i, f in enumerate(feature_names)}
        D = len(feature_names)

        # Sort timestamps
        all_ts = sorted(set(ev.timestamp for ev in events))
        ts_to_idx = {t: i for i, t in enumerate(all_ts)}
        T = len(all_ts)

        # Build dense tensor
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

        # Forward fill missing values
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
