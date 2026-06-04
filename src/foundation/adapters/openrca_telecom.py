"""OpenRCA Telecom adapter.

Telecom is a telecommunications infrastructure system with heterogeneous
entities: hosts, network interfaces, services, and processes.
The data uses metric_app.csv (different column names than Bank/Market).

Entity model:
- SERVICE entities from metric_app.csv 'serviceName' column
- HOST entities (inferred from container/process data)
- NETWORK_INTERFACE entities (if available)
"""

import os
import re
from typing import List, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema.entity import Entity, EntityType
from ..schema.relation import Relation, RelationType
from ..schema.event import ObservationEvent, Modality, EventBatch
from ..schema.episode import RootCauseLabel


class OpenRCATelecomAdapter(BaseAdapter):
    """Adapter for OpenRCA Telecom dataset.

    Directory structure:
        Telecom/Telecom/
        ├── record.csv
        └── telemetry/
            └── {date}/
                ├── metric/
                │   ├── metric_app.csv
                │   └── metric_container.csv
                ├── log/
                └── trace/
    """

    system_name = "telecom"

    # Telecom service/app metric columns
    APP_METRIC_COLS = {
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
        "qps": "request_rate",
        "error_rate": "error_rate",
        "err_rate": "error_rate",
    }

    # Container KPI patterns (same as Bank, plus telecom-specific)
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
        ("disk_io", re.compile(r".*Disk_.*|.*disk.*io|.*disk_util", re.I)),
        ("threads", re.compile(r".*thread.*used|.*thread.*pct", re.I)),
        ("sessions", re.compile(r".*session.*used", re.I)),
        ("fgc", re.compile(r".*fgc|.*full.*gc|.*jvm.*gc", re.I)),
        ("jvm_cpu", re.compile(r".*jvm.*cpu|.*JVM.*CPU", re.I)),
        ("jvm_mem", re.compile(r".*jvm.*mem|.*JVM.*heap|.*JVM.*OOM", re.I)),
        # Telecom-specific
        ("signal", re.compile(r".*signal.*quality|.*rssi", re.I)),
        ("packet_loss", re.compile(r".*packet.*loss|.*drop.*rate", re.I)),
        ("bandwidth", re.compile(r".*bandwidth|.*throughput", re.I)),
    ]

    def __init__(self, max_days: int = 5, max_container_timestamps: int = 5000,
                 include_dates: Optional[Set[str]] = None):
        self.max_days = max_days
        self.max_container_timestamps = max_container_timestamps
        self.include_dates = include_dates

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
        seen_ids: Set[str] = set()

        # 1. Service/App entities from metric_app.csv
        services = self._discover_services(data_dir)
        for svc in services:
            if svc not in seen_ids:
                seen_ids.add(svc)
                entities.append(Entity(
                    entity_id=svc,
                    entity_type=EntityType.SERVICE,
                ))

        # 2. Container/host entities from metric_container.csv
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
        services: Set[str] = set()
        for day in days:
            # Try metric_app.csv first, then metric_service.csv
            for fname in ["metric_app.csv", "metric_service.csv"]:
                mfile = os.path.join(data_dir, "telemetry", day,
                                     "metric", fname)
                if not os.path.exists(mfile):
                    continue
                try:
                    df = pd.read_csv(mfile, nrows=10000)
                    svc_col = next(
                        (c for c in ["serviceName", "service", "tc",
                         "app"] if c in df.columns), None)
                    if svc_col:
                        services.update(
                            df[svc_col].dropna().astype(str).unique())
                    break
                except Exception:
                    continue
        return sorted(services)

    def _discover_containers(self, data_dir: str) -> List[str]:
        days = self._get_days(data_dir)
        containers: Set[str] = set()
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

        service_entities = [e for e in entities
                           if e.entity_type == EntityType.SERVICE]
        container_entities = [e for e in entities
                              if e.entity_type == EntityType.CONTAINER]

        days = self._get_days(data_dir)
        for day in days:
            metric_dir = os.path.join(data_dir, "telemetry", day, "metric")

            if service_entities:
                events.extend(self._extract_app_events(
                    metric_dir, service_entities))

            if container_entities:
                events.extend(self._extract_container_events(
                    metric_dir, container_entities))

        return events

    def _extract_app_events(self, metric_dir: str,
                            entities: List[Entity]) -> List[ObservationEvent]:
        events: List[ObservationEvent] = []
        entity_ids = {e.entity_id for e in entities}

        for fname in ["metric_app.csv", "metric_service.csv"]:
            mfile = os.path.join(metric_dir, fname)
            if not os.path.exists(mfile):
                continue

            try:
                df = pd.read_csv(mfile, nrows=200000)
            except Exception:
                continue

            ts_col = next(
                (c for c in ["timestamp", "startTime", "time"] if c in df.columns), None)
            if ts_col is None:
                continue

            svc_col = next(
                (c for c in ["serviceName", "service", "tc", "app"]
                    if c in df.columns), None)
            if svc_col is None:
                continue

            if ts_col == "startTime":
                df["_ts"] = (df[ts_col].astype(float) / 1000).astype(int)
            else:
                df["_ts"] = df[ts_col].astype(int)

            for _, row in df.iterrows():
                svc = str(row[svc_col])
                if svc not in entity_ids:
                    continue
                ts = self._safe_int(row["_ts"])

                for raw_col, feat_name in self.APP_METRIC_COLS.items():
                    if raw_col in df.columns:
                        val = self._safe_float(row[raw_col])
                        events.append(ObservationEvent(
                            timestamp=ts,
                            entity_id=svc,
                            modality=Modality.METRIC,
                            feature_name=feat_name,
                            value=val,
                        ))
            break

        return events

    def _extract_container_events(self, metric_dir: str,
                                  entities: List[Entity]) -> List[ObservationEvent]:
        mfile = os.path.join(metric_dir, "metric_container.csv")
        if not os.path.exists(mfile):
            return []

        events: List[ObservationEvent] = []
        entity_ids = {e.entity_id for e in entities}

        try:
            for chunk in pd.read_csv(mfile, chunksize=500000):
                cols = list(chunk.columns)
                ts_col = next(
                    (c for c in cols if c.lower() == "timestamp"), cols[0])
                entity_col = next(
                    (c for c in cols if c.lower() in ("cmdb_id", "cmdb_id")), cols[1])
                kpi_col = next(
                    (c for c in cols if c.lower() in ("kpi_name", "name")), cols[2])
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
        except Exception:
            pass

        return events

    # ---- Relations ----

    def extract_relations(self, data_dir: str,
                          entities: List[Entity]) -> List[Relation]:
        relations: List[Relation] = []
        services = [e for e in entities if e.entity_type == EntityType.SERVICE]
        containers = [e for e in entities if e.entity_type == EntityType.CONTAINER]

        for cont in containers:
            cont_lower = cont.lower().replace(
                "_", "").replace("-", "").replace(" ", "")
            for svc in services:
                svc_lower = svc.lower().replace(
                    "_", "").replace("-", "").replace(" ", "")
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
                       entity_ids: Set[str]) -> Optional[str]:
        comp_lower = component.lower().replace(
            "_", "").replace("-", "").replace(" ", "")
        for eid in entity_ids:
            eid_lower = eid.lower().replace(
                "_", "").replace("-", "").replace(" ", "")
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
