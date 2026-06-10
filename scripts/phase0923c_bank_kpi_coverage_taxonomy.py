"""Phase 0.9.2.3c: Bank KPI Coverage Taxonomy Audit.

Comprehensive inventory and semantic classification of all raw KPI names
in the Bank dataset. Produces a catalog CSV and Markdown report identifying:
- Source family, entity scope, semantic family, metric kind, unit hint
- Existing bank_safe_v1 status (accepted/dropped/unmapped)
- High-confidence bank_safe_v2 candidates
- New-slot candidates
- Manual-review KPIs

Does NOT modify the adapter, model, or any production code.
"""

import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Chapter 1: Classifier Rules (ALL rules centralized here)
# ---------------------------------------------------------------------------

# ── Source family ──

def classify_source_family(kpi_name: str) -> str:
    """Classify which system component emitted this KPI."""
    n = kpi_name.lower()
    if n.startswith("container"):
        return "Container"
    if n.startswith("oslinux"):
        return "OSLinux"
    if n.startswith("tomcat"):
        return "Tomcat"
    if n.startswith("jvm"):
        return "JVM"
    if n.startswith("mysql"):
        return "MySQL"
    if n.startswith("redis"):
        return "Redis"
    return "Unknown"

# ── Entity scope ──

ENTITY_SCOPE_RULES: List[Tuple[str, str]] = [
    ("container", "container"),
    ("oslinux", "host"),
    ("tomcat", "application_server"),
    ("jvm", "jvm"),
    ("mysql", "database"),
    ("redis", "cache"),
]

def classify_entity_scope(kpi_name: str) -> str:
    n = kpi_name.lower()
    for prefix, scope in ENTITY_SCOPE_RULES:
        if n.startswith(prefix):
            return scope
    return "unknown"

# ── Semantic family ──

SEMANTIC_RULES: List[Tuple[str, str]] = [
    # CPU
    (r"cpu|cpuload|cpuutil|cpu_percent|cputime|cpuwio|cpuidle|singlecpu",
     "cpu"),
    # Memory
    (r"(?<![a-z])(mem(?:ory)?)(?![a-z])|mem_percent|memusage|mempercent|memlimit|heap",
     "memory"),
    # Network
    (r"network|net_|eth\d|ens\d|tcp_|networkrx|networktx|rx_bytes|tx_bytes"
     r"|incoming_network|outgoing_network|bandwidth|packets?",
     "network"),
    # Disk
    (r"disk_|localdisk|dsk\w*$|dsk\w+\b",
     "disk"),
    # Filesystem
    (r"filesystem|fsavailable|fscapacity|fsused|fsinode",
     "filesystem"),
    # Swap
    (r"\bswap\b|swptot|swp_",
     "swap"),
    # Process
    (r"\bproc(ess)?\b|proczombie|procppmem|procppcount|procppcpu",
     "process"),
    # Request/HTTP
    (r"request|http|errorcount|processingtime|maxtime",
     "request"),
    # Latency
    (r"latency|avg_time|mrt|response.?time|row.?lock.?time",
     "latency"),
    # Thread
    (r"thread",
     "thread"),
    # Session
    (r"session",
     "session"),
    # GC
    (r"\bfgc\b|full.?gc|garbage.?collect",
     "gc"),
    # Connection
    (r"connection|connected|aborted.*(client|connect)|threads.?connected"
     r"|threads.?running",
     "connection"),
    # Database IO
    (r"mysql|innodb|binlog|handler_|qcache|key_|sort_|table_|rows_"
     r"|select_|com_|created_tmp|opened_|open_|max_conn|queries|questions"
     r"|slow.*quer|trx|lock|bytes_received|bytes_sent",
     "database_io"),
    # Cache
    (r"redis|cache|aof_|rdb_|keyspace|lru",
     "cache"),
    # NTP/Time
    (r"ntp|time.?offset",
     "time_sync"),
    # Uptime
    (r"uptime",
     "uptime"),
    # System
    (r"system|hostname|default.?route|check-",
     "system"),
    # Zabbix monitoring
    (r"zabbix",
     "monitoring"),
]

def classify_semantic_family(kpi_name: str) -> str:
    n = kpi_name.lower().replace(" ", "_")
    for pattern, family in SEMANTIC_RULES:
        if re.search(pattern, n, re.I):
            return family
    return "unknown"

# ── Metric kind ──

METRIC_KIND_HINTS: List[Tuple[str, str]] = [
    (r"percent|perc\b|pct\b|ratio|cpuutil", "ratio"),
    (r"ops_per_sec|per_sec|bytes_per_sec|bandwidth|rate|b\s*ps", "rate"),
    (r"time|latency|duration|uptime|ms\b|seconds?|microsec", "duration"),
    (r"count|number|total|num|sum|hits|misses|requests|commands|queries"
     r"|connections|reads|writes|sessions|threads|processes|opened|created"
     r"|flushed|waits|wait_time|pending|active|busy", "counter"),
    (r"per_sec|_sec|_min", "rate"),
]

def classify_metric_kind(kpi_name: str) -> str:
    n = kpi_name.lower().replace(" ", "_")
    # Duration first (most specific)
    if re.search(r"time|latency|duration|uptime|ms\b|microsec", n, re.I):
        return "duration"
    # Rate
    if re.search(r"per_sec|_per_|_sec|bytes_|bandwidth|ops_|bps|throughput", n, re.I):
        return "rate"
    # Counter (before gauge to avoid 'total', 'cache' etc. false positives)
    if re.search(r"\bcount\b|number|num_of|sum|hits|misses|commands|connections"
                 r"|reads|writes|sessions|threads|processes|opened|created"
                 r"|flushed|waits|wait_time|pending|active|busy|deletes?|inserts?"
                 r"|updates?|rollback|savepoint|aborted|received|sent|connected"
                 r"|rejected|accepted", n, re.I):
        return "counter"
    # Ratio
    if re.search(r"percent|perc\b|pct\b|ratio|util\b|cpuutil|idleutil", n, re.I):
        return "ratio"
    # Gauge
    if re.search(r"\bmax\b|limit|size|cache\b|total|free|available|used\b|usage\b"
                 r"|memory|heap|pool|load|capacity", n, re.I):
        return "gauge"
    # Counter fallback (broader)
    if re.search(r"count|number|num|sum|total|hits|misses|commands|connections"
                 r"|reads|writes|sessions|threads|processes|opened|created"
                 r"|flushed|waits|pending|active|busy|deletes?|inserts?|updates?"
                 r"|rollback|savepoint|aborted|received|sent|connected"
                 r"|rejected|accepted", n, re.I):
        return "counter"
    return "unknown"

# ── Unit hint ──

def classify_unit_hint(kpi_name: str) -> str:
    n = kpi_name.lower()
    if re.search(r"percent|perc\b(?![a-z])|pct\b(?![a-z])|util\b|cpuutil|idleutil", n):
        return "percent"
    if re.search(r"ratio|fragmentation", n):
        return "ratio"
    if re.search(r"bytes|b\s*(ps|per.?sec)|networkrx|networktx|rx_bytes|tx_bytes", n):
        return "bytes"
    if re.search(r"b\s*ps|bytes_per_sec|k(b|B).*per.?sec|bandwidth", n):
        return "bytes_per_second"
    if re.search(r"milli?sec|m\s*sec|ms\b|latency|time|processing.?time|row.?lock.?time", n):
        return "milliseconds"
    if re.search(r"seconds|sec\b|uptime|interval", n):
        return "seconds"
    if re.search(r"count|number|num|total|connections|threads|sessions|processes"
                 r"|requests|reads|writes|hits|misses|commands|queries|operations"
                 r"|rows?|inserts?|deletes?|updates?|sort|lock|waits|wait_time"
                 r"|pending|active|busy|created|opened|flushed|aborted|accepted"
                 r"|rejected|received|sent|connected|running|cached|savepoint"
                 r"|rollback|commit|binlog|handler|qcache|key_", n):
        return "count"
    if re.search(r"per_sec|per.?sec|_rate\b|ops_|qps|tps|throughput", n):
        return "rate"
    return "unknown"

# ── Existing slot candidate ──

def classify_existing_slot_candidate(kpi_name: str, semantic: str,
                                      source: str, unit: str) -> Optional[str]:
    """Return existing canonical slot if high-confidence mapping exists."""
    n = kpi_name.lower()
    # cpu slot: only percent-based CPU from containers or OS
    if semantic == "cpu" and unit == "percent":
        return "cpu"
    # mem slot: only percent-based memory
    if semantic == "memory" and unit == "percent" and source != "JVM":
        return "mem"
    # net_rx: container rx bytes
    if semantic == "network" and re.search(r"networkrx|rx_bytes|incoming", n):
        return "net_rx"
    # net_tx: container tx bytes
    if semantic == "network" and re.search(r"networktx|tx_bytes|outgoing", n):
        return "net_tx"
    return None

# ── New slot candidate ──

def classify_new_slot_candidate(kpi_name: str, semantic: str,
                                 source: str, unit: str) -> Optional[str]:
    """If KPI has RCA value but can't fit existing slots, suggest a new one."""
    n = kpi_name.lower()
    # Database IO
    if semantic == "database_io" and source == "MySQL":
        return "mysql_io"
    # Disk IO
    if semantic == "disk":
        return "disk_io"
    # Tomcat requests (QPS, latency, errors)
    if semantic == "request" and source in ("Tomcat", "application_server"):
        return "tomcat_requests"
    # Tomcat sessions
    if semantic == "session":
        return "sessions"
    # Thread utilization
    if semantic == "thread":
        return "threads"
    # JVM GC
    if semantic == "gc":
        return "fgc"
    # JVM CPU
    if semantic == "cpu" and source in ("JVM", "jvm"):
        return "jvm_cpu"
    # JVM memory
    if semantic == "memory" and source in ("JVM", "jvm"):
        return "jvm_mem"
    # Filesystem capacity/usage
    if semantic == "filesystem":
        return "filesystem_usage"
    # Swap
    if semantic == "swap":
        return "swap_io"
    # DB latency
    if semantic == "latency" and source == "MySQL":
        return "db_latency"
    # Process health
    if semantic == "process":
        return "process_health"
    # Redis cache
    if semantic == "cache":
        return "cache_metrics"
    # Tomcat memory at process level
    if semantic == "memory" and source == "Tomcat":
        return "jvm_mem"
    # JVM thread count
    if semantic == "thread" and source == "JVM":
        return "threads"
    # Tomcat threads
    if semantic == "thread" and source == "Tomcat":
        return "threads"
    # OS network (not rx/tx bytes)
    if semantic == "network" and source == "OSLinux":
        return "os_network"
    # Tomcat latency
    if semantic == "latency" and source == "Tomcat":
        return "tomcat_requests"
    return None

# ── Transform hint ──

def classify_transform_hint(kpi_name: str, metric_kind: str,
                              unit: str, source: str) -> str:
    n = kpi_name.lower()
    if unit == "bytes" and (
        "networkrx" in n or "networktx" in n or "rx_bytes" in n
        or "tx_bytes" in n):
        return "log1p"
    if metric_kind == "counter":
        if re.search(r"bytes|network", n):
            return "delta_then_log1p"
        return "delta_then_log1p"
    if unit == "percent" and re.search(r"percent|perc|pct", n):
        return "identity"
    if metric_kind == "rate":
        return "log1p"
    if unit == "count":
        return "log1p"
    return "unknown"

# ── Recommended action ──

def classify_recommended_action(
    kpi_name: str,
    existing_slot: Optional[str],
    new_slot: Optional[str],
    source: str,
    semantic: str,
    metric_kind: str,
    unit: str,
    row_count: int,
    unique_components: int,
) -> str:
    n = kpi_name.lower()

    # High-confidence existing slot mapping
    if existing_slot is not None:
        return "map_safe_v2"

    # Valid new-slot candidates with sufficient data
    if new_slot is not None and row_count >= 100:
        return "requires_new_slot"

    # Zero-value or all-identical metrics
    # (We can't tell from name alone; handled via defer)
    if unit == "percent" and source == "Container" and new_slot is None:
        return "defer_unit_check"

    # Unknown semantics or unit
    if semantic == "unknown" or unit == "unknown":
        return "manual_review"

    # Low-value named metrics
    if metric_kind in ("duration",) and source == "JVM":
        return "manual_review"

    # Everything else that has a new slot candidate but low row count
    if new_slot is not None:
        return "requires_new_slot"

    return "manual_review"

# ── Confidence ──

def classify_confidence(kpi_name: str, recommended_action: str,
                         existing_slot: Optional[str],
                         unit: str, semantic: str) -> str:
    if existing_slot is not None and unit != "unknown":
        return "high"
    if recommended_action == "map_safe_v2":
        return "high"
    if recommended_action == "requires_new_slot" and semantic != "unknown":
        return "medium"
    if recommended_action == "defer_unit_check":
        return "low"
    return "low"

# ── Reason ──

def build_reason(kpi_name: str, existing_slot: Optional[str],
                  new_slot: Optional[str], unit: str,
                  metric_kind: str, source: str, semantic: str,
                  recommended_action: str) -> str:
    if existing_slot:
        return (f"Maps to existing slot '{existing_slot}'; "
                f"semantic={semantic}, unit={unit}, source={source}")
    if new_slot:
        return (f"Requires new slot '{new_slot}'; "
                f"semantic={semantic}, unit={unit}, source={source}")
    if recommended_action == "defer_unit_check":
        return f"Percent-based {source} KPI; confirm 0-100% vs 0-1 range"
    return (f"semantic={semantic}, unit={unit}, kind={metric_kind}; "
            f"cannot auto-classify")

# ── Full KPI classifier ──

def classify_kpi(kpi_name: str, row_count: int,
                  unique_components: int) -> Dict:
    source_family = classify_source_family(kpi_name)
    entity_scope = classify_entity_scope(kpi_name)
    semantic_family = classify_semantic_family(kpi_name)
    metric_kind = classify_metric_kind(kpi_name)
    unit_hint = classify_unit_hint(kpi_name)
    existing_slot = classify_existing_slot_candidate(
        kpi_name, semantic_family, source_family, unit_hint)
    new_slot = classify_new_slot_candidate(
        kpi_name, semantic_family, source_family, unit_hint)
    recommended_action = classify_recommended_action(
        kpi_name, existing_slot, new_slot, source_family,
        semantic_family, metric_kind, unit_hint,
        row_count, unique_components)
    confidence = classify_confidence(
        kpi_name, recommended_action, existing_slot,
        unit_hint, semantic_family)
    transform_hint = classify_transform_hint(
        kpi_name, metric_kind, unit_hint, source_family)
    reason = build_reason(
        kpi_name, existing_slot, new_slot,
        unit=unit_hint, metric_kind=metric_kind,
        source=source_family, semantic=semantic_family,
        recommended_action=recommended_action)

    return {
        "raw_kpi_name": kpi_name,
        "row_count": row_count,
        "row_ratio": 0.0,  # filled later
        "cumulative_row_ratio": 0.0,  # filled later
        "unique_component_count": unique_components,
        "source_family": source_family,
        "entity_scope": entity_scope,
        "semantic_family": semantic_family,
        "metric_kind": metric_kind,
        "unit_hint": unit_hint,
        "candidate_existing_slot": existing_slot or "",
        "candidate_new_slot": new_slot or "",
        "recommended_action": recommended_action,
        "confidence": confidence,
        "transform_hint": transform_hint,
        "reason": reason,
    }

# ── Existing adapter mapping (matching phase0923b logic) ──

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

SAFE_V1_KEPT = {"cpu", "mem", "net_rx", "net_tx"}

def get_bank_safe_v1_status(kpi_name: str) -> str:
    for cat, pat in CONTAINER_KPI_PATTERNS:
        if pat.match(str(kpi_name)):
            if cat in SAFE_V1_KEPT:
                return "accepted"
            return "dropped"
    return "unmapped"

# ---------------------------------------------------------------------------
# Chapter 2: Data loading
# ---------------------------------------------------------------------------

def load_all_kpi_stats(data_dir: str, max_days: int = 99) -> Dict:
    """Load all metric_container.csv, return per-KPI row counts and components."""
    telemetry_dir = os.path.join(data_dir, "telemetry")
    days = sorted([d for d in os.listdir(telemetry_dir) if d.startswith("20")])[:max_days]

    kpi_counts: Dict[str, int] = defaultdict(int)
    kpi_components: Dict[str, Set[str]] = defaultdict(set)

    for day in days:
        mfile = os.path.join(telemetry_dir, day, "metric", "metric_container.csv")
        if not os.path.exists(mfile):
            continue
        try:
            for chunk in pd.read_csv(mfile, chunksize=200000):
                cols = list(chunk.columns)
                kpi_col = next((c for c in cols if c.lower() in ("kpi_name", "name")), cols[2])
                entity_col = next((c for c in cols if c.lower() in ("cmdb_id", "cmdb_id")), cols[1])
                for _, row in chunk.iterrows():
                    kpi = str(row[kpi_col])
                    cmdb = str(row[entity_col])
                    kpi_counts[kpi] += 1
                    kpi_components[kpi].add(cmdb)
        except Exception:
            continue

    return {
        "kpi_counts": dict(kpi_counts),
        "kpi_components": {k: len(v) for k, v in kpi_components.items()},
        "total_rows": sum(kpi_counts.values()),
        "unique_kpis": len(kpi_counts),
    }

# ---------------------------------------------------------------------------
# Chapter 3: Main pipeline
# ---------------------------------------------------------------------------

def build_catalog(stats: Dict) -> List[Dict]:
    total = stats["total_rows"]
    catalog = []
    for kpi_name, row_count in sorted(stats["kpi_counts"].items(),
                                       key=lambda x: -x[1]):
        n_comps = stats["kpi_components"].get(kpi_name, 0)
        row = classify_kpi(kpi_name, row_count, n_comps)
        row["existing_bank_safe_v1_status"] = get_bank_safe_v1_status(kpi_name)
        catalog.append(row)

    # Compute ratios
    cum = 0
    for i, row in enumerate(catalog):
        row["row_ratio"] = row["row_count"] / total
        cum += row["row_ratio"]
        row["cumulative_row_ratio"] = cum

    return catalog

# ── Report helpers ──

STATUS_ORDER = ["accepted", "dropped", "unmapped"]

def generate_report(catalog: List[Dict], total_rows: int, output_path: str):
    lines = []

    def w(line=""):
        lines.append(str(line))

    # ── Compute aggregates ──
    by_status: Dict[str, Dict] = {"accepted": {"rows": 0, "kpis": set()},
                                    "dropped": {"rows": 0, "kpis": set()},
                                    "unmapped": {"rows": 0, "kpis": set()}}
    by_source: Dict[str, int] = defaultdict(int)
    by_semantic: Dict[str, int] = defaultdict(int)

    for r in catalog:
        st = r["existing_bank_safe_v1_status"]
        by_status[st]["rows"] += r["row_count"]
        by_status[st]["kpis"].add(r["raw_kpi_name"])
        by_source[r["source_family"]] += r["row_count"]
        by_semantic[r["semantic_family"]] += r["row_count"]

    # Top KPI coverage
    cov_10 = sum(r["row_ratio"] for r in catalog[:10])
    cov_25 = sum(r["row_ratio"] for r in catalog[:25])
    cov_50 = sum(r["row_ratio"] for r in catalog[:50])
    cov_100 = sum(r["row_ratio"] for r in catalog[:100])

    # high-confidence v2 candidates
    v2_candidates = [r for r in catalog if r["recommended_action"] == "map_safe_v2"]
    v2_new_rows = sum(r["row_count"] for r in v2_candidates)

    # ── Report ──

    w("# Phase 0.9.2.3c — Bank KPI Coverage Taxonomy Audit Report")
    w()
    w("## 1. Executive Summary")
    w()
    w(f"- **Total raw records**: {total_rows:,}")
    w(f"- **Unique KPI names**: {len(catalog)}")
    w()
    w("### Acceptance Status Breakdown")
    w()
    w("| Status | Records | Record % | Unique KPIs |")
    w("|---|---|---|---|")
    for st in STATUS_ORDER:
        d = by_status[st]
        w(f"| {st} | {d['rows']:,} | "
          f"{d['rows']/total_rows*100:.2f}% | {len(d['kpis'])} |")
    w()

    w("### Source Family Breakdown")
    w()
    w("| Source Family | Records | Record % |")
    w("|---|---|---|")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        w(f"| {src} | {cnt:,} | {cnt/total_rows*100:.2f}% |")
    w()

    w("### Semantic Family Breakdown")
    w()
    w("| Semantic Family | Records | Record % |")
    w("|---|---|---|")
    for sem, cnt in sorted(by_semantic.items(), key=lambda x: -x[1]):
        w(f"| {sem} | {cnt:,} | {cnt/total_rows*100:.2f}% |")
    w()

    w("### Top KPI Coverage Concentration")
    w()
    w(f"- **Top 10 KPIs**: {cov_10*100:.1f}% of all records")
    w(f"- **Top 25 KPIs**: {cov_25*100:.1f}%")
    w(f"- **Top 50 KPIs**: {cov_50*100:.1f}%")
    w(f"- **Top 100 KPIs**: {cov_100*100:.1f}%")
    w()

    unmapped = [r for r in catalog if r["existing_bank_safe_v1_status"] == "unmapped"]

    w("## 2. Top Unmapped KPIs")
    w()
    w("(Top 50 by row count)")
    w()
    w("| # | Raw KPI Name | Rows | Row% | Cum% | Source | Semantic | "
      "Kind | Unit | Action | Confidence |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(unmapped[:50], 1):
        w(f"| {i} | `{r['raw_kpi_name'][:80]}` | {r['row_count']:,} | "
          f"{r['row_ratio']*100:.2f}% | {r['cumulative_row_ratio']*100:.1f}% | "
          f"{r['source_family']} | {r['semantic_family']} | {r['metric_kind']} | "
          f"{r['unit_hint']} | {r['recommended_action']} | {r['confidence']} |")
    w()

    w("## 3. High-Confidence `bank_safe_v2` Candidates")
    w()
    if not v2_candidates:
        w("No high-confidence candidates identified.")
        w()
    else:
        w("### Candidate KPIs")
        w()
        w("| # | Raw KPI Name | Target Slot | Entity | Kind | Unit | "
          "Transform | New Rows | New Cov% | Reason |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(v2_candidates, 1):
            w(f"| {i} | `{r['raw_kpi_name'][:60]}` | "
              f"`{r['candidate_existing_slot']}` | "
              f"{r['entity_scope']} | {r['metric_kind']} | {r['unit_hint']} | "
              f"{r['transform_hint']} | {r['row_count']:,} | "
              f"{r['row_ratio']*100:.3f}% | {r['reason']} |")
        w()

        w("### Coverage Impact")
        w()
        accepted_rows = by_status["accepted"]["rows"]
        current_cov = accepted_rows / total_rows * 100
        new_cov = (accepted_rows + v2_new_rows) / total_rows * 100
        w(f"- **Current accepted coverage**: {current_cov:.2f}% ({accepted_rows:,} rows)")
        w(f"- **After high-confidence v2 candidates**: {new_cov:.2f}% "
          f"({accepted_rows + v2_new_rows:,} rows)")
        w(f"- **Coverage gain**: +{(new_cov - current_cov):.2f}% "
          f"(+{v2_new_rows:,} rows)")
        w()

    w("## 4. New-Slot Candidates")
    w()
    new_slot_kpis = [r for r in catalog if r["recommended_action"] == "requires_new_slot"]
    if not new_slot_kpis:
        w("No new-slot candidates identified.")
        w()
    else:
        w("### By proposed new slot")
        w()
        by_new_slot: Dict[str, List] = defaultdict(list)
        for r in new_slot_kpis:
            by_new_slot[r["candidate_new_slot"]].append(r)

        for slot, kpis in sorted(by_new_slot.items()):
            total_r = sum(r["row_count"] for r in kpis)
            w(f"**`{slot}`** ({len(kpis)} KPIs, {total_r:,} rows, "
              f"{total_r/total_rows*100:.2f}% coverage)")
            for r in kpis[:5]:
                w(f"  - `{r['raw_kpi_name'][:70]}` ({r['row_count']:,} rows, "
                  f"{r['source_family']}, {r['semantic_family']}, "
                  f"{r['metric_kind']}, {r['unit_hint']})")
            if len(kpis) > 5:
                w(f"  - ... and {len(kpis)-5} more KPIs")
            w()

    w("## 5. Manual-Review List")
    w()
    manual = [r for r in catalog if r["recommended_action"] == "manual_review"]
    if not manual:
        w("No KPIs require manual review.")
        w()
    else:
        w(f"{len(manual)} KPIs require manual review:")
        w()
        w("| # | Raw KPI Name | Rows | Source | Semantic | Kind | Unit |")
        w("|---|---|---|---|---|---|---|")
        for i, r in enumerate(manual[:50], 1):
            w(f"| {i} | `{r['raw_kpi_name'][:70]}` | {r['row_count']:,} | "
              f"{r['source_family']} | {r['semantic_family']} | "
              f"{r['metric_kind']} | {r['unit_hint']} |")
        if len(manual) > 50:
            w(f"*(... and {len(manual)-50} more KPIs)*")
        w()

    w("## 6. Full Action Summary")
    w()
    w("| Recommended Action | KPI Count | Total Rows | Row % |")
    w("|---|---|---|---|")
    by_action: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "rows": 0})
    for r in catalog:
        a = r["recommended_action"]
        by_action[a]["count"] += 1
        by_action[a]["rows"] += r["row_count"]
    for a in ["map_safe_v2", "requires_new_slot", "defer_unit_check",
              "defer_zero_semantics_check", "drop_low_value", "manual_review"]:
        d = by_action.get(a)
        if d and d["count"] > 0:
            w(f"| {a} | {d['count']} | {d['rows']:,} | "
              f"{d['rows']/total_rows*100:.2f}% |")
    w()

    w("## 7. Recommendations for Phase 0.9.2.3d")
    w()
    w("### Minimal `bank_safe_v2` implementation")
    w()
    if v2_candidates:
        w("The following high-confidence KPIs should be added to `bank_safe_v2`:")
        w()
        for r in v2_candidates:
            w(f"- `{r['raw_kpi_name']}` → slot `{r['candidate_existing_slot']}` "
              f"({r['transform_hint']}, {r['confidence']} confidence)")
        w()
    else:
        w("No high-confidence KPIs identified for `bank_safe_v2`.")
        w()

    w("### Implementation steps for Phase 0.9.2.3d")
    w()
    w("1. Add the above KPIs (or their regex families) to `bank_safe_v2` in the adapter")
    w("2. Add corresponding transform rules (identity/log1p/delta_then_log1p)")
    w("3. Do NOT add new slots — keep to existing 4-slot structure")
    w("4. Re-run Phase 0.9.2.3b scale audit on the expanded `bank_safe_v2` output")
    w("5. Defer unit normalization, new slots, and per-container sub-slotting")
    w()
    w("### Deferred to later phases")
    w()
    w("- New-slot candidates (Phase 0.9.2.4): mysql_io, disk_io, tomcat_requests, "
      "sessions, threads, cache_metrics, filesystem_usage, process_health, swap_io")
    w("- Unit normalization for mixed-scale slots (Phase 0.9.2.5)")
    w("- Per-container sparse-signal treatment (Phase 0.9.2.6)")
    w("- 77.4% unmapped OSLinux metrics: comprehensive regex rewrite (Phase 0.9.3)")
    w()

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {output_path}")


def write_catalog_csv(catalog: List[Dict], output_path: str):
    fieldnames = [
        "raw_kpi_name", "row_count", "row_ratio",
        "cumulative_row_ratio", "unique_component_count",
        "existing_bank_safe_v1_status",
        "source_family", "entity_scope", "semantic_family",
        "metric_kind", "unit_hint",
        "candidate_existing_slot", "candidate_new_slot",
        "recommended_action", "confidence", "transform_hint", "reason",
    ]
    with open(output_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in catalog:
            writer.writerow(row)
    print(f"Catalog CSV written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 0.9.2.3c: Bank KPI Coverage Taxonomy Audit")
    parser.add_argument("--data-dir", type=str,
                        default="/home/dell2/RCA513/yyx/OpenRCA/Bank/Bank")
    parser.add_argument("--output-report", type=str,
                        default="docs/phase0923c_bank_kpi_coverage_taxonomy.md")
    parser.add_argument("--output-csv", type=str,
                        default="artifacts/phase0923c_bank_kpi_catalog.csv")
    parser.add_argument("--max-days", type=int, default=99)
    args = parser.parse_args()

    print(f"Loading Bank KPIs from {args.data_dir} ...")
    t0 = time.time()
    stats = load_all_kpi_stats(args.data_dir, max_days=args.max_days)
    dt = time.time() - t0
    print(f"Loaded {stats['total_rows']:,} rows, "
          f"{stats['unique_kpis']} unique KPIs in {dt:.1f}s")

    t0 = time.time()
    catalog = build_catalog(stats)
    print(f"Classification complete in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    generate_report(catalog, stats["total_rows"], args.output_report)
    write_catalog_csv(catalog, args.output_csv)

    # Quick summary
    by_action = defaultdict(int)
    for r in catalog:
        by_action[r["recommended_action"]] += 1
    print(f"  map_safe_v2: {by_action['map_safe_v2']} KPIs")
    print(f"  requires_new_slot: {by_action['requires_new_slot']} KPIs")
    print(f"  defer_unit_check: {by_action['defer_unit_check']} KPIs")
    print(f"  manual_review: {by_action['manual_review']} KPIs")


if __name__ == "__main__":
    main()
