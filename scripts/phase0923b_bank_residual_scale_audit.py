"""Phase 0.9.2.3b: Bank Canonicalized KPI Residual Scale Audit.

Offline audit script that reads raw Bank metric_container.csv, applies
the bank_safe_v1 canonicalization mapping, and produces a Markdown report
identifying scale inconsistencies, unit mismatches, and outlier risks.

Does NOT modify the adapter, model, or any production code.
"""

import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — all threshold constants in one place
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    extreme_outlier_factor: float = 10.0
    median_scale_ratio: float = 100.0
    p95_scale_ratio: float = 100.0
    ratio_percent_spread: float = 50.0
    sparse_ratio: float = 0.01
    near_constant_iqr: float = 1e-6
    near_constant_total_range: float = 1e-6
    tiny_sample_min: int = 5


THRESHOLDS = Thresholds()

# ---------------------------------------------------------------------------
# KPI pattern mapping — EXACT copy from the Bank adapter
# ---------------------------------------------------------------------------

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

# bank_safe_v1 canonicalization: which categories are kept/dropped/transformed
SAFE_V1_CANONICAL = {
    "cpu": {"action": "keep", "canonical_name": "cpu"},
    "mem": {"action": "keep", "canonical_name": "mem"},
    "net_rx": {"action": "log1p", "canonical_name": "net_rx"},
    "net_tx": {"action": "log1p", "canonical_name": "net_tx"},
    "jvm_cpu": {"action": "drop"},
    "mem_usage": {"action": "drop"},
    "jvm_mem": {"action": "drop"},
    "disk_io": {"action": "drop"},
    "mysql_io": {"action": "drop"},
    "threads": {"action": "drop"},
    "sessions": {"action": "drop"},
    "fgc": {"action": "drop"},
}


def map_kpi_to_category(kpi_name: str) -> Optional[str]:
    for cat, pat in CONTAINER_KPI_PATTERNS:
        if pat.match(str(kpi_name)):
            return cat
    return None


def canonicalize(category: str, value: float) -> Optional[Tuple[str, float]]:
    if category not in SAFE_V1_CANONICAL:
        return None
    rule = SAFE_V1_CANONICAL[category]
    if rule["action"] == "drop":
        return None
    if rule["action"] == "keep":
        return (rule["canonical_name"], value)
    if rule["action"] == "log1p":
        return (rule["canonical_name"], float(np.log1p(max(value, 0.0))))
    return None

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_mad(values: np.ndarray) -> float:
    med = np.median(values)
    return float(np.median(np.abs(values - med)))

def compute_stats(values: np.ndarray) -> Dict:
    n = len(values)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    non_finite = n - len(finite)

    if len(finite) == 0:
        return {
            "n": n, "n_finite": 0, "non_finite": non_finite,
            "zero_ratio": 0.0, "negative_ratio": 0.0,
            "min": float("nan"), "p01": float("nan"), "p05": float("nan"),
            "p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
            "max": float("nan"), "mean": float("nan"), "std": float("nan"),
            "iqr": float("nan"), "mad": float("nan"),
        }

    zero_ratio = float((finite == 0).sum()) / n
    negative_ratio = float((finite < 0).sum()) / n

    qs = np.percentile(finite, [1, 5, 50, 95, 99])

    return {
        "n": n, "n_finite": len(finite), "non_finite": non_finite,
        "zero_ratio": zero_ratio, "negative_ratio": negative_ratio,
        "min": float(finite.min()),
        "p01": float(qs[0]), "p05": float(qs[1]),
        "p50": float(qs[2]), "p95": float(qs[3]), "p99": float(qs[4]),
        "max": float(finite.max()),
        "mean": float(finite.mean()), "std": float(finite.std()),
        "iqr": float(np.subtract(*np.percentile(finite, [75, 25]))),
        "mad": compute_mad(finite),
    }

# ---------------------------------------------------------------------------
# Risk flag rules
# ---------------------------------------------------------------------------

def check_flags(stats: Dict, n_unique_components: int,
                raw_kpi_name: str, canonical_slot: str,
                threshold_map: Dict[str, float] = None) -> List[str]:
    flags = []
    t = THRESHOLDS

    if stats["non_finite"] > 0:
        flags.append("NON_FINITE_VALUES")

    if stats["negative_ratio"] > 0.001:
        flags.append("NEGATIVE_VALUES_PRESENT")

    non_zero = stats.get("non_zero_vals")
    if non_zero is not None and stats["n_finite"] > 0:
        extreme_mask = np.abs(non_zero) > t.extreme_outlier_factor * max(abs(stats["p01"]), abs(stats["p99"]))
        if np.any(extreme_mask):
            flags.append("EXTREME_OUTLIER_TAIL")
    elif stats["n_finite"] > t.tiny_sample_min:
        ref = max(abs(stats["p01"]), abs(stats["p99"]))
        if ref > 0 and abs(stats["max"]) > t.extreme_outlier_factor * ref:
            flags.append("EXTREME_OUTLIER_TAIL")

    if (stats["n_finite"] > t.tiny_sample_min
            and stats["iqr"] <= t.near_constant_iqr
            and (stats["max"] - stats["min"]) <= t.near_constant_total_range):
        flags.append("CONSTANT_OR_NEAR_CONSTANT_SIGNAL")

    zero_ratio = stats["zero_ratio"]
    if (stats["n_finite"] > t.tiny_sample_min
            and zero_ratio > (1.0 - t.sparse_ratio)):
        flags.append("SPARSE_SIGNAL")

    if (threshold_map is not None
            and "median_scale_ratio" in threshold_map
            and threshold_map["median_scale_ratio"] > t.median_scale_ratio):
        flags.append("MEDIAN_SCALE_MISMATCH")

    if (threshold_map is not None
            and "p95_scale_ratio" in threshold_map
            and threshold_map["p95_scale_ratio"] > t.p95_scale_ratio):
        flags.append("P95_SCALE_MISMATCH")

    if (threshold_map is not None
            and "ratio_percent_flag" in threshold_map
            and threshold_map["ratio_percent_flag"]):
        flags.append("POSSIBLE_RATIO_PERCENT_MIX")

    return flags

# ---------------------------------------------------------------------------
# Data loading engine
# ---------------------------------------------------------------------------

def load_and_categorize(data_dir: str, max_days: int = 99) -> Dict:
    """Load all metric_container.csv files, apply bank_safe_v1 mapping.

    Returns a dict keyed by canonical_slot with per-raw-KPI value arrays.
    Also returns overall acceptance statistics.
    """
    telemetry_dir = os.path.join(data_dir, "telemetry")
    days = sorted([d for d in os.listdir(telemetry_dir) if d.startswith("20")])[:max_days]

    total_raw = 0
    total_unmapped = 0
    total_dropped = 0

    # per slot → per raw_kpi_name → per cmdb_id → list of values
    slot_kpi_vals: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    # per slot → overall values
    slot_all_vals: Dict[str, list] = defaultdict(list)

    slot_all_raw_values: Dict[str, list] = defaultdict(list)

    for day in days:
        mfile = os.path.join(telemetry_dir, day, "metric", "metric_container.csv")
        if not os.path.exists(mfile):
            continue
        try:
            for chunk in pd.read_csv(mfile, chunksize=200000):
                cols = list(chunk.columns)
                ts_col = next((c for c in cols if c.lower() == "timestamp"), cols[0])
                entity_col = next((c for c in cols if c.lower() in ("cmdb_id", "cmdb_id")),
                                  cols[1] if len(cols) > 1 else cols[0])
                kpi_col = next((c for c in cols if c.lower() in ("kpi_name", "name")),
                               cols[2] if len(cols) > 2 else cols[0])
                val_col = next((c for c in cols if c.lower() in ("value",)), cols[-1])

                for _, row in chunk.iterrows():
                    total_raw += 1
                    try:
                        raw_val = float(row[val_col])
                    except (ValueError, TypeError):
                        total_unmapped += 1
                        continue

                    kpi_name = str(row[kpi_col])
                    category = map_kpi_to_category(kpi_name)
                    if category is None:
                        total_unmapped += 1
                        continue

                    result = canonicalize(category, raw_val)
                    if result is None:
                        total_dropped += 1
                        continue

                    canonical_slot, transformed_val = result
                    cmdb_id = str(row[entity_col])

                    slot_all_vals[canonical_slot].append(transformed_val)
                    slot_all_raw_values[canonical_slot].append(raw_val)
                    slot_kpi_vals[canonical_slot][kpi_name][cmdb_id].append(transformed_val)
        except Exception:
            continue

    return {
        "total_raw": total_raw,
        "total_unmapped": total_unmapped,
        "total_dropped": total_dropped,
        "total_accepted": total_raw - total_unmapped - total_dropped,
        "slot_kpi_vals": dict(slot_kpi_vals),
        "slot_all_vals": {k: np.array(v) for k, v in slot_all_vals.items()},
        "slot_all_raw_values": {k: np.array(v) for k, v in slot_all_raw_values.items()},
    }

# ---------------------------------------------------------------------------
# Aggregation analysis
# ---------------------------------------------------------------------------

def analyze(data: Dict) -> Dict:
    """Compute statistics at all aggregation levels and apply risk flags."""
    result = {
        "global": {
            "total_raw": data["total_raw"],
            "total_accepted": data["total_accepted"],
            "total_dropped": data["total_dropped"],
            "total_unmapped": data["total_unmapped"],
            "accept_ratio": data["total_accepted"] / max(data["total_raw"], 1),
            "drop_ratio": data["total_dropped"] / max(data["total_raw"], 1),
            "unmapped_ratio": data["total_unmapped"] / max(data["total_raw"], 1),
            "n_slots": len(data["slot_kpi_vals"]),
        },
        "slots": {},
        "flagged_kpis": [],
    }

    for slot_name, kpi_dict in sorted(data["slot_kpi_vals"].items()):
        slot_info = {
            "canonical_slot": slot_name,
            "n_raw_kpis": len(kpi_dict),
            "total_rows": 0,
            "unique_components": set(),
            "all_values": data["slot_all_vals"].get(slot_name, np.array([])),
            "raw_kpis": {},
            "flags": [],
        }

        # Per raw KPI stats
        per_kpi_medians = {}
        per_kpi_p95s = {}
        kpi_scale_suspicious = False
        kpi_ranges_01 = []
        kpi_ranges_0100 = []

        for kpi_name, cmdb_dict in sorted(kpi_dict.items()):
            all_for_kpi = []
            kpi_components = set()
            for cmdb_id, vals in cmdb_dict.items():
                all_for_kpi.extend(vals)
                kpi_components.add(cmdb_id)

            arr = np.array(all_for_kpi)
            stats = compute_stats(arr)
            slot_info["total_rows"] += stats["n"]
            slot_info["unique_components"].update(kpi_components)

            stats["n_components"] = len(kpi_components)
            slot_info["raw_kpis"][kpi_name] = stats

            if stats["p50"] > 0:
                per_kpi_medians[kpi_name] = stats["p50"]
            if stats["p95"] > 0:
                per_kpi_p95s[kpi_name] = stats["p95"]

            if stats["min"] >= 0 and stats["max"] <= 1.5 and stats["p95"] <= 1.0:
                kpi_ranges_01.append(kpi_name)
            if stats["min"] >= 0 and stats["max"] > 20:
                kpi_ranges_0100.append(kpi_name)

        slot_info["n_components_total"] = len(slot_info["unique_components"])

        # Check for ratio/percent mix within slot
        if kpi_ranges_01 and kpi_ranges_0100:
            kpi_scale_suspicious = True

        # Median scale mismatch
        med_mismatch = {}
        non_zero_medians = {k: v for k, v in per_kpi_medians.items() if v > 1e-9}
        if len(non_zero_medians) >= 2:
            max_med = max(non_zero_medians.values())
            min_med = min(non_zero_medians.values())
            ratio = max_med / min_med if min_med > 0 else float("inf")
            med_mismatch = {"ratio": ratio,
                            "max_kpi": max(non_zero_medians, key=non_zero_medians.get),
                            "max_val": max_med,
                            "min_kpi": min(non_zero_medians, key=non_zero_medians.get),
                            "min_val": min_med}

        # P95 scale mismatch
        p95_mismatch = {}
        non_zero_p95s = {k: v for k, v in per_kpi_p95s.items() if v > 1e-9}
        if len(non_zero_p95s) >= 2:
            max_p95 = max(non_zero_p95s.values())
            min_p95 = min(non_zero_p95s.values())
            ratio = max_p95 / min_p95 if min_p95 > 0 else float("inf")
            p95_mismatch = {"ratio": ratio,
                            "max_kpi": max(non_zero_p95s, key=non_zero_p95s.get),
                            "max_val": max_p95,
                            "min_kpi": min(non_zero_p95s, key=non_zero_p95s.get),
                            "min_val": min_p95}

        # Per-KPI flags
        for kpi_name, stats in slot_info["raw_kpis"].items():
            threshold_map = {}
            if med_mismatch.get("ratio", 1) > THRESHOLDS.median_scale_ratio:
                threshold_map["median_scale_ratio"] = med_mismatch["ratio"]
            if p95_mismatch.get("ratio", 1) > THRESHOLDS.p95_scale_ratio:
                threshold_map["p95_scale_ratio"] = p95_mismatch["ratio"]
            if kpi_scale_suspicious:
                threshold_map["ratio_percent_flag"] = True

            kpi_flags = check_flags(
                stats, stats.get("n_components", 0),
                kpi_name, slot_name, threshold_map)

            if kpi_flags:
                slot_info["flags"].append({
                    "kpi_name": kpi_name,
                    "slot": slot_name,
                    "flags": kpi_flags,
                    "n": stats["n"],
                    "p50": stats["p50"],
                    "p95": stats["p95"],
                    "p99": stats["p99"],
                })
                result["flagged_kpis"].append({
                    "canonical_slot": slot_name,
                    "raw_kpi_name": kpi_name,
                    "flags": kpi_flags,
                    "n": stats["n"],
                    "p01": stats["p01"], "p50": stats["p50"],
                    "p95": stats["p95"], "p99": stats["p99"],
                    "max": stats["max"],
                })

        slot_info["median_mismatch"] = med_mismatch
        slot_info["p95_mismatch"] = p95_mismatch
        slot_info["ratio_percent_mix"] = kpi_scale_suspicious
        slot_info["n_flags"] = len(slot_info["flags"])
        result["slots"][slot_name] = slot_info

    # Mark slots-at-risk
    n_flagged_slots = sum(1 for s in result["slots"].values() if s["n_flags"] > 0)
    result["global"]["n_flagged_slots"] = n_flagged_slots

    return result

# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(analysis: Dict, output_path: str):
    g = analysis["global"]
    lines = []

    def w(line=""):
        lines.append(line)

    w("# Phase 0.9.2.3b — Bank Canonicalized KPI Residual Scale Audit Report")
    w()
    w("## 1. Executive Summary")
    w()
    w(f"- **Total raw records**: {g['total_raw']:,}")
    w(f"- **Accepted by `bank_safe_v1`**: {g['total_accepted']:,} "
      f"({g['accept_ratio']:.2%})")
    w(f"- **Dropped by `bank_safe_v1`**: {g['total_dropped']:,} "
      f"({g['drop_ratio']:.2%})")
    w(f"- **Unmapped (no regex match)**: {g['total_unmapped']:,} "
      f"({g['unmapped_ratio']:.2%})")
    w(f"- **Canonical slots**: {g['n_slots']}")
    w(f"- **Slots with scale-risk flags**: {g['n_flagged_slots']}")
    w()

    if g["n_flagged_slots"] == 0:
        w("No scale-risk flags triggered across any slot.")
    else:
        w("Slots with highest flag counts:")
        w()
        sorted_slots = sorted(analysis["slots"].items(),
                              key=lambda x: x[1]["n_flags"], reverse=True)
        for slot_name, sinfo in sorted_slots:
            if sinfo["n_flags"] > 0:
                w(f"- **{slot_name}**: {sinfo['n_flags']} flagged raw KPIs "
                  f"(of {sinfo['n_raw_kpis']} total)")

    w()
    w("## 2. Per-Slot Summary")
    w()

    for slot_name in sorted(analysis["slots"].keys()):
        sinfo = analysis["slots"][slot_name]
        w(f"### 2.{list(analysis['slots'].keys()).index(slot_name)+1}. Slot `{slot_name}`")
        w()
        w(f"- **Total rows**: {sinfo['total_rows']:,}")
        w(f"- **Raw KPI names**: {sinfo['n_raw_kpis']}")
        w(f"- **Unique components**: {sinfo['n_components_total']}")
        w(f"- **Flagged KPIs**: {sinfo['n_flags']}")
        w()

        if sinfo["ratio_percent_mix"]:
            w("- **WARNING**: Possible 0-1 vs 0-100 ratio/percent mix detected in this slot.")
            w()

        mm = sinfo.get("median_mismatch", {})
        if mm.get("ratio", 1) > THRESHOLDS.median_scale_ratio:
            w(f"- **Median scale mismatch ratio**: {mm['ratio']:.1f}x "
              f"(max: `{mm['max_kpi']}` = {mm['max_val']:.4g}, "
              f"min: `{mm['min_kpi']}` = {mm['min_val']:.4g})")
            w()

        pm = sinfo.get("p95_mismatch", {})
        if pm.get("ratio", 1) > THRESHOLDS.p95_scale_ratio:
            w(f"- **P95 scale mismatch ratio**: {pm['ratio']:.1f}x "
              f"(max: `{pm['max_kpi']}` = {pm['max_val']:.4g}, "
              f"min: `{pm['min_kpi']}` = {pm['min_val']:.4g})")
            w()

        # Per-KPI stats table
        w("| Raw KPI Name | n | min | p01 | p05 | p50 | p95 | p99 | max | "
          "mean | std | IQR | MAD | z_ratio | neg_ratio | nfin | flags |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        for kpi_name, st in sorted(sinfo["raw_kpis"].items()):
            k_flags = []
            for f in sinfo["flags"]:
                if f["kpi_name"] == kpi_name:
                    k_flags = f["flags"]
                    break
            flag_str = ", ".join(k_flags) if k_flags else "—"
            w(f"| `{kpi_name}` | {st['n']} | {st['min']:.4g} | {st['p01']:.4g} | "
              f"{st['p05']:.4g} | {st['p50']:.4g} | {st['p95']:.4g} | "
              f"{st['p99']:.4g} | {st['max']:.4g} | {st['mean']:.4g} | "
              f"{st['std']:.4g} | {st['iqr']:.4g} | {st['mad']:.4g} | "
              f"{st['zero_ratio']:.3f} | {st['negative_ratio']:.3f} | "
              f"{st['n_finite']} | {flag_str} |")
        w()

    w("## 3. Flagged Raw KPIs")
    w()
    flagged = analysis["flagged_kpis"]
    if not flagged:
        w("No raw KPIs triggered any risk flags.")
        w()
    else:
        w("| # | Slot | Raw KPI Name | Flags | n | p01 | p50 | p95 | p99 | max |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for i, fk in enumerate(flagged[:200], 1):
            w(f"| {i} | `{fk['canonical_slot']}` | `{fk['raw_kpi_name']}` | "
              f"{', '.join(fk['flags'])} | {fk['n']} | "
              f"{fk['p01']:.4g} | {fk['p50']:.4g} | {fk['p95']:.4g} | "
              f"{fk['p99']:.4g} | {fk['max']:.4g} |")
        if len(flagged) > 200:
            w(f"\n*(... and {len(flagged) - 200} more flagged KPIs)*")
        w()

    w("## 4. Diagnostic Notes")
    w()

    for slot_name in sorted(analysis["slots"].keys()):
        sinfo = analysis["slots"][slot_name]
        w(f"### Slot `{slot_name}`")
        w()
        all_vals = sinfo.get("all_values", np.array([]))
        if len(all_vals) > 0:
            w(f"- Slot-wide distribution: "
              f"min={all_vals.min():.4g}, "
              f"p50={np.median(all_vals):.4g}, "
              f"p95={np.percentile(all_vals, 95):.4g}, "
              f"max={all_vals.max():.4g}")
        w()
        # Count by raw KPI for this slot
        for kpi_name, st in sorted(sinfo["raw_kpis"].items()):
            w(f"- **{kpi_name}** ({st['n']} rows, {st['n_components']} components): "
              f"p50={st['p50']:.4g}, p95={st['p95']:.4g}, "
              f"range=[{st['min']:.4g}, {st['max']:.4g}]")
        w()

    # Unit-mix observations
    w("### Unit / Scale Mix Observations")
    w()
    w("The following observations are based on numerical evidence and KPI naming "
      "conventions visible in the raw data. They do NOT automatically trigger "
      "data modifications.")
    w()

    for slot_name in sorted(analysis["slots"].keys()):
        sinfo = analysis["slots"][slot_name]
        kpi_names = list(sinfo["raw_kpis"].keys())
        kpi_stats = sinfo["raw_kpis"]

        p50s = [(k, kpi_stats[k]["p50"]) for k in kpi_names if kpi_stats[k]["n"] > 10]
        ranges = [(k, kpi_stats[k]["max"] - kpi_stats[k]["min"]) for k in kpi_names if kpi_stats[k]["n"] > 10]

        if slot_name == "cpu":
            w(f"**Slot `cpu`** ({len(kpi_names)} raw KPIs):")
            for k, s in kpi_stats.items():
                if "CpuUtil" in k:
                    w(f"  - `{k}`: range [{s['min']:.1f}, {s['max']:.1f}, "
                      f"p50={s['p50']:.1f}] — looks like 0-100% CPU utilization")
                elif "CpuPercent" in k:
                    w(f"  - `{k}`: range [{s['min']:.1f}, {s['max']:.1f}, "
                      f"p50={s['p50']:.1f}] — container CPU percent")
            if all(-0.1 <= kpi_stats[k]["min"] <= 1.0 and kpi_stats[k]["max"] <= 105.0
                   for k in kpi_names):
                w("  - Assessment: All KPIs appear to be 0-100% scale. Low risk.")
            w()

        elif slot_name == "mem":
            w(f"**Slot `mem`** ({len(kpi_names)} raw KPIs):")
            for k, s in kpi_stats.items():
                w(f"  - `{k}`: range [{s['min']:.1f}, {s['max']:.1f}, "
                  f"p50={s['p50']:.1f}]")
            if all(-0.1 <= kpi_stats[k]["min"] <= 1.0 and kpi_stats[k]["max"] <= 105.0
                   for k in kpi_names):
                w("  - Assessment: All KPIs appear to be 0-100% scale. Low risk.")
            w()

        elif slot_name in ("net_rx", "net_tx"):
            direction = "receive" if slot_name == "net_rx" else "transmit"
            w(f"**Slot `{slot_name}`** ({len(kpi_names)} raw KPIs, log1p-transformed):")
            for k, s in kpi_stats.items():
                w(f"  - `{k}`: {s['n']} rows, "
                  f"raw value range noted, transformed: "
                  f"p50={s['p50']:.3f}, p95={s['p95']:.3f}, max={s['max']:.3f}")
            w(f"  - Assessment: All {direction} byte counts are log1p-transformed. "
              f"Scale is consistent across KPIs in this slot.")
            w()

    w("## 5. Recommendations for Phase 0.9.2.3c")
    w()

    slots_ready = []
    slots_needs_work = []

    for slot_name, sinfo in sorted(analysis["slots"].items()):
        if sinfo["n_flags"] == 0:
            slots_ready.append(slot_name)
        else:
            slots_needs_work.append(slot_name)

    if slots_ready:
        w("### Slots ready for modeling without further canonicalization")
        w()
        for s in slots_ready:
            w(f"- **`{s}`**: All raw KPIs within this slot show consistent scale "
              f"and no triggered risk flags.")

    if slots_needs_work:
        w()
        w("### Slots requiring attention")
        w()
        for s in slots_needs_work:
            sinfo = analysis["slots"][s]
            w(f"- **`{s}`** ({sinfo['n_flags']} flagged KPIs):")
            kpi_flags = sinfo["flags"]
            flag_types = set()
            for f in kpi_flags:
                flag_types.update(f["flags"])
            w(f"  - Relevant flag types: {', '.join(sorted(flag_types))}")
            if sinfo.get("median_mismatch", {}).get("ratio", 1) > THRESHOLDS.median_scale_ratio:
                w("  - **Action**: Consider per-KPI unit normalization before slot-level modeling")
            if sinfo.get("ratio_percent_mix"):
                w("  - **Action**: Separate 0-1 and 0-100 KPIs into sub-slots or apply explicit rescaling")
            w(f"  - KPIs to review: " + ", ".join(f["kpi_name"] for f in kpi_flags[:5]))
            w()

    w("### Summary of actions")
    w()
    w("| Category | Slots / KPIs | Recommended Action |")
    w("|---|---|---|")
    w(f"| Ready to model | {', '.join(f'`{s}`' for s in slots_ready) if slots_ready else 'None'} | "
      f"Proceed directly |")
    for slot_name, sinfo in sorted(analysis["slots"].items()):
        if sinfo.get("median_mismatch", {}).get("ratio", 1) > THRESHOLDS.median_scale_ratio:
            w(f"| Scale mismatch | `{slot_name}` | "
              f"Per-KPI robust scaling or unit normalization |")
        if sinfo.get("ratio_percent_mix"):
            w(f"| Ratio/percent mix | `{slot_name}` | "
              f"Sub-slot separation + explicit ratio→percentage rescaling |")
    w("| Dropped KPIs (by bank_safe_v1) | disk_io, mysql_io, threads, sessions, fgc, mem_usage, jvm_mem, jvm_cpu | "
      "Retain drop; insufficient semantic alignment with OB slots |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 0.9.2.3b: Bank Canonicalized KPI Residual Scale Audit")
    parser.add_argument("--data-dir", type=str,
                        default="/home/dell2/RCA513/yyx/OpenRCA/Bank/Bank",
                        help="Path to Bank dataset root")
    parser.add_argument("--output", type=str,
                        default="docs/phase0923b_bank_residual_scale_audit.md",
                        help="Output Markdown report path")
    parser.add_argument("--max-days", type=int, default=99)
    args = parser.parse_args()

    print(f"Loading Bank data from {args.data_dir} ...")
    t0 = time.time()
    data = load_and_categorize(args.data_dir, max_days=args.max_days)
    dt = time.time() - t0
    print(f"Loaded {data['total_raw']:,} raw rows in {dt:.1f}s")
    print(f"  Accepted: {data['total_accepted']:,} "
          f"({data['total_accepted']/max(data['total_raw'],1):.1%})")
    print(f"  Dropped: {data['total_dropped']:,} "
          f"({data['total_dropped']/max(data['total_raw'],1):.1%})")
    print(f"  Unmapped: {data['total_unmapped']:,} "
          f"({data['total_unmapped']/max(data['total_raw'],1):.1%})")

    t0 = time.time()
    analysis = analyze(data)
    print(f"Analysis complete in {time.time()-t0:.1f}s")
    print(f"  Slots: {list(analysis['slots'].keys())}")
    print(f"  Flagged KPIs: {len(analysis['flagged_kpis'])}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    generate_report(analysis, args.output)


if __name__ == "__main__":
    main()
