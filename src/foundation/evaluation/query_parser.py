"""Query parser for OpenRCA query.csv format.

Strict GT isolation:
  - InferenceQuery: only query metadata + observation window. NEVER contains GT.
  - EvalTarget: ground-truth answers for evaluation ONLY. NEVER passed to model.

These two data classes must remain in physically separate code paths:
  - run_inference.py imports InferenceQuery, NEVER EvalTarget.
  - evaluate_predictions.py imports EvalTarget, reads record.csv.

UTC+8 timezone handling:
  OpenRCA official: all fault times in UTC+8 (Asia/Shanghai).
  Telemetry timestamps may also need conversion; adapter handles per-system.
"""

import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

OPENRCA_TZ = ZoneInfo("Asia/Shanghai")


# ═══════════════════════════════════════════════════════════════════════
# InferenceQuery: visible to model inference (NO GT)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class InferenceQuery:
    """Query metadata for model inference — ZERO ground truth.

    This is the ONLY query object that should be visible to:
      - telemetry loading
      - episode building
      - model forward pass
      - joint scoring
      - prediction generation

    Attributes:
        query_id: Unique query identifier (row index in query.csv).
        instruction: Raw natural-language instruction text.
        window_start: Observation window start (UTC+8 aware datetime).
        window_end: Observation window end (UTC+8 aware datetime).
        need_time: Whether occurrence time is requested (from task_index).
        need_component: Whether root-cause component is requested.
        need_reason: Whether failure reason is requested.
        expected_fault_count: How many faults to predict (from query text).
    """

    query_id: int
    instruction: str
    window_start: datetime
    window_end: datetime
    need_time: bool = True
    need_component: bool = True
    need_reason: bool = True
    expected_fault_count: int = 1


# ═══════════════════════════════════════════════════════════════════════
# EvalTarget: ground-truth answers for evaluation ONLY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvalTarget:
    """Ground-truth answers for one query — evaluation ONLY.

    This object MUST NEVER be visible to model inference, episode builder,
    or any code path that touches telemetry or model forward pass.

    Attributes:
        query_id: Matches InferenceQuery.query_id.
        root_causes: List of (component, datetime, reason, tolerance_min) tuples.
        need_time: Which fields were asked (used for exact match scoring).
        need_component: See above.
        need_reason: See above.
    """

    query_id: int
    root_causes: List[Tuple[str, str, str, int]] = field(default_factory=list)
    need_time: bool = True
    need_component: bool = True
    need_reason: bool = True


# ═══════════════════════════════════════════════════════════════════════
# Task field mapping
# ═══════════════════════════════════════════════════════════════════════

TASK_FIELD_MAP: Dict[str, Dict[str, bool]] = {
    "task_1": {"need_time": True, "need_component": False, "need_reason": False},
    "task_2": {"need_time": False, "need_component": False, "need_reason": True},
    "task_3": {"need_time": False, "need_component": True, "need_reason": False},
    "task_4": {"need_time": True, "need_component": False, "need_reason": True},
    "task_5": {"need_time": True, "need_component": True, "need_reason": False},
    "task_6": {"need_time": False, "need_component": True, "need_reason": True},
    "task_7": {"need_time": True, "need_component": True, "need_reason": True},
}

# Regex patterns
_DATE_PATTERN = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
    re.IGNORECASE,
)
_TIME_RANGE_PATTERN = re.compile(
    r'(?:from|between)\s+(\d{1,2}):(\d{2})\s+(?:to|and)\s+(\d{1,2}):(\d{2})',
    re.IGNORECASE,
)
_GT_TIME_PATTERN = re.compile(
    r'root cause occurrence time is within\s+(\d+)\s+minutes.*?of\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
    re.IGNORECASE,
)
_GT_COMPONENT_PATTERN = re.compile(
    r'(?:The\s+[\d]+-th\s+)?predicted root cause component is\s+([^\n\r]+)',
    re.IGNORECASE,
)
_GT_REASON_PATTERN = re.compile(
    r'(?:The\s+[\d]+-th\s+)?predicted root cause reason is\s+([^\n\r]+)',
    re.IGNORECASE,
)
_MULTI_COMPONENT_PATTERN = re.compile(
    r'The\s+(\d+)-th\s+predicted root cause component is\s+([^\n\r]+)',
    re.IGNORECASE,
)
_MULTI_REASON_PATTERN = re.compile(
    r'The\s+(\d+)-th\s+predicted root cause reason is\s+([^\n\r]+)',
    re.IGNORECASE,
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _make_utc8_dt(year: int, month: int, day: int,
                   hour: int = 0, minute: int = 0,
                   second: int = 0) -> datetime:
    """Create a UTC+8 aware datetime."""
    return datetime(year, month, day, hour, minute, second,
                    tzinfo=OPENRCA_TZ)


def parse_date_from_text(text: str) -> Optional[datetime]:
    """Extract date from instruction text (returns UTC+8 aware datetime at 00:00)."""
    m = _DATE_PATTERN.search(text)
    if not m:
        return None
    month = _MONTH_MAP.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3))
    if month is None:
        return None
    return _make_utc8_dt(year, month, day)


def parse_time_range_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
    """Extract time range as (start_hour, start_min, end_hour, end_min)."""
    m = _TIME_RANGE_PATTERN.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _parse_gt_time(scoring_points: str) -> Tuple[str, int]:
    m = _GT_TIME_PATTERN.search(scoring_points)
    if m:
        return m.group(2), int(m.group(1))
    return "", 1


def _parse_gt_components(scoring_points: str) -> List[str]:
    components = []
    for m in _MULTI_COMPONENT_PATTERN.finditer(scoring_points):
        components.append(m.group(2).strip())
    if not components:
        m = _GT_COMPONENT_PATTERN.search(scoring_points)
        if m:
            components.append(m.group(1).strip())
    return components


def _parse_gt_reasons(scoring_points: str) -> List[str]:
    reasons = []
    for m in _MULTI_REASON_PATTERN.finditer(scoring_points):
        reasons.append(m.group(2).strip())
    if not reasons:
        m = _GT_REASON_PATTERN.search(scoring_points)
        if m:
            reasons.append(m.group(1).strip())
    return reasons


# ═══════════════════════════════════════════════════════════════════════
# Main parse: returns separated InferenceQuery + EvalTarget
# ═══════════════════════════════════════════════════════════════════════

def parse_query_csv(query_csv_path: str) -> Tuple[
        List[InferenceQuery], Dict[int, EvalTarget]]:
    """Parse OpenRCA query.csv into GT-isolated InferenceQuery and EvalTarget.

    InferenceQuery objects contain ONLY observation metadata:
      - query_id, instruction, window_start, window_end
      - need_time / need_component / need_reason (from task_index)
      - expected_fault_count

    EvalTarget objects contain ONLY ground truth (for evaluation):
      - List of (component, datetime, reason, tolerance_min) tuples

    These two types must NEVER be passed to the same function.
    Inference code imports InferenceQuery ONLY.
    Evaluation code imports EvalTarget + reads record.csv.

    Returns:
        (inference_queries, eval_targets_by_id)
          inference_queries: one per query row (not per fault).
          eval_targets_by_id: dict keyed by query_id.
    """
    df = pd.read_csv(query_csv_path)
    inference_queries: List[InferenceQuery] = []
    eval_targets: Dict[int, EvalTarget] = {}

    for idx, row in df.iterrows():
        task = str(row.get("task_index", ""))
        instruction = str(row.get("instruction", ""))
        scoring = str(row.get("scoring_points", ""))

        date = parse_date_from_text(instruction)
        time_range = parse_time_range_from_text(instruction)
        if date is None or time_range is None:
            continue

        sh, sm, eh, em = time_range
        window_start = _make_utc8_dt(date.year, date.month, date.day, sh, sm)
        window_end = _make_utc8_dt(date.year, date.month, date.day, eh, em)

        # Handle cross-midnight: if window_end <= window_start, it wraps to next day
        if window_end <= window_start:
            window_end += timedelta(days=1)

        fields = TASK_FIELD_MAP.get(task, {})
        need_time = fields.get("need_time", True)
        need_component = fields.get("need_component", True)
        need_reason = fields.get("need_reason", True)

        # Parse GT from scoring_points (hidden)
        gt_datetime, gt_tolerance = _parse_gt_time(scoring)
        gt_components = _parse_gt_components(scoring)
        gt_reasons = _parse_gt_reasons(scoring)
        expected_fault_count = max(len(gt_components), len(gt_reasons), 1)

        # InferenceQuery: zero GT
        inference_queries.append(InferenceQuery(
            query_id=idx,
            instruction=instruction,
            window_start=window_start,
            window_end=window_end,
            need_time=need_time,
            need_component=need_component,
            need_reason=need_reason,
            expected_fault_count=expected_fault_count,
        ))

        # EvalTarget: only GT
        root_causes = []
        for fi in range(expected_fault_count):
            comp = gt_components[fi] if fi < len(gt_components) else ""
            dt_str = gt_datetime if fi == 0 else ""
            reason = gt_reasons[fi] if fi < len(gt_reasons) else ""
            root_causes.append((comp, dt_str, reason, gt_tolerance))
        eval_targets[idx] = EvalTarget(
            query_id=idx,
            root_causes=root_causes,
            need_time=need_time,
            need_component=need_component,
            need_reason=need_reason,
        )

    return inference_queries, eval_targets


def format_episode_window(window_start: datetime,
                           window_end: datetime,
                           burn_in_min: int = 60) -> Tuple[datetime, datetime]:
    """Return (data_start, data_end) including burn-in period before window."""
    return window_start - timedelta(minutes=burn_in_min), window_end
