"""Query parser for OpenRCA query.csv format.

Parses natural-language queries to extract:
  - Observation window (date + time range), usable for telemetry loading.
  - Requested output fields (time, component, reason) from task_index.
  - Ground-truth scoring points (ONLY for evaluation, must NOT leak to model).

Each query becomes an independent Episode in the strict evaluation protocol.
"""

import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd


@dataclass
class ParsedQuery:
    """A single OpenRCA query, parsed into machine-usable fields.

    Attributes:
        query_id: Unique query identifier (row index in query.csv).
        task_index: Task type string (task_1 .. task_7).
        instruction: Raw natural-language instruction text.
        window_start: Observation window start (datetime).
        window_end: Observation window end (datetime).
        need_time: Whether occurrence time is requested.
        need_component: Whether root-cause component is requested.
        need_reason: Whether failure reason is requested.
        expected_fault_count: How many faults the query expects.

        # Ground truth (ONLY for evaluation — must NOT be read during inference):
        gt_datetime: Ground-truth occurrence datetime string (e.g., "2021-03-04 14:57:00").
        gt_component: Ground-truth root cause component name.
        gt_reason: Ground-truth failure reason.
        gt_tolerance_min: Time tolerance in minutes for exact match.
    """

    query_id: int
    task_index: str
    instruction: str
    window_start: datetime
    window_end: datetime
    need_time: bool = True
    need_component: bool = True
    need_reason: bool = True
    expected_fault_count: int = 1

    # Ground truth (hidden from model)
    gt_datetime: str = ""
    gt_component: str = ""
    gt_reason: str = ""
    gt_tolerance_min: int = 1


# Mapping from task_index to requested output fields
TASK_FIELD_MAP: Dict[str, Dict[str, bool]] = {
    "task_1": {"need_time": True, "need_component": False, "need_reason": False},
    "task_2": {"need_time": False, "need_component": False, "need_reason": True},
    "task_3": {"need_time": False, "need_component": True, "need_reason": False},
    "task_4": {"need_time": True, "need_component": False, "need_reason": True},
    "task_5": {"need_time": True, "need_component": True, "need_reason": False},
    "task_6": {"need_time": False, "need_component": True, "need_reason": True},
    "task_7": {"need_time": True, "need_component": True, "need_reason": True},
}

# Regex patterns for parsing instruction text
# Matches: "March 4, 2021"  or  "March 20, 2022"  or  "April 11, 2020"  or  "May 22, 2020"
_DATE_PATTERN = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
    re.IGNORECASE,
)

# Matches: "from 14:30 to 15:00"  or  "between 09:00 and 09:30"  or  "from 00:30 to 01:00"
_TIME_RANGE_PATTERN = re.compile(
    r'(?:from|between)\s+(\d{1,2}):(\d{2})\s+(?:to|and)\s+(\d{1,2}):(\d{2})',
    re.IGNORECASE,
)

# Parse ground truth from scoring_points column
# "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-04 14:57:00"
_GT_TIME_PATTERN = re.compile(
    r'root cause occurrence time is within\s+(\d+)\s+minutes.*?of\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
    re.IGNORECASE,
)

# "The only predicted root cause component is Redis02"
_GT_COMPONENT_PATTERN = re.compile(
    r'(?:The\s+[\d]+-th\s+)?predicted root cause component is\s+([^\n\r]+)',
    re.IGNORECASE,
)

# "The only predicted root cause reason is high memory usage"
_GT_REASON_PATTERN = re.compile(
    r'(?:The\s+[\d]+-th\s+)?predicted root cause reason is\s+([^\n\r]+)',
    re.IGNORECASE,
)

# For multi-fault queries (e.g., "The 1-th predicted root cause component is...")
_MULTI_COMPONENT_PATTERN = re.compile(
    r'The\s+(\d+)-th\s+predicted root cause component is\s+([^\n\r]+)',
    re.IGNORECASE,
)
_MULTI_REASON_PATTERN = re.compile(
    r'The\s+(\d+)-th\s+predicted root cause reason is\s+([^\n\r]+)',
    re.IGNORECASE,
)

# Month name to number
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date_from_text(text: str) -> Optional[datetime]:
    """Extract date from instruction text (returns date with time set to 00:00)."""
    m = _DATE_PATTERN.search(text)
    if not m:
        return None
    month = _MONTH_MAP.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3))
    if month is None:
        return None
    return datetime(year, month, day)


def parse_time_range_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
    """Extract time range as (start_hour, start_min, end_hour, end_min)."""
    m = _TIME_RANGE_PATTERN.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def parse_gt_time(scoring_points: str) -> Tuple[str, int]:
    """Extract ground-truth datetime and tolerance (minutes) from scoring_points."""
    m = _GT_TIME_PATTERN.search(scoring_points)
    if m:
        return m.group(2), int(m.group(1))
    return "", 1


def parse_gt_components(scoring_points: str) -> List[str]:
    """Extract ground-truth components from scoring_points (supports multi-fault)."""
    components = []
    for m in _MULTI_COMPONENT_PATTERN.finditer(scoring_points):
        components.append(m.group(2).strip())
    if not components:
        m = _GT_COMPONENT_PATTERN.search(scoring_points)
        if m:
            components.append(m.group(1).strip())
    return components


def parse_gt_reasons(scoring_points: str) -> List[str]:
    """Extract ground-truth reasons from scoring_points (supports multi-fault)."""
    reasons = []
    for m in _MULTI_REASON_PATTERN.finditer(scoring_points):
        reasons.append(m.group(2).strip())
    if not reasons:
        m = _GT_REASON_PATTERN.search(scoring_points)
        if m:
            reasons.append(m.group(1).strip())
    return reasons


def parse_query_csv(query_csv_path: str) -> List[ParsedQuery]:
    """Parse OpenRCA query.csv into a list of structured ParsedQuery objects.

    Each row in query.csv contains:
      - task_index: task type (task_1 .. task_7)
      - instruction: natural language query with observation window
      - scoring_points: ground truth answers (hidden from model)

    Returns:
        List of ParsedQuery objects, one per query row.
    """
    df = pd.read_csv(query_csv_path)
    queries: List[ParsedQuery] = []

    for idx, row in df.iterrows():
        task = str(row.get("task_index", ""))
        instruction = str(row.get("instruction", ""))
        scoring = str(row.get("scoring_points", ""))

        # Parse observation window
        date = parse_date_from_text(instruction)
        time_range = parse_time_range_from_text(instruction)

        if date is None or time_range is None:
            continue

        sh, sm, eh, em = time_range
        window_start = date.replace(hour=sh, minute=sm, second=0, microsecond=0)
        window_end = date.replace(hour=eh, minute=em, second=0, microsecond=0)

        # Parse requested fields from task_index
        fields = TASK_FIELD_MAP.get(task, {})
        need_time = fields.get("need_time", True)
        need_component = fields.get("need_component", True)
        need_reason = fields.get("need_reason", True)

        # Parse ground truth (only for evaluation)
        gt_datetime, gt_tolerance = parse_gt_time(scoring)
        gt_components = parse_gt_components(scoring)
        gt_reasons = parse_gt_reasons(scoring)
        expected_fault_count = max(len(gt_components), len(gt_reasons), 1)

        # Each query row may contain multiple faults; create one ParsedQuery per fault
        # for simpler evaluation loop
        for fi in range(expected_fault_count):
            queries.append(ParsedQuery(
                query_id=idx,
                task_index=task,
                instruction=instruction,
                window_start=window_start,
                window_end=window_end,
                need_time=need_time,
                need_component=need_component,
                need_reason=need_reason,
                expected_fault_count=expected_fault_count,
                gt_datetime=gt_datetime if fi == 0 else "",
                gt_component=(gt_components[fi] if fi < len(gt_components) else ""),
                gt_reason=(gt_reasons[fi] if fi < len(gt_reasons) else ""),
                gt_tolerance_min=gt_tolerance,
            ))

    return queries


def format_episode_window(window_start: datetime,
                           window_end: datetime,
                           burn_in_min: int = 60) -> Tuple[datetime, datetime]:
    """Return (data_start, data_end) including burn-in period before window.

    The burn_in period allows the RSSM to build context before the query window.
    It is part of model input but NOT part of the query's observation window.
    """
    return window_start - timedelta(minutes=burn_in_min), window_end
