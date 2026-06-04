"""Episode and query schema for RCAWorld-Foundation.

A SystemEpisode captures one system's full telemetry history with optional
root-cause labels. QueryMask specifies which fields the model should output,
matching OpenRCA's conditional query protocol.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .entity import Entity
from .relation import Relation
from .event import ObservationEvent


@dataclass
class QueryMask:
    """Specifies which root-cause fields the user query requires.

    OpenRCA protocol: queries may ask for any subset of {time, component, reason}.
    The model must output only the requested fields.

    Attributes:
        need_time: Whether occurrence time is requested.
        need_component: Whether root-cause component is requested.
        need_reason: Whether failure reason/mechanism is requested.
        expected_fault_count: If known, how many faults to return (None = unknown).
        natural_query: The raw natural language query string.
    """

    need_time: bool = True
    need_component: bool = True
    need_reason: bool = True
    expected_fault_count: Optional[int] = None
    natural_query: str = ""


@dataclass
class RootCauseLabel:
    """Ground-truth root cause annotation for one fault.

    Attributes:
        occurrence_datetime: Fault onset timestamp (string or float).
        component: Root cause entity ID.
        reason: Failure mechanism (e.g. 'high memory usage').
        level: Granularity (e.g. 'pod', 'service', 'container').
        secondary_components: Entities affected by propagation.
    """

    occurrence_datetime: str
    component: str
    reason: str = ""
    level: str = "service"
    secondary_components: List[str] = field(default_factory=list)


@dataclass
class SystemEpisode:
    """A complete snapshot of one system over a time window.

    Includes entities, their relationships, all observation events,
    an optional natural-language query, and optional root-cause labels.

    Attributes:
        system_id: Unique identifier for the system (e.g. 'Bank', 'Market/cloudbed-1').
        entities: Ordered list of typed entities.
        relations: Ordered list of typed relations.
        events: All observation events (can be huge; lazy loading recommended).
        normal_mask: [T] boolean mask indicating normal (non-fault) timesteps.
        query: Natural language query (OpenRCA protocol).
        query_mask: Structured query mask specifying required output fields.
        labels: Ground-truth root cause labels (list for multi-fault episodes).
        metadata: System-level metadata dictionary.
    """

    system_id: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    events: List[ObservationEvent] = field(default_factory=list)
    normal_mask: Optional[Any] = None          # [T] bool array
    query: Optional[str] = None
    query_mask: Optional[QueryMask] = None
    labels: Optional[List[RootCauseLabel]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def get_entity_ids(self) -> List[str]:
        return [e.entity_id for e in self.entities]

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None
