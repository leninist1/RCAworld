"""Relation schema for typed relationships between entities."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RelationType(str, Enum):
    """Typed relations between entities.

    Explicit topology from system architecture or deployment manifests.
    """

    # service interactions
    CALLS = "calls"               # service A -> service B (RPC/HTTP)
    READS_FROM = "reads_from"     # service -> database
    WRITES_TO = "writes_to"       # service -> database

    # deployment / infrastructure
    DEPLOYED_ON = "deployed_on"   # service/process -> container/pod
    HOSTED_ON = "hosted_on"       # container/pod -> host
    BINDS_TO = "binds_to"         # process -> port

    # network
    CONNECTS = "connects"         # network_link -> host

    # data-dependency
    DEPENDS_ON = "depends_on"     # generic dependency

    # learnt / inferred
    CORRELATED_WITH = "correlated_with"  # data-driven correlation


@dataclass
class Relation:
    """A typed directed edge between two entities.

    Attributes:
        source_id: Entity ID of the source.
        target_id: Entity ID of the target.
        relation_type: Semantic relationship type.
        features: Optional per-relation features (e.g. call_count, latency).
        metadata: Additional relation metadata.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    features: Optional[dict] = None
    metadata: Optional[dict] = None
