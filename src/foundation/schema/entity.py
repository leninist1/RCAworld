"""Universal entity schema for RCAWorld-Foundation.

Every diagnosable object is an Entity with a typed observation specification.
Entities of different types may have different numbers and kinds of observable
features; type-specific encoders project them to a common latent space.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class EntityType(str, Enum):
    """Typed entity vocabulary covering heterogeneous software systems.

    Microservice systems:   SERVICE, CONTAINER, POD, PROCESS
    Banking systems:        DATABASE, MIDDLEWARE, CONTAINER
    Telecom systems:        HOST, NETWORK_INTERFACE, PROCESS, SERVICE
    Generic systems:        HOST, PROCESS, STORAGE_VOLUME
    """

    # compute / runtime
    HOST = "host"
    VIRTUAL_MACHINE = "virtual_machine"
    CONTAINER = "container"
    POD = "pod"
    PROCESS = "process"

    # application
    SERVICE = "service"
    API_OPERATION = "api_operation"

    # data
    DATABASE = "database"
    MIDDLEWARE = "middleware"

    # network
    NETWORK_INTERFACE = "network_interface"
    NETWORK_LINK = "network_link"

    # storage
    STORAGE_VOLUME = "storage_volume"

    # business
    BUSINESS_JOURNEY = "business_journey"

    # catch-all for unknown types
    GENERIC = "generic"


@dataclass
class ObservationSpec:
    """Specification of one observable feature for an entity type.

    Attributes:
        name: Human-readable feature name (e.g. 'cpu_usage', 'error_rate').
        feature_type: Semantic category (metric, log_count, trace_latency, alert).
        default_value: Fill value for missing data.
        normalize: Whether to apply standard scaling.
    """

    name: str
    feature_type: str = "metric"  # metric | log_template_count | trace_latency | alert
    default_value: float = 0.0
    normalize: bool = True


# Pre-defined observation specs for each entity type.
# These are the default feature spaces; adapters may override them.

SERVICE_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_usage"),
    ObservationSpec("memory_usage"),
    ObservationSpec("net_rx_bytes"),
    ObservationSpec("net_tx_bytes"),
    ObservationSpec("latency_p90"),
    ObservationSpec("latency_p99"),
    ObservationSpec("request_rate"),
    ObservationSpec("error_rate"),
]

CONTAINER_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_percent"),
    ObservationSpec("mem_percent"),
    ObservationSpec("mem_usage"),
    ObservationSpec("net_rx_bytes"),
    ObservationSpec("net_tx_bytes"),
    ObservationSpec("disk_io"),
    ObservationSpec("threads"),
    ObservationSpec("sessions"),
    ObservationSpec("fgc"),
    ObservationSpec("mysql_io"),
    ObservationSpec("jvm_cpu"),
    ObservationSpec("jvm_mem"),
]

DATABASE_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_usage"),
    ObservationSpec("memory_usage"),
    ObservationSpec("disk_io"),
    ObservationSpec("connections"),
    ObservationSpec("lock_wait"),
    ObservationSpec("slow_query_count"),
    ObservationSpec("qps"),
]

HOST_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_usage"),
    ObservationSpec("memory_usage"),
    ObservationSpec("disk_io"),
    ObservationSpec("net_rx_bytes"),
    ObservationSpec("net_tx_bytes"),
    ObservationSpec("load_avg"),
]

PROCESS_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_percent"),
    ObservationSpec("mem_percent"),
    ObservationSpec("thread_count"),
    ObservationSpec("fd_count"),
    ObservationSpec("restart_count"),
]

NETWORK_INTERFACE_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("rx_bytes"),
    ObservationSpec("tx_bytes"),
    ObservationSpec("rx_errors"),
    ObservationSpec("tx_errors"),
    ObservationSpec("rx_dropped"),
    ObservationSpec("tx_dropped"),
]

MIDDLEWARE_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("cpu_usage"),
    ObservationSpec("memory_usage"),
    ObservationSpec("request_rate"),
    ObservationSpec("error_rate"),
    ObservationSpec("latency_p99"),
    ObservationSpec("connection_pool"),
]

STORAGE_VOLUME_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("read_iops"),
    ObservationSpec("write_iops"),
    ObservationSpec("read_latency"),
    ObservationSpec("write_latency"),
    ObservationSpec("capacity_used_pct"),
]

GENERIC_OBS_SPECS: List[ObservationSpec] = [
    ObservationSpec("metric_0"),
    ObservationSpec("metric_1"),
    ObservationSpec("metric_2"),
    ObservationSpec("metric_3"),
    ObservationSpec("metric_4"),
    ObservationSpec("metric_5"),
    ObservationSpec("metric_6"),
    ObservationSpec("metric_7"),
]

# Lookup table: EntityType -> default observation specs
DEFAULT_OBS_SPECS: Dict[EntityType, List[ObservationSpec]] = {
    EntityType.SERVICE: SERVICE_OBS_SPECS,
    EntityType.CONTAINER: CONTAINER_OBS_SPECS,
    EntityType.POD: CONTAINER_OBS_SPECS,
    EntityType.DATABASE: DATABASE_OBS_SPECS,
    EntityType.HOST: HOST_OBS_SPECS,
    EntityType.PROCESS: PROCESS_OBS_SPECS,
    EntityType.NETWORK_INTERFACE: NETWORK_INTERFACE_OBS_SPECS,
    EntityType.MIDDLEWARE: MIDDLEWARE_OBS_SPECS,
    EntityType.STORAGE_VOLUME: STORAGE_VOLUME_OBS_SPECS,
    EntityType.GENERIC: GENERIC_OBS_SPECS,
}


@dataclass
class Entity:
    """A typed diagnosable object in a system.

    Attributes:
        entity_id: Unique identifier within the episode (e.g. 'mysql01', 'checkoutservice').
        entity_type: Semantic type from EntityType enum.
        observation_specs: Ordered list of observable feature specifications.
        metadata: Optional key-value metadata (e.g. host IP, deployment info).
    """

    entity_id: str
    entity_type: EntityType
    observation_specs: List[ObservationSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_specs:
            self.observation_specs = DEFAULT_OBS_SPECS.get(
                self.entity_type, GENERIC_OBS_SPECS
            )

    @property
    def obs_dim(self) -> int:
        """Number of observation features for this entity."""
        return len(self.observation_specs)

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names."""
        return [s.name for s in self.observation_specs]
