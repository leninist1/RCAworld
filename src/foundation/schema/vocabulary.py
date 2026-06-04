"""Mechanism vocabulary for reason inference.

Each mechanism token corresponds to a latent perturbation operator that
modifies the world model's state transition. This enables counterfactual
reasoning: "which mechanism, when applied to which entity at which time,
best explains the observed trajectory?"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class MechanismToken(str, Enum):
    """Canonical failure mechanism tokens.

    Each token represents a class of fault that can be applied as a latent
    perturbation to the world model's state transition function.
    """

    CPU_SATURATION = "cpu_saturation"
    MEMORY_PRESSURE = "memory_pressure"
    NETWORK_LATENCY = "network_latency"
    PACKET_LOSS = "packet_loss"
    DISK_IO_LOAD = "disk_io_load"
    CONTAINER_RESTART = "container_restart"
    PROCESS_CRASH = "process_crash"
    DB_CONNECTION_EXHAUSTION = "db_connection_exhaustion"
    LOCK_CONTENTION = "lock_contention"
    CONFIG_ERROR = "config_error"
    OOM_KILL = "oom_kill"
    SLOW_QUERY = "slow_query"
    JVM_GC_PRESSURE = "jvm_gc_pressure"
    THREAD_EXHAUSTION = "thread_exhaustion"
    CASCADING_FAILURE = "cascading_failure"
    UNKNOWN_MECHANISM = "unknown_mechanism"

    # aliases for common labels in existing datasets
    CUSTOM = "custom"


# Mapping from common dataset labels to canonical mechanism tokens
LABEL_TO_MECHANISM: Dict[str, MechanismToken] = {
    # Nezha labels
    "cpu_contention": MechanismToken.CPU_SATURATION,
    "cpu_consumed": MechanismToken.CPU_SATURATION,
    "high_cpu": MechanismToken.CPU_SATURATION,
    "memory_leak": MechanismToken.MEMORY_PRESSURE,
    "high_memory": MechanismToken.MEMORY_PRESSURE,
    "oom": MechanismToken.OOM_KILL,
    "network_delay": MechanismToken.NETWORK_LATENCY,
    "network_packet_loss": MechanismToken.PACKET_LOSS,
    "disk_io": MechanismToken.DISK_IO_LOAD,
    "exception": MechanismToken.PROCESS_CRASH,
    "container_read_io": MechanismToken.DISK_IO_LOAD,
    "container_write_io": MechanismToken.DISK_IO_LOAD,
    "jvm_cpu": MechanismToken.CPU_SATURATION,
    "jvm_oom": MechanismToken.OOM_KILL,
    # OpenRCA Bank labels
    "mysql connection exhaustion": MechanismToken.DB_CONNECTION_EXHAUSTION,
    "disk io": MechanismToken.DISK_IO_LOAD,
    "high cpu utilization": MechanismToken.CPU_SATURATION,
    "high memory usage": MechanismToken.MEMORY_PRESSURE,
    "network latency": MechanismToken.NETWORK_LATENCY,
    "packet loss": MechanismToken.PACKET_LOSS,
    "container restart": MechanismToken.CONTAINER_RESTART,
    "restart": MechanismToken.CONTAINER_RESTART,
    "cpu consumed": MechanismToken.CPU_SATURATION,
    # OpenRCA Market labels
    "container cpu": MechanismToken.CPU_SATURATION,
    "container memory": MechanismToken.MEMORY_PRESSURE,
    "network delay": MechanismToken.NETWORK_LATENCY,
    # OpenRCA Telecom labels
    "disk utilization": MechanismToken.DISK_IO_LOAD,
    "jvm full gc": MechanismToken.JVM_GC_PRESSURE,
    "jvm oom": MechanismToken.OOM_KILL,
    "thread pool exhaustion": MechanismToken.THREAD_EXHAUSTION,
}


@dataclass
class MechanismVocabulary:
    """Manages the mechanism token vocabulary with learnable embeddings.

    Each mechanism has an embedding that parameterizes a perturbation operator
    applied to the RSSM state transition.

    Attributes:
        num_mechanisms: Number of canonical mechanism types.
        embed_dim: Dimension of mechanism embeddings.
        token_to_idx: Mapping from MechanismToken to index.
        idx_to_token: Inverse mapping.
        embeddings: [num_mechanisms, embed_dim] learnable embeddings.
    """

    embed_dim: int = 64
    _include_unknown: bool = True

    def __post_init__(self):
        tokens = list(MechanismToken)
        if self._include_unknown:
            pass  # UNKNOWN_MECHANISM is already in the enum
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}
        self.num_mechanisms = len(tokens)
        self._embeddings: Optional[np.ndarray] = None
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        rng = np.random.RandomState(42)
        self._embeddings = rng.randn(self.num_mechanisms, self.embed_dim).astype(
            np.float32
        ) * 0.02

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: np.ndarray):
        self._embeddings = value

    def get_idx(self, token: MechanismToken) -> int:
        return self.token_to_idx[token]

    def get_token(self, idx: int) -> MechanismToken:
        return self.idx_to_token[idx]

    def resolve_label(self, label: str) -> MechanismToken:
        """Map a raw dataset label string to a canonical mechanism token."""
        label_lower = label.lower().strip().replace("_", " ").replace("-", " ")
        for key, token in LABEL_TO_MECHANISM.items():
            if key in label_lower or label_lower in key:
                return token
        return MechanismToken.UNKNOWN_MECHANISM

    def get_embedding(self, token: MechanismToken) -> np.ndarray:
        return self._embeddings[self.get_idx(token)]

    def get_all_tokens(self) -> List[MechanismToken]:
        return list(self.token_to_idx.keys())
