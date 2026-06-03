"""Synthetic microservice system for Phase 0 MVP validation.

Generates multi-service metrics time series with a known dependency graph,
normal operation dynamics, and fault injection with propagation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ServiceConfig:
    name: str
    base_cpu: float = 0.3
    base_memory: float = 0.4
    base_qps: float = 100.0
    base_latency_p50: float = 5.0
    base_latency_p99: float = 20.0
    base_error_rate: float = 0.001
    noise_scale: float = 0.02


@dataclass
class EdgeConfig:
    src: int
    dst: int
    base_call_count: float = 50.0
    base_latency: float = 10.0
    base_error_rate: float = 0.001


@dataclass
class FaultConfig:
    root_cause: int
    fault_type: str = "cpu_hog"
    start_step: int = 200
    duration: int = 80
    intensity: float = 1.0


@dataclass
class EpisodeConfig:
    name: str
    total_steps: int = 400
    workload: str = "medium"
    fault: Optional[FaultConfig] = None

# Default microservice topology (8 services, 10 edges)
DEFAULT_SERVICES = [
    ServiceConfig("frontend", base_cpu=0.35, base_qps=200.0, base_latency_p50=8.0),
    ServiceConfig("auth", base_cpu=0.25, base_qps=80.0, base_latency_p50=3.0),
    ServiceConfig("cart", base_cpu=0.30, base_qps=120.0, base_latency_p50=6.0),
    ServiceConfig("catalog", base_cpu=0.28, base_qps=150.0, base_latency_p50=4.0),
    ServiceConfig("order", base_cpu=0.32, base_qps=90.0, base_latency_p50=7.0),
    ServiceConfig("payment", base_cpu=0.40, base_qps=85.0, base_latency_p50=10.0),
    ServiceConfig("shipping", base_cpu=0.22, base_qps=60.0, base_latency_p50=5.0),
    ServiceConfig("db", base_cpu=0.50, base_memory=0.70, base_qps=300.0, base_latency_p50=2.0),
]

DEFAULT_EDGES = [
    EdgeConfig(0, 1, base_call_count=60.0, base_latency=12.0),    # frontend -> auth
    EdgeConfig(0, 2, base_call_count=80.0, base_latency=15.0),    # frontend -> cart
    EdgeConfig(0, 3, base_call_count=100.0, base_latency=10.0),   # frontend -> catalog
    EdgeConfig(2, 4, base_call_count=50.0, base_latency=18.0),    # cart -> order
    EdgeConfig(4, 5, base_call_count=45.0, base_latency=20.0),    # order -> payment
    EdgeConfig(4, 6, base_call_count=40.0, base_latency=15.0),    # order -> shipping
    EdgeConfig(5, 7, base_call_count=200.0, base_latency=5.0),    # payment -> db
    EdgeConfig(2, 7, base_call_count=150.0, base_latency=6.0),    # cart -> db
    EdgeConfig(3, 7, base_call_count=180.0, base_latency=4.0),    # catalog -> db
    EdgeConfig(1, 7, base_call_count=50.0, base_latency=5.0),     # auth -> db
]

FAULT_TYPES = ["cpu_hog", "memory_leak", "network_delay", "pod_kill"]

N_METRICS = 6  # cpu, memory, qps, latency_p50, latency_p99, error_rate
N_EDGE_FEATURES = 3  # call_count, error_rate, latency
N_CONTEXT_FEATURES = 3  # total_qps, active_users, instance_count

METRIC_NAMES = ["cpu", "memory", "qps", "latency_p50", "latency_p99", "error_rate"]
EDGE_FEATURE_NAMES = ["call_count", "error_rate", "latency"]
CONTEXT_NAMES = ["total_qps", "active_users", "instance_count"]


class SyntheticMicroserviceSystem:
    """Generates synthetic microservice telemetry with normal and fault dynamics."""

    def __init__(
        self,
        services: List[ServiceConfig] = None,
        edges: List[EdgeConfig] = None,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.services = services or DEFAULT_SERVICES
        self.edges = edges or DEFAULT_EDGES
        self.N = len(self.services)
        self.E = len(self.edges)

        # Build adjacency structure
        self.children = {i: [] for i in range(self.N)}
        self.parents = {i: [] for i in range(self.N)}
        for e in self.edges:
            self.children[e.src].append(e)
            self.parents[e.dst].append(e)

        # edge_index format: [2, E]
        self.edge_index = np.array([[e.src for e in self.edges],
                                    [e.dst for e in self.edges]], dtype=np.int32)

    def _make_workload(self, steps: int, workload: str) -> np.ndarray:
        """Generate workload context features [steps, N_CONTEXT_FEATURES]."""
        w = np.zeros((steps, N_CONTEXT_FEATURES), dtype=np.float32)
        t = np.arange(steps)
        base_qps = {"low": 0.6, "medium": 1.0, "high": 1.8, "periodic": 1.0, "burst": 1.0}
        scale = base_qps.get(workload, 1.0)

        w[:, 0] = scale * (1.0 + 0.1 * np.sin(2 * np.pi * t / 360) + 0.05 * self.rng.randn(steps))
        if workload == "periodic":
            w[:, 0] *= 0.7 + 0.3 * np.sin(2 * np.pi * t / 200)
        elif workload == "burst":
            burst_mask = (t % 300 < 30).astype(np.float32)
            w[:, 0] *= 1.0 + 1.5 * burst_mask

        w[:, 0] = np.maximum(w[:, 0], 0.1)
        w[:, 1] = w[:, 0] * 5.0 + 2.0 * self.rng.randn(steps)  # active_users ~ proportional to qps
        w[:, 1] = np.maximum(w[:, 1], 1.0)
        w[:, 2] = np.full(steps, float(self.N), dtype=np.float32)  # instance count
        return w

    def _ar1_noise(self, steps: int, noise_scale: float, start_value: float = 0.0) -> np.ndarray:
        """Generate AR(1) noise process."""
        noise = np.zeros(steps, dtype=np.float32)
        noise[0] = start_value + noise_scale * self.rng.randn()
        phi = 0.9
        for t in range(1, steps):
            noise[t] = phi * noise[t-1] + noise_scale * self.rng.randn()
        return noise

    def generate_normal_metrics(self, steps: int, workload: str = "medium") -> np.ndarray:
        """Generate normal-operation node features [steps, N, N_METRICS]."""
        w = self._make_workload(steps, workload)
        nodes = np.zeros((steps, self.N, N_METRICS), dtype=np.float32)

        # Per-service base levels with AR(1) noise around them
        for i, svc in enumerate(self.services):
            noise = self._ar1_noise(steps, svc.noise_scale)
            cpu_ar = svc.base_cpu * (1.0 + 0.3 * noise)
            mem_ar = svc.base_memory * (1.0 + 0.1 * noise)
            lat_ar = svc.base_latency_p50 * (1.0 + 0.2 * noise)
            lat99_ar = svc.base_latency_p99 * (1.0 + 0.3 * noise)
            err_ar = svc.base_error_rate * (1.0 + 0.5 * self.rng.randn(steps))
            err_ar = np.maximum(err_ar, 0.0)

            # QPS modulated by workload
            qps = svc.base_qps * w[:, 0] * (1.0 + 0.1 * noise)

            # CPU and latency scale with QPS (square root law)
            cpu = cpu_ar * (1.0 + 0.5 * (w[:, 0] - 1.0))
            cpu = np.clip(cpu, 0.0, 0.95)
            mem = np.clip(mem_ar, 0.0, 0.95)
            lat50 = lat_ar * (1.0 + 0.3 * np.maximum(w[:, 0] - 1.0, 0.0))
            lat99 = lat99_ar * (1.0 + 0.4 * np.maximum(w[:, 0] - 1.0, 0.0))

            nodes[:, i, 0] = cpu
            nodes[:, i, 1] = mem
            nodes[:, i, 2] = np.maximum(qps, 0.0)
            nodes[:, i, 3] = np.maximum(lat50, 1.0)
            nodes[:, i, 4] = np.maximum(lat99, 1.0)
            nodes[:, i, 5] = err_ar

        return nodes

    def generate_normal_edges(self, steps: int) -> np.ndarray:
        """Generate normal-operation edge features [steps, E, N_EDGE_FEATURES]."""
        edges = np.zeros((steps, self.E, N_EDGE_FEATURES), dtype=np.float32)

        for j, ec in enumerate(self.edges):
            src_cfg = self.services[ec.src]
            noise = self._ar1_noise(steps, 0.03)
            call_count = ec.base_call_count * (1.0 + 0.2 * noise)
            error_rate = ec.base_error_rate * (1.0 + 0.3 * self.rng.randn(steps))
            error_rate = np.maximum(error_rate, 0.0)
            latency = ec.base_latency * (1.0 + 0.15 * noise)

            edges[:, j, 0] = np.maximum(call_count, 1.0)
            edges[:, j, 1] = error_rate
            edges[:, j, 2] = np.maximum(latency, 1.0)

        return edges

    def apply_fault(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        fault: FaultConfig,
    ) -> np.ndarray:
        """Apply fault to node features and propagate downstream. Modifies in-place."""
        steps = nodes.shape[0]
        r = fault.root_cause
        intensity = fault.intensity
        t0 = fault.start_step
        t1 = min(fault.start_step + fault.duration, steps)
        propagate_delay = 5  # steps before downstream effects appear

        # Direct fault effect on root cause service
        if fault.fault_type == "cpu_hog":
            nodes[t0:t1, r, 0] = np.clip(nodes[t0:t1, r, 0] + 0.5 * intensity, 0.0, 0.98)
            nodes[t0:t1, r, 3] *= 1.0 + 1.5 * intensity  # latency spike
            nodes[t0:t1, r, 4] *= 1.0 + 2.0 * intensity
            nodes[t0:t1, r, 5] += 0.05 * intensity  # error rate increase

        elif fault.fault_type == "memory_leak":
            ramp = np.linspace(0, 0.4 * intensity, t1 - t0)
            nodes[t0:t1, r, 1] = np.clip(nodes[t0:t1, r, 1] + ramp, 0.0, 0.98)
            nodes[t0:t1, r, 0] *= 1.0 + 0.3 * intensity
            nodes[t0:t1, r, 3] *= 1.0 + 0.8 * intensity
            nodes[t0:t1, r, 4] *= 1.0 + 1.2 * intensity
            nodes[t0:t1, r, 5] += 0.03 * intensity

        elif fault.fault_type == "network_delay":
            nodes[t0:t1, r, 3] *= 1.0 + 3.0 * intensity  # major latency spike
            nodes[t0:t1, r, 4] *= 1.0 + 5.0 * intensity
            nodes[t0:t1, r, 5] += 0.08 * intensity

        elif fault.fault_type == "pod_kill":
            nodes[t0:t1, r, 2] *= 0.1 * intensity  # QPS drops
            nodes[t0:t1, r, 0] *= 0.5
            nodes[t0:t1, r, 5] += 0.1 * intensity
            # downstream effects amplified (retries)

        # Propagate to downstream services with delay
        visited = set()
        queue = [(r, 0)]

        while queue:
            cur, depth = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)

            for ec in self.children.get(cur, []):
                child = ec.dst
                delay = propagate_delay * (depth + 1)
                ct0 = min(t0 + delay, steps)
                ct1 = min(t1 + delay, steps)
                if ct0 >= steps:
                    continue

                decay = 0.7 ** depth * intensity
                # Downstream effects: latency increase, error rate increase
                nodes[ct0:ct1, child, 3] *= 1.0 + 0.8 * decay
                nodes[ct0:ct1, child, 4] *= 1.0 + 1.2 * decay
                nodes[ct0:ct1, child, 5] += 0.02 * decay
                nodes[ct0:ct1, child, 2] *= 1.0 + 0.1 * decay  # retry traffic

                # Edge effects on incoming edges to child
                for inc_e in self.parents.get(child, []):
                    eidx = self.edges.index(inc_e)
                    edges[ct0:ct1, eidx, 1] += 0.03 * decay  # edge error rate
                    edges[ct0:ct1, eidx, 2] *= 1.0 + 0.5 * decay  # edge latency

                queue.append((child, depth + 1))

        return nodes, edges

    def generate_episode(
        self, config: EpisodeConfig
    ) -> Dict:
        """Generate one episode of data, optionally with a fault."""
        steps = config.total_steps

        nodes = self.generate_normal_metrics(steps, config.workload)
        edges = self.generate_normal_edges(steps)
        workload_ctx = self._make_workload(steps, config.workload)

        fault_info = {
            "has_fault": False,
            "root_cause": -1,
            "fault_start": -1,
            "fault_end": -1,
            "fault_type": "none",
        }

        if config.fault is not None:
            nodes, edges = self.apply_fault(nodes, edges, config.fault)
            fault_info = {
                "has_fault": True,
                "root_cause": config.fault.root_cause,
                "fault_start": config.fault.start_step,
                "fault_end": min(config.fault.start_step + config.fault.duration, steps),
                "fault_type": config.fault.fault_type,
            }

        return {
            "node_features": nodes.astype(np.float32),
            "edge_index": self.edge_index.copy(),
            "edge_features": edges.astype(np.float32),
            "workload_context": workload_ctx.astype(np.float32),
            "fault_info": fault_info,
            "config": config,
            "service_names": [s.name for s in self.services],
            "metric_names": METRIC_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
        }

    def generate_dataset(
        self,
        normal_episodes: int = 5,
        fault_episodes: int = 5,
        steps_per_episode: int = 400,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate a full dataset of normal and fault episodes."""
        normal_data = []
        for i in range(normal_episodes):
            workloads = ["low", "medium", "high", "periodic", "burst"]
            cfg = EpisodeConfig(
                name=f"normal_{i:02d}",
                total_steps=steps_per_episode,
                workload=workloads[i % len(workloads)],
            )
            normal_data.append(self.generate_episode(cfg))

        fault_data = []
        for i in range(fault_episodes):
            # Vary root cause and fault type
            rc = i % self.N
            ft = FAULT_TYPES[i % len(FAULT_TYPES)]
            cfg = EpisodeConfig(
                name=f"fault_{ft}_svc{rc}",
                total_steps=steps_per_episode,
                workload=self.rng.choice(["medium", "high"]),
                fault=FaultConfig(
                    root_cause=rc,
                    fault_type=ft,
                    start_step=150 + self.rng.randint(0, 50),
                    duration=60 + self.rng.randint(0, 40),
                    intensity=0.5 + 0.5 * self.rng.rand(),
                ),
            )
            fault_data.append(self.generate_episode(cfg))

        return normal_data, fault_data


if __name__ == "__main__":
    system = SyntheticMicroserviceSystem(seed=42)
    normal, fault = system.generate_dataset(normal_episodes=3, fault_episodes=3)

    for ep in normal:
        nf = ep["node_features"]
        print(f"Normal {ep['config'].name}: shape={nf.shape}, "
              f"fault={ep['fault_info']['has_fault']}")

    for ep in fault:
        nf = ep["node_features"]
        fi = ep["fault_info"]
        print(f"Fault {ep['config'].name}: shape={nf.shape}, "
              f"rc={fi['root_cause']}, start={fi['fault_start']}, type={fi['fault_type']}")
