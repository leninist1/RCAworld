"""Training pipeline for Graph-RSSM world model.

Phase 0 MVP: trains on synthetic normal data.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
from dataclasses import dataclass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph_rssm import GraphRSSM
from models.rssm import RSSMState
from training.losses import graph_rssm_loss
from data.synthetic import SyntheticMicroserviceSystem, EpisodeConfig, FaultConfig


@dataclass
class TrainConfig:
    # Data
    window_size: int = 24       # input sequence length
    batch_size: int = 8
    normal_episodes: int = 10
    val_episodes: int = 2
    steps_per_episode: int = 400

    # Model
    graph_hidden: int = 128
    num_heads: int = 4
    det_dim: int = 256
    stoch_dim: int = 16
    stoch_classes: int = 16

    # Training
    learning_rate: float = 1e-3
    max_epochs: int = 50
    kl_weight: float = 0.1
    edge_weight: float = 0.3
    free_bits: float = 1.0
    grad_clip: float = 10.0

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


def create_window_sequences(
    episode: Dict,
    window_size: int,
) -> List[Dict]:
    """Slice an episode into overlapping training windows.

    Returns list of dicts with:
        node_features: [window_size, N, D_node]
        edge_features: [window_size, E, D_edge]
        workload: [window_size, D_ctx]
    """
    T = episode["node_features"].shape[0]
    windows = []
    for start in range(0, T - window_size, window_size // 2):
        end = start + window_size
        if end > T:
            break
        w = {
            "node_features": episode["node_features"][start:end],
            "edge_features": episode["edge_features"][start:end],
            "workload": episode["workload_context"][start:end],
        }
        windows.append(w)
    return windows


class Trainer:
    def __init__(self, config: TrainConfig, rng: jax.random.PRNGKey):
        self.config = config
        self.rng = rng

        # Create model
        self.model = GraphRSSM(
            node_dim=6,
            edge_dim=3,
            ctx_dim=3,
            graph_hidden=config.graph_hidden,
            num_heads=config.num_heads,
            det_dim=config.det_dim,
            stoch_dim=config.stoch_dim,
            stoch_classes=config.stoch_classes,
        )

        # Initialize with dummy data
        rng, init_rng = jax.random.split(self.rng)
        dummy_nodes = jnp.ones((config.batch_size, config.window_size, 8, 6))
        dummy_edges = jnp.ones((config.batch_size, config.window_size, 10, 3))
        dummy_edge_index = jnp.zeros((2, 10), dtype=jnp.int32)
        dummy_workload = jnp.ones((config.batch_size, config.window_size, 3))

        self.params = self.model.init(
            init_rng,
            rng,
            dummy_nodes,
            dummy_edges,
            dummy_edge_index,
            dummy_workload,
        )

        # Set up optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adam(config.learning_rate),
        )

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.optimizer,
        )

        print(f"Model initialized. Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(self.params)):,}")

    @jax.jit
    def train_step(
        self,
        state: train_state.TrainState,
        rng: jax.random.PRNGKey,
        batch: Dict,
    ) -> Tuple[train_state.TrainState, Dict]:
        """Single training step."""

        def loss_fn(params):
            output = state.apply_fn(
                params,
                rng,
                batch["node_features"],
                batch["edge_features"],
                batch["edge_index"],
                batch["workload"],
                use_posterior=True,
                mutable=False,
            )

            # Targets: shift by 1 (predict next step)
            node_targets = batch["node_features"][:, 1:, :, :]
            edge_targets = batch["edge_features"][:, 1:, :, :]

            total_loss, loss_dict = graph_rssm_loss(
                output,
                node_targets,
                edge_targets,
                kl_weight=self.config.kl_weight,
                edge_weight=self.config.edge_weight,
                free_bits=self.config.free_bits,
            )
            return total_loss, (output, loss_dict)

        (loss, (output, loss_dict)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss_dict

    @jax.jit
    def eval_step(
        self,
        state: train_state.TrainState,
        rng: jax.random.PRNGKey,
        batch: Dict,
    ) -> Dict:
        """Evaluation step (no gradient)."""
        output = state.apply_fn(
            state.params,
            rng,
            batch["node_features"],
            batch["edge_features"],
            batch["edge_index"],
            batch["workload"],
            use_posterior=True,
            mutable=False,
        )

        node_targets = batch["node_features"][:, 1:, :, :]
        edge_targets = batch["edge_features"][:, 1:, :, :]

        _, loss_dict = graph_rssm_loss(
            output,
            node_targets,
            edge_targets,
            kl_weight=self.config.kl_weight,
            edge_weight=self.config.edge_weight,
            free_bits=self.config.free_bits,
        )
        return loss_dict

    def prepare_batch(self, windows: List[Dict], edge_index: np.ndarray) -> Dict:
        """Prepare a batch from a list of windows."""
        idx = np.random.choice(len(windows), self.config.batch_size, replace=True)
        batch_nodes = np.stack([windows[i]["node_features"] for i in idx])
        batch_edges = np.stack([windows[i]["edge_features"] for i in idx])
        batch_workload = np.stack([windows[i]["workload"] for i in idx])

        # Broadcast edge_index to batch
        batch_edge_index = np.broadcast_to(
            edge_index[np.newaxis, :, :],
            (self.config.batch_size, edge_index.shape[0], edge_index.shape[1])
        )

        return {
            "node_features": jnp.array(batch_nodes),
            "edge_features": jnp.array(batch_edges),
            "edge_index": jnp.array(edge_index, dtype=jnp.int32),
            "workload": jnp.array(batch_workload),
        }

    def train(
        self,
        train_windows: List[Dict],
        val_windows: List[Dict],
        edge_index: np.ndarray,
    ) -> Dict:
        """Main training loop."""
        history = {"train_loss": [], "val_loss": [], "metric_loss": [], "edge_loss": [], "kl_loss": []}

        steps_per_epoch = max(1, len(train_windows) // self.config.batch_size)

        for epoch in range(self.config.max_epochs):
            # Training
            train_losses = []
            for _ in range(steps_per_epoch):
                batch = self.prepare_batch(train_windows, edge_index)
                self.rng, step_rng = jax.random.split(self.rng)
                self.state, loss_dict = self.train_step(self.state, step_rng, batch)
                train_losses.append(loss_dict["total_loss"])

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                val_losses = []
                val_metric = []
                val_edge = []
                val_kl = []

                val_steps = max(1, len(val_windows) // self.config.batch_size)
                for _ in range(val_steps):
                    batch = self.prepare_batch(val_windows, edge_index)
                    self.rng, eval_rng = jax.random.split(self.rng)
                    loss_dict = self.eval_step(self.state, eval_rng, batch)
                    val_losses.append(loss_dict["total_loss"])
                    val_metric.append(loss_dict["metric_loss"])
                    val_edge.append(loss_dict["edge_loss"])
                    val_kl.append(loss_dict["kl_loss"])

                print(
                    f"Epoch {epoch+1:3d} | "
                    f"train_loss: {avg_train_loss:.4f} | "
                    f"val_loss: {np.mean(val_losses):.4f} | "
                    f"metric: {np.mean(val_metric):.4f} | "
                    f"edge: {np.mean(val_edge):.4f} | "
                    f"kl: {np.mean(val_kl):.4f}"
                )
                history["val_loss"].append(np.mean(val_losses))
                history["metric_loss"].append(np.mean(val_metric))
                history["edge_loss"].append(np.mean(val_edge))
                history["kl_loss"].append(np.mean(val_kl))

        return history

    def compute_residuals(
        self,
        episode: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction residuals for anomaly detection.

        Args:
            episode: dict with node_features, edge_features, workload_context, edge_index

        Returns:
            residuals: [T-1, N] per-node Gaussian NLL
            predictions: dict with predicted mu, sigma
        """
        T = episode["node_features"].shape[0]
        N = episode["node_features"].shape[1]

        # Use the whole episode (normal + potential fault)
        # TODO: use the first half as context to warm up, then predict the rest
        warmup = self.config.window_size // 2

        # Use initial warmup steps
        context = {
            "node_features": jnp.array(episode["node_features"][:warmup][None, :, :, :]),
            "edge_features": jnp.array(episode["edge_features"][:warmup][None, :, :, :]),
            "workload": jnp.array(episode["workload_context"][:warmup][None, :, :]),
        }

        # Get initial RSSM state from warmup
        dummy_edge_index = jnp.array(episode["edge_index"], dtype=jnp.int32)
        out = self.state.apply_fn(
            self.state.params,
            self.rng,
            context["node_features"],
            context["edge_features"],
            dummy_edge_index,
            context["workload"],
            use_posterior=True,
            mutable=False,
        )
        init_state = RSSMState(
            h=out["h_seq"][:, -1, :, :],
            z=out["z_seq"][:, -1, :, :],
            z_logits=out["post_logits"][:, -1, :, :],
        )

        # Predict step by step from warmup to end, using prior only
        residuals = []
        for t in range(warmup, T - 1):
            pass  # TODO: implement step-by-step prior rollout

        return np.array(residuals), {}


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    system = SyntheticMicroserviceSystem(seed=42)
    normal_eps, fault_eps = system.generate_dataset(
        normal_episodes=10,
        fault_episodes=3,
        steps_per_episode=400,
    )

    # Create training windows from normal episodes
    config = TrainConfig(
        window_size=24,
        batch_size=8,
        normal_episodes=10,
        max_epochs=50,
        kl_weight=0.1,
        edge_weight=0.3,
    )

    train_windows = []
    for ep in normal_eps:
        train_windows.extend(create_window_sequences(ep, config.window_size))

    # Use one normal episode for validation
    val_windows = create_window_sequences(normal_eps[0], config.window_size)

    print(f"Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    edge_index = system.edge_index

    # Initialize trainer
    rng = jax.random.PRNGKey(42)
    trainer = Trainer(config, rng)

    # Train
    print("Starting training...")
    history = trainer.train(train_windows, val_windows, edge_index)

    print("Training complete!")
