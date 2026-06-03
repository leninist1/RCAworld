"""Full Graph-RSSM world model for microservice dynamics.

Assembles:
- GATEncoder: per-timestep graph message passing
- RSSM: latent state dynamics
- MetricsDecoder + EdgeDecoder: future prediction heads
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Tuple, Dict, Optional
import optax

from .rssm import RSSM, RSSMState
from .graph_encoder import GATEncoderSimple
from .decoders import MetricsDecoder, EdgeDecoder


class GraphRSSM(nn.Module):
    """Graph-RSSM world model for microservice system dynamics.

    Input:
        node_features: [batch, T, N, D_node]
        edge_features: [batch, T, E, D_edge]
        edge_index: [2, E]
        workload_context: [batch, T, D_ctx]

    Output:
        predictions for future node metrics and edge features
    """
    node_dim: int = 6
    edge_dim: int = 3
    ctx_dim: int = 3

    # Encoder
    graph_hidden: int = 128
    num_heads: int = 4

    # RSSM
    det_dim: int = 256
    stoch_dim: int = 32
    stoch_classes: int = 32

    # Decoder
    dec_hidden: int = 256

    def setup(self):
        # Per-timestep node encoder (simple MLP, temporal GRU added later)
        self.node_encoder = nn.Sequential([
            nn.Dense(self.graph_hidden),
            nn.elu,
            nn.Dense(self.graph_hidden),
        ])

        # Edge encoder
        self.edge_encoder = nn.Sequential([
            nn.Dense(self.graph_hidden // 2),
            nn.elu,
            nn.Dense(self.graph_hidden // 2),
        ])

        # GAT graph encoder
        self.gat = GATEncoderSimple(
            hidden_dim=self.graph_hidden,
            num_heads=self.num_heads,
            dropout_rate=0.0,
        )

        # RSSM latent dynamics
        self.rssm = RSSM(
            det_dim=self.det_dim,
            stoch_dim=self.stoch_dim,
            stoch_classes=self.stoch_classes,
        )

        # Decoder heads
        self.metrics_decoder = MetricsDecoder(
            hidden_dim=self.dec_hidden,
            node_dim=self.node_dim,
        )
        self.edge_decoder = EdgeDecoder(
            hidden_dim=self.dec_hidden // 2,
            edge_dim=self.edge_dim,
        )

    def encode_observations(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Encode observations at each timestep into graph embeddings.

        Args:
            node_features: [batch, T, N, D_node]
            edge_features: [batch, T, E, D_edge]
            edge_index: [2, E]
        Returns:
            graph_embeddings: [batch, T, N, D_graph]
        """
        batch, T, N, _ = node_features.shape
        E = edge_features.shape[2]

        # Flatten time into batch for parallel processing
        nodes_flat = node_features.reshape(batch * T, N, self.node_dim)
        edges_flat = edge_features.reshape(batch * T, E, self.edge_dim)

        # Encode nodes
        node_emb = self.node_encoder(nodes_flat)  # [B*T, N, D_hid]

        # Encode edges
        edge_emb = self.edge_encoder(edges_flat)  # [B*T, E, D_hid/2]

        # GAT message passing (per timestep, in parallel)
        # Need to treat T as batch dim for GAT
        graph_emb_flat = self.gat(node_emb, edge_index, edge_emb, deterministic=True)

        # Reshape back
        graph_emb = graph_emb_flat.reshape(batch, T, N, self.graph_hidden)
        return graph_emb

    def rollout_latents(
        self,
        rng: jax.random.PRNGKey,
        graph_embs: jnp.ndarray,
        workload: jnp.ndarray,
        initial_state: Optional[RSSMState] = None,
        use_posterior: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, RSSMState]:
        """Unroll RSSM over observation embeddings.

        Args:
            rng: PRNG key
            graph_embs: [batch, T, N, D_graph]
            workload: [batch, T, D_ctx]
            initial_state: optional initial state
            use_posterior: use posterior during rollout (True for training)

        Returns:
            h_seq: [batch, T, N, D_det]
            z_seq: [batch, T, N, D_stoch, C]
            prior_seq: [batch, T, N, D_stoch, C]
            post_seq: [batch, T, N, D_stoch, C]
            final_state: RSSMState
        """
        return self.rssm(
            rng=rng,
            observations=graph_embs,
            contexts=workload,
            initial_state=initial_state,
            use_posterior=use_posterior,
        )

    def predict_next(
        self,
        h: jnp.ndarray,
        z: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict next-step observations from latent state.

        Args:
            h: [batch, N, D_det]  deterministic state
            z: [batch, N, D_stoch, C]  stochastic state
            edge_index: [2, E]

        Returns:
            metrics_mu: [batch, N, D_node]
            metrics_logsigma: [batch, N, D_node]
            edge_pred: [batch, E, D_edge]
        """
        metrics_mu, metrics_logsigma = self.metrics_decoder(h, z)
        edge_pred = self.edge_decoder(h, z, edge_index)
        return metrics_mu, metrics_logsigma, edge_pred

    def __call__(
        self,
        rng: jax.random.PRNGKey,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        edge_index: jnp.ndarray,
        workload_context: jnp.ndarray,
        initial_state: Optional[RSSMState] = None,
        use_posterior: bool = True,
    ) -> Dict:
        """Full forward pass.

        Args:
            rng: PRNG key
            node_features: [batch, T, N, D_node]
            edge_features: [batch, T, E, D_edge]
            edge_index: [2, E]
            workload_context: [batch, T, D_ctx]
            initial_state: optional initial RSSMState
            use_posterior: use posterior for RSSM

        Returns:
            dict with:
                h_seq, z_seq, prior_logits, post_logits
                metrics_mu, metrics_logsigma (predictions for t+1..T)
                edge_preds (predictions for t+1..T)
                graph_embs
                final_state
        """
        batch, T, N, _ = node_features.shape

        # 1. Encode observations to graph embeddings
        graph_embs = self.encode_observations(node_features, edge_features, edge_index)

        # 2. Unroll RSSM
        rng, rssm_rng = jax.random.split(rng)
        h_seq, z_seq, prior_seq, post_seq, final_state = self.rollout_latents(
            rssm_rng, graph_embs, workload_context, initial_state, use_posterior
        )

        # 3. Predict next-step observations for each timestep except the last
        # h_seq[t] and z_seq[t] predict observations at t+1
        # We predict for T-1 steps (can't predict beyond sequence)
        def predict_step(h_t, z_t):
            return self.predict_next(h_t, z_t, edge_index)

        # vmapped over time (but only first T-1 steps)
        h_pred = h_seq[:, :-1, :, :]  # [batch, T-1, N, D_det]
        z_pred = z_seq[:, :-1, :, :]  # [batch, T-1, N, D_stoch, C]

        # Reshape for vmapped prediction
        batch_Tminus1 = batch * (T - 1)
        h_flat = h_pred.reshape(batch_Tminus1, N, self.det_dim)
        z_flat = z_pred.reshape(batch_Tminus1, N, self.stoch_dim, self.stoch_classes)

        metrics_mu_flat, metrics_logsigma_flat, edge_preds_flat = self.predict_next(
            h_flat, z_flat, edge_index
        )

        metrics_mu = metrics_mu_flat.reshape(batch, T - 1, N, self.node_dim)
        metrics_logsigma = metrics_logsigma_flat.reshape(batch, T - 1, N, self.node_dim)
        edge_preds = edge_preds_flat.reshape(batch, T - 1, -1, self.edge_dim)

        return {
            "graph_embs": graph_embs,
            "h_seq": h_seq,
            "z_seq": z_seq,
            "prior_logits": prior_seq,
            "post_logits": post_seq,
            "metrics_mu": metrics_mu,
            "metrics_logsigma": metrics_logsigma,
            "edge_preds": edge_preds,
            "final_state": final_state,
        }
