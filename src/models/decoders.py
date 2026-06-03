"""Decoders for the Graph-RSSM world model.

- MetricsDecoder: predicts future metrics (μ, σ) from latent states.
- EdgeDecoder: predicts future edge features from latent state pairs.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class MetricsDecoder(nn.Module):
    """Predicts future node metrics from latent state.

    Input:  h [batch, N, D_det], z [batch, N, D_stoch, C]
    Output: mu [batch, N, D_node], log_sigma [batch, N, D_node]
    """
    hidden_dim: int = 256
    node_dim: int = 6  # N_METRICS

    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(self.hidden_dim),
            nn.elu,
        ])
        self.mu_head = nn.Dense(self.node_dim)
        self.log_sigma_head = nn.Dense(self.node_dim)

    def __call__(
        self, h: jnp.ndarray, z: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            h: [batch, N, D_det]
            z: [batch, N, D_stoch, C] one-hot
        Returns:
            mu: [batch, N, D_node]
            log_sigma: [batch, N, D_node]
        """
        batch, N = h.shape[0], h.shape[1]
        z_flat = z.reshape(batch, N, -1)  # [batch, N, D_stoch * C]
        feat = jnp.concatenate([h, z_flat], axis=-1)
        feat = self.net(feat)
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat)
        # Bound log_sigma for stability
        log_sigma = jnp.clip(log_sigma, -5.0, 5.0)
        return mu, log_sigma


class EdgeDecoder(nn.Module):
    """Predicts future edge features from node latent state pairs.

    Input:  h [batch, N, D_det], z [batch, N, D_stoch, C], edge_index [2, E]
    Output: predicted edge features [batch, E, D_edge]
    """
    hidden_dim: int = 128
    edge_dim: int = 3  # N_EDGE_FEATURES

    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(self.hidden_dim),
            nn.elu,
        ])
        self.pred_head = nn.Dense(self.edge_dim)

    def __call__(
        self,
        h: jnp.ndarray,
        z: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            h: [batch, N, D_det]
            z: [batch, N, D_stoch, C]
            edge_index: [2, E]
        Returns:
            edge_pred: [batch, E, D_edge]
        """
        batch, N = h.shape[0], h.shape[1]
        E = edge_index.shape[1]
        z_flat = z.reshape(batch, N, -1)
        feat = jnp.concatenate([h, z_flat], axis=-1)  # [batch, N, D_total]

        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]

        feat_src = feat[:, src_idx, :]  # [batch, E, D_total]
        feat_dst = feat[:, dst_idx, :]  # [batch, E, D_total]

        edge_feat = jnp.concatenate([feat_src, feat_dst], axis=-1)  # [batch, E, 2*D_total]
        edge_feat = self.net(edge_feat)
        edge_pred = self.pred_head(edge_feat)
        return edge_pred
