"""Graph encoder: GAT (Graph Attention Network) for service interaction modeling.

Learns how faults/state changes propagate through the microservice dependency graph.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


class GATLayer(nn.Module):
    """Single GAT layer with edge feature incorporation.

    Args:
        out_dim: Output feature dimension per head.
        num_heads: Number of attention heads.
        dropout_rate: Attention dropout rate.
    """
    out_dim: int = 64
    num_heads: int = 4
    dropout_rate: float = 0.0

    def setup(self):
        total_dim = self.out_dim * self.num_heads
        self.W_src = nn.Dense(total_dim, use_bias=False)
        self.W_dst = nn.Dense(total_dim, use_bias=False)
        self.W_edge = nn.Dense(self.num_heads, use_bias=False)
        self.attn_src = nn.Dense(self.num_heads, use_bias=False)
        self.attn_dst = nn.Dense(self.num_heads, use_bias=False)
        self.leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.2)

    def __call__(
        self,
        node_embeds: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            node_embeds: [batch, N, D_in]
            edge_index: [2, E] (static, same across batch)
            edge_features: [batch, E, D_edge]
            deterministic: if True, no dropout
        Returns:
            node_embeds: [batch, N, out_dim * num_heads]
        """
        batch, N, D_in = node_embeds.shape
        E = edge_features.shape[1]

        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]

        # Linear transforms
        h_src = self.W_src(node_embeds)  # [batch, N, total_dim]
        h_dst = self.W_dst(node_embeds)  # [batch, N, total_dim]

        # Reshape for multi-head: [batch, N, num_heads, out_dim]
        h_src = h_src.reshape(batch, N, self.num_heads, self.out_dim)
        h_dst = h_dst.reshape(batch, N, self.num_heads, self.out_dim)

        # Gather source and destination features per edge
        h_src_e = h_src[:, src_idx, :, :]  # [batch, E, num_heads, out_dim]
        h_dst_e = h_dst[:, dst_idx, :, :]  # [batch, E, num_heads, out_dim]

        # Attention scores
        attn_s = self.attn_src(node_embeds)  # [batch, N, num_heads]
        attn_d = self.attn_dst(node_embeds)  # [batch, N, num_heads]
        attn_s_e = attn_s[:, src_idx, :]  # [batch, E, num_heads]
        attn_d_e = attn_d[:, dst_idx, :]  # [batch, E, num_heads]

        # Edge feature contribution to attention
        edge_attn = self.W_edge(edge_features)  # [batch, E, num_heads]

        # Raw attention: e_ij = LeakyReLU(attn_s_i + attn_d_j + edge_ij)
        e_raw = attn_s_e + attn_d_e + edge_attn  # [batch, E, num_heads]
        e_raw = self.leaky_relu(e_raw)

        # Segment softmax per destination node
        num_segments = N
        e_max_per_dst = jnp.full((batch, N, self.num_heads), -1e9)
        e_max_per_dst = e_max_per_dst.at[:, dst_idx, :].max(e_raw)
        e_shifted = e_raw - e_max_per_dst[:, dst_idx, :]
        e_exp = jnp.exp(e_shifted)  # [batch, E, num_heads]

        alpha_sum = jnp.zeros((batch, N, self.num_heads))
        alpha_sum = alpha_sum.at[:, dst_idx, :].add(e_exp)

        alpha = e_exp / (alpha_sum[:, dst_idx, :] + 1e-8)  # [batch, E, num_heads]

        # Apply dropout to attention
        if not deterministic and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            rng = self.make_rng('dropout')
            alpha = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
                alpha, deterministic=deterministic
            )

        # Message aggregation: sum over incoming edges per destination
        # Weighted message = alpha * h_src
        msg = alpha[:, :, :, jnp.newaxis] * h_src_e  # [batch, E, num_heads, out_dim]

        # Scatter sum to destination nodes
        h_out = jnp.zeros((batch, N, self.num_heads, self.out_dim))
        h_out = h_out.at[:, dst_idx, :, :].add(msg)

        # Flatten heads
        h_out = h_out.reshape(batch, N, self.num_heads * self.out_dim)
        return h_out


class GATEncoder(nn.Module):
    """2-layer GAT encoder for service interaction modeling.

    Input:
        node_embeds: [batch, N, D_temporal]
        edge_index: [2, E]
        edge_features: [batch, E, D_edge]
    Output:
        graph_embeds: [batch, N, D_graph]
    """
    hidden_dim: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        self.gat1 = GATLayer(
            out_dim=self.hidden_dim // self.num_heads,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.gat2 = GATLayer(
            out_dim=self.hidden_dim // self.num_heads,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.out_proj = nn.Dense(self.hidden_dim)

    def __call__(
        self,
        node_embeds: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            node_embeds: [batch, N, D_temporal]
            edge_index: [2, E]
            edge_features: [batch, E, D_edge]
            deterministic: bool
        Returns:
            graph_embeds: [batch, N, hidden_dim]
        """
        # First GAT layer
        h1 = self.gat1(node_embeds, edge_index, edge_features, deterministic)
        h1 = self.norm1(h1 + self.out_proj(node_embeds))  # residual + norm
        h1 = jax.nn.elu(h1)

        # Second GAT layer
        h2 = self.gat2(h1, edge_index, edge_features, deterministic)
        h2 = self.norm2(h2 + self.out_proj(h1))

        return h2


class GATEncoderSimple(nn.Module):
    """Single-layer GAT encoder for Phase 0 prototyping. Faster training."""
    hidden_dim: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.0

    def setup(self):
        self.gat = GATLayer(
            out_dim=self.hidden_dim // self.num_heads,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.hidden_dim)

    def __call__(
        self,
        node_embeds: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_features: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        h = self.gat(node_embeds, edge_index, edge_features, deterministic)
        h = self.norm(h + self.proj(node_embeds))
        return h
