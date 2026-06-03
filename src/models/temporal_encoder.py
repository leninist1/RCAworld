"""Temporal encoder: shared-parameter GRU over per-service metric windows."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder shared across all services.

    Input:  node_features [batch, window, N, D_node]
    Output: local_embeddings [batch, N, D_hidden]

    A single GRU processes each service's time series independently,
    with weights shared across all services.
    """
    hidden_dim: int = 128
    num_layers: int = 2

    def setup(self):
        self.gru_cells = [
            nn.GRUCell(features=self.hidden_dim) for _ in range(self.num_layers)
        ]

    def __call__(self, node_features: jnp.ndarray) -> jnp.ndarray:
        # node_features: [batch, window, N, D_node]
        batch, window, N, D_node = node_features.shape

        # Transpose to [N, batch, window, D_node] for per-service processing
        x = jnp.transpose(node_features, (2, 0, 1, 3))  # [N, batch, window, D_node]

        # vmap over services: each service gets its own GRU pass
        def process_service(service_seq):
            # service_seq: [batch, window, D_node]
            carry = self.initialize_carry(batch)
            for layer_idx in range(self.num_layers):
                # Scan over time dimension
                def gru_step(carry, xt):
                    new_carry, out = self.gru_cells[layer_idx](carry, xt)
                    return new_carry, out

                carry, outputs = jax.lax.scan(gru_step, carry, service_seq)
                # outputs: [window, batch, hidden_dim]
                # Carry forward last output to next layer
                service_seq = jnp.transpose(outputs, (1, 0, 2))  # [batch, window, hidden_dim]
            # Return last timestep output
            return service_seq[:, -1, :]  # [batch, hidden_dim]

        service_embeds = jax.vmap(process_service)(x)  # [N, batch, hidden_dim]
        return jnp.transpose(service_embeds, (1, 0, 2))  # [batch, N, hidden_dim]

    def initialize_carry(self, batch: int) -> jnp.ndarray:
        return jnp.zeros((batch, self.hidden_dim))


class TemporalEncoderSimple(nn.Module):
    """Simpler single-layer GRU temporal encoder. Faster for Phase 0."""
    hidden_dim: int = 128

    def setup(self):
        self.gru = nn.GRUCell(features=self.hidden_dim)

    def __call__(self, node_features: jnp.ndarray) -> jnp.ndarray:
        # node_features: [batch, window, N, D_node]
        batch, window, N, D_node = node_features.shape
        x = jnp.transpose(node_features, (2, 0, 1, 3))  # [N, batch, window, D_node]

        def process_service(service_seq):
            # service_seq: [batch, window, D_node]
            carry = jnp.zeros((batch, self.hidden_dim))

            def gru_step(carry, xt):
                new_carry, _ = self.gru(carry, xt)
                return new_carry, new_carry

            _, outputs = jax.lax.scan(gru_step, carry, service_seq)
            # outputs: [window, batch, hidden_dim]
            outputs = jnp.transpose(outputs, (1, 0, 2))  # [batch, window, hidden_dim]
            return outputs[:, -1, :]  # [batch, hidden_dim] - last timestep

        service_embeds = jax.vmap(process_service)(x)  # [N, batch, hidden_dim]
        return jnp.transpose(service_embeds, (1, 0, 2))  # [batch, N, hidden_dim]
