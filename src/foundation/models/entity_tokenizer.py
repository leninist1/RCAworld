"""Entity tokenizer and metric encoder for heterogeneous entities.

Projects typed entity observations with different feature dimensions into a
common latent space. Uses shared encoders with observation masks and type
embeddings for JAX-compatible heterogeneous entity processing.

Design:
  - All entities are padded to max_obs_dim across the episode.
  - An observation mask indicates which features are valid per entity.
  - Type embeddings make entity semantics distinguishable at the latent level.
  - A single shared MLP encoder projects observations to common_dim.
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class MetricEncoder(nn.Module):
    """Encodes typed entity observations into a common latent space.

    All entities share the same encoder weights. Type differentiation is
    achieved through learnable type embeddings added after encoding.
    Invalid features (from padding) are zeroed out via obs_mask.

    Attributes:
        common_dim: Output embedding dimension.
        max_obs_dim: Maximum observation dimension across all entity types.
        num_entity_types: Number of distinct entity types in the vocabulary.
        hidden_dims: Hidden layer dimensions for the encoder MLP.
    """

    common_dim: int = 128
    max_obs_dim: int = 16
    num_entity_types: int = 10
    hidden_dims: Tuple[int, ...] = (128,)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 type_idx: jnp.ndarray,
                 obs_mask: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """Encode entity observations.

        Args:
            x: [..., max_obs_dim] observation tensor.
               Can be [B,T,N,D] or any leading batch dimensions.
            type_idx: [N] integer indices into type embedding table.
            obs_mask: [N, max_obs_dim] or broadcastable. 1=valid, 0=invalid.

        Returns:
            encoded: [..., common_dim] entity embeddings.
        """
        # Zero out invalid (padded) features
        if obs_mask is not None:
            # obs_mask: [N, D] -> expand leading dims to match x
            ndim_extra = x.ndim - obs_mask.ndim - 1
            for _ in range(ndim_extra):
                obs_mask = obs_mask[None, ...]
            # obs_mask now: [..., N, D] or broadcastable
            x = x * obs_mask

        # Shared MLP encoder
        for i, hdim in enumerate(self.hidden_dims):
            x = nn.Dense(hdim, name=f"enc_{i}")(x)
            x = nn.elu(x)
        x = nn.Dense(self.common_dim, name="enc_out")(x)

        # Add type embeddings
        type_emb = self.param(
            "type_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_entity_types, self.common_dim),
            jnp.float32,
        )
        # type_idx: [N] -> type_emb[type_idx]: [N, common_dim]
        type_add = type_emb[type_idx]  # [N, common_dim]
        # Expand to match x: [..., N, common_dim]
        ndim = x.ndim - 2  # extra dims beyond N and common_dim
        for _ in range(ndim):
            type_add = type_add[None, ...]
        x = x + type_add

        return x


class TypeAwareDecoder(nn.Module):
    """Type-specific decoder for predicting entity observations.

    Takes latent state and produces predictions in the original observation
    space of each entity type. Uses type-specific output heads.

    Attributes:
        common_dim: Input latent dimension.
        max_obs_dim: Maximum output feature dimension.
        num_entity_types: Number of entity types.
    """

    common_dim: int = 128
    max_obs_dim: int = 16
    num_entity_types: int = 10

    @nn.compact
    def __call__(self,
                 z: jnp.ndarray,
                 type_idx: jnp.ndarray,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode latent state to observation predictions.

        Args:
            z: [..., common_dim] latent state.
            type_idx: [N] entity type indices.

        Returns:
            mu: [..., max_obs_dim] predicted mean.
            log_sigma: [..., max_obs_dim] predicted log standard deviation.
        """
        # Type-conditioned hidden layer
        type_emb = self.param(
            "dec_type_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_entity_types, self.common_dim),
            jnp.float32,
        )
        type_add = type_emb[type_idx]
        ndim = z.ndim - 2
        for _ in range(ndim):
            type_add = type_add[None, ...]
        h = nn.elu(nn.Dense(self.common_dim, name="dec_hidden")(z + type_add))

        mu = nn.Dense(self.max_obs_dim, name="dec_mu")(h)
        log_sigma = nn.Dense(self.max_obs_dim, name="dec_logsigma")(h)
        return mu, log_sigma


class EntityTokenizer(nn.Module):
    """Unified encoder-decoder for heterogeneous entity observations.

    Wraps MetricEncoder and TypeAwareDecoder for convenience.

    Attributes:
        common_dim: Latent embedding dimension.
        max_obs_dim: Maximum feature dimension across all entity types.
        num_entity_types: Number of distinct entity types.
    """

    common_dim: int = 128
    max_obs_dim: int = 16
    num_entity_types: int = 10

    def encode(self, x, type_idx, obs_mask=None):
        """Encode entity observations to common latent space."""
        encoder = MetricEncoder(
            common_dim=self.common_dim,
            max_obs_dim=self.max_obs_dim,
            num_entity_types=self.num_entity_types,
            name="encoder",
        )
        return encoder(x, type_idx, obs_mask)

    def decode(self, z, type_idx):
        """Decode latent to observation predictions."""
        decoder = TypeAwareDecoder(
            common_dim=self.common_dim,
            max_obs_dim=self.max_obs_dim,
            num_entity_types=self.num_entity_types,
            name="decoder",
        )
        return decoder(z, type_idx)
