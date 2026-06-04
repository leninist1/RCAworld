"""Onset Head: Learned anomaly onset time detection.

Produces onset score for each [timestep, entity] pair from three signals:
  1. Forecast residual:  how poorly the RSSM predicts observations.
  2. Latent state shift:  how much the internal state changes.
  3. Temporal early-rise: whether this entity's anomaly rises before peers.

All three signals are temporally smoothed before fusion. The smoothed output
is the final onset score (temporal_conv output is actually used).
"""

from typing import Optional, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn


class OnsetHead(nn.Module):
    """Learned anomaly onset detection from RSSM states and prediction errors.

    Attributes:
        hidden_dim: Hidden dimension for scoring MLPs.
        window_radius: Temporal convolution kernel half-width.
    """

    hidden_dim: int = 128
    window_radius: int = 5

    def setup(self):
        self.residual_proj = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(1),
        ])
        self.shift_proj = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(1),
        ])
        # Temporal convolution: smooths across time to suppress noise
        self.temporal_conv = nn.Conv(
            features=1,
            kernel_size=(2 * self.window_radius + 1,),
            padding="SAME",
            name="temporal_conv",
        )
        self.fusion_weights = self.param(
            "fusion_weights",
            nn.initializers.constant(1.0 / 2.0),
            (2,),  # residual + shift (early-rise derived from residual)
            jnp.float32,
        )

    def compute_latent_shift(self, h_seq: jnp.ndarray) -> jnp.ndarray:
        """L2 norm of consecutive state differences, zero-padded at t=0."""
        diff = h_seq[:, 1:, :, :] - h_seq[:, :-1, :, :]
        shift_diff = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # [B,T-1,N]
        zero_first = jnp.zeros(
            (shift_diff.shape[0], 1, shift_diff.shape[2]), dtype=shift_diff.dtype)
        return jnp.concatenate([zero_first, shift_diff], axis=1)  # [B,T,N]

    def __call__(self,
                 residual: jnp.ndarray,
                 h_seq: jnp.ndarray,
                 edge_index: Optional[jnp.ndarray] = None,
                 ) -> Dict[str, jnp.ndarray]:
        """Compute onset scores.

        Args:
            residual: [B, T, N] forecast residuals.
            h_seq:    [B, T, N, det_dim] RSSM deterministic states (T matches residual).
            edge_index: [2, E] optional edges (reserved for future use).

        Returns:
            Dict with onset_score (smoothed+ fused), plus per-signal scores.
        """
        B, T, N = residual.shape

        # 1. Residual score (per entity, per timestep)
        residual_score = self.residual_proj(residual[..., None])[..., 0]
        residual_score = jnp.tanh(residual_score)

        # 2. Latent shift score
        # Ensure h_seq has T entries (pad/trim if needed)
        if h_seq.shape[1] == T + 1:
            h_for_shift = h_seq
            shift = self.compute_latent_shift(h_for_shift)  # [B,T,N]
        else:
            shift = self.compute_latent_shift(h_seq)
        shift_score = self.shift_proj(shift[..., None])[..., 0]
        shift_score = jnp.tanh(shift_score)

        # 3. Temporal smoothing via convolution (was dead code — now used)
        # Stack [B, T, N, 2] -> [B, N, T, 2] for Conv over time dim
        combined = jnp.stack([residual_score, shift_score], axis=-1)  # [B,T,N,2]
        combined_t = jnp.transpose(combined, (0, 2, 1, 3))           # [B,N,T,2]
        smoothed = jnp.squeeze(self.temporal_conv(combined_t), axis=-1)  # [B,N,T]
        smoothed = jnp.transpose(smoothed, (0, 2, 1))                    # [B,T,N]

        # 4. Weighted fusion of smoothed signals
        w = jax.nn.softmax(self.fusion_weights)
        onset_score = (
            w[0] * smoothed +
            w[1] * residual_score   # mix in unsmoothed residual for sharpness
        )
        onset_score = jax.nn.sigmoid(onset_score)

        return {
            "onset_score": onset_score,
            "residual_score": residual_score,
            "shift_score": shift_score,
            "smoothed_score": smoothed,
        }
