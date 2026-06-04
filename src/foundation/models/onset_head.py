"""Onset Head: Learned anomaly onset time detection.

Replaces the fixed oracle-window preprocessing with a learnable module that
predicts when the anomaly started for each entity. Combines three signals:

1. Forecast residual: how poorly the RSSM predicts future observations.
2. Latent state shift: how much the internal state deviates from normal.
3. Propagation precedence: whether this entity's anomaly precedes others.

Output: p(onset_time = t | entity, observations)
"""

from typing import Optional, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn


class OnsetHead(nn.Module):
    """Learned anomaly onset detection from RSSM states and prediction errors.

    Produces an onset score for each [timestep, entity] pair.

    Attributes:
        hidden_dim: Hidden dimension for the scoring MLP.
        window_radius: Context window radius for temporal convolution.
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

        self.temporal_conv = nn.Conv(
            features=1,
            kernel_size=(2 * self.window_radius + 1,),
            padding="SAME",
            name="temporal_conv",
        )

        self.fusion_weights = self.param(
            "fusion_weights",
            nn.initializers.constant(1.0 / 3.0),
            (3,),
            jnp.float32,
        )

    def compute_forecast_residual(self,
                                   obs_true: jnp.ndarray,
                                   obs_pred_mu: jnp.ndarray,
                                   obs_pred_logsigma: jnp.ndarray,
                                   ) -> jnp.ndarray:
        """Gaussian NLL per entity per timestep.

        Args:
            obs_true: [B, T, N, D]
            obs_pred_mu: [B, T, N, D]
            obs_pred_logsigma: [B, T, N, D]

        Returns:
            residual: [B, T, N]
        """
        sigma = jnp.exp(obs_pred_logsigma) + 1e-6
        nll = ((obs_true - obs_pred_mu) ** 2) / (2 * sigma ** 2) + obs_pred_logsigma
        return jnp.mean(nll, axis=-1)

    def compute_latent_shift(self, h_seq: jnp.ndarray) -> jnp.ndarray:
        """L2 norm of consecutive state differences.

        If h_seq has T entries, returns T entries (first is zero-padded).

        Args:
            h_seq: [B, T, N, det_dim]

        Returns:
            shift: [B, T, N]
        """
        diff = h_seq[:, 1:, :, :] - h_seq[:, :-1, :, :]
        shift_diff = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # [B, T-1, N]
        # Pad first timestep with zero
        zero_first = jnp.zeros((shift_diff.shape[0], 1, shift_diff.shape[2]),
                               dtype=shift_diff.dtype)
        return jnp.concatenate([zero_first, shift_diff], axis=1)  # [B, T, N]

    def compute_propagation_precedence(self,
                                        residual: jnp.ndarray,
                                        edge_index: Optional[jnp.ndarray] = None,
                                        ) -> jnp.ndarray:
        """Propagation precedence: entities that deviate early get higher score.

        Two components:
        a) Temporal earliness: higher score at earlier timesteps when residual is high.
        b) Cross-entity contrast: entity whose residual rises before others.

        Args:
            residual: [B, T, N]
            edge_index: [2, E] optional directed edges.

        Returns:
            precedence: [B, T, N]
        """
        B, T, N = residual.shape

        # Temporal earliness: earlier timesteps weighted higher
        time_decay = 1.0 / (1.0 + 0.1 * jnp.arange(T))  # [T] declining
        time_decay = time_decay[None, :, None]  # [1, T, 1]

        # Cross-entity contrast: how much this entity exceeds the median
        median_res = jnp.median(residual, axis=-1, keepdims=True)  # [B, T, 1]
        excess = jax.nn.relu(residual - median_res)  # [B, T, N]

        # Combine: early + excessive = likely root cause
        precedence = excess * time_decay

        # Normalize per-batch
        precedence = precedence / (jnp.max(precedence, axis=(1, 2), keepdims=True) + 1e-8)

        return precedence

    def __call__(self,
                 residual: jnp.ndarray,
                 h_seq: jnp.ndarray,
                 edge_index: Optional[jnp.ndarray] = None,
                 ) -> Dict[str, jnp.ndarray]:
        """Compute onset scores.

        Args:
            residual: [B, T, N] forecast residuals.
            h_seq: [B, T+1, N, det_dim] RSSM deterministic states.
            edge_index: [2, E] optional relation edges.

        Returns:
            Dict with onset_score, residual_score, shift_score, precedence_score.
        """
        # 1. Residual component
        residual_score = self.residual_proj(
            residual[..., None])[..., 0]  # [B, T, N]
        residual_score = jnp.tanh(residual_score)

        # 2. Latent shift component
        shift = self.compute_latent_shift(h_seq)  # [B, T, N]
        shift_score = self.shift_proj(shift[..., None])[..., 0]
        shift_score = jnp.tanh(shift_score)

        # 3. Propagation precedence
        precedence_score = self.compute_propagation_precedence(
            residual, edge_index)  # [B, T, N]

        # 4. Temporal smoothing per entity
        # [B, T, N, 3] -> [B, N, T, 3]
        combined = jnp.stack(
            [residual_score, shift_score, precedence_score], axis=-1)
        combined_t = jnp.transpose(combined, (0, 2, 1, 3))  # [B, N, T, 3]
        smoothed = jnp.squeeze(
            self.temporal_conv(combined_t), axis=-1)  # [B, N, T]
        smoothed = jnp.transpose(smoothed, (0, 2, 1))    # [B, T, N]

        # 5. Weighted fusion
        w = jax.nn.softmax(self.fusion_weights)
        onset_score = (
            w[0] * residual_score +
            w[1] * shift_score +
            w[2] * precedence_score
        )
        onset_score = jax.nn.sigmoid(onset_score)

        return {
            "onset_score": onset_score,
            "residual_score": residual_score,
            "shift_score": shift_score,
            "precedence_score": precedence_score,
        }
