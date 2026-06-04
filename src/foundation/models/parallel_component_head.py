"""Parallel Component Head: p(c|t) and p(c) dual-path root cause localization.

Instead of strict cascade p(t) -> p(c|t), this module computes two parallel
paths and fuses them:

  Path A - Time-conditional:  p(c | t, state_t)
    How likely entity c is the root cause given onset time t.
    Uses attention over entities at each candidate time.

  Path B - Time-free (marginal):  p(c | all_states)
    How likely entity c is the root cause regardless of onset time.
    Uses entity-wise residual aggregation and state deviation.

  Fusion:  score_c = alpha * p_c_given_t + (1-alpha) * p_c_marginal

This design handles both abrupt faults (where time-conditional signal is
strong) and silent/accumulating faults (where time-free signal dominates).
"""

from typing import Optional, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn


class ParallelComponentHead(nn.Module):
    """Dual-path root cause entity scoring.

    Attributes:
        hidden_dim: Feature dimension for scoring networks.
        num_attention_heads: Number of heads for time-conditional attention.
        use_graph: Whether to incorporate graph structure in scoring.
        alpha_init: Initial value for the fusion gate (0.5 = equal weight).
    """

    hidden_dim: int = 128
    num_attention_heads: int = 4
    use_graph: bool = False
    alpha_init: float = 0.5

    def setup(self):
        # Path A: Time-conditional scoring p(c|t)
        # Query: onset candidate embedding
        # Key/Value: entity states at time t
        self.q_proj = nn.Dense(self.hidden_dim, name="q_proj")
        self.k_proj = nn.Dense(self.hidden_dim, name="k_proj")
        self.v_proj = nn.Dense(self.hidden_dim, name="v_proj")

        # Path B: Time-free scoring p(c)
        self.entity_encoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(self.hidden_dim),
        ])

        # Fusion gate (explicit Dense layers to avoid nn.Sequential+elu issues)
        self.fusion_dense1 = nn.Dense(self.hidden_dim, name="fusion_dense1")
        self.fusion_dense2 = nn.Dense(1, name="fusion_dense2")

        # Final scoring projection
        self.score_proj = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.elu,
            nn.Dense(1),
        ])

    def _compute_p_c_given_t(self,
                              h_seq: jnp.ndarray,
                              onset_scores: jnp.ndarray,
                              ) -> jnp.ndarray:
        """Path A: p(c | t) scoring via cross-entity attention.

        For each candidate onset time, compute attention over entities to
        identify which entity is most anomalous at that time.

        Args:
            h_seq: [B, T, N, det_dim] deterministic RSSM states.
            onset_scores: [B, T, N] onset likelihood from OnsetHead.

        Returns:
            p_c_given_t: [B, T, N] time-conditional entity scores.
        """
        B, T, N, D = h_seq.shape

        # Use h as key/value, onset-weighted query over time
        # Aggregate onset-weighted state
        onset_weighted = h_seq * onset_scores[..., None]  # [B, T, N, D]
        query = jnp.mean(onset_weighted, axis=1)  # [B, N, D]  aggregate over time

        # Key: each entity's state averaged over time
        key = jnp.mean(h_seq, axis=1)  # [B, N, D]
        value = h_seq  # [B, T, N, D]  per-timestep details

        # Compute attention scores
        q = self.q_proj(query)  # [B, N, H]
        k = self.k_proj(key)    # [B, N, H]
        v = self.v_proj(value)  # [B, T, N, H]

        # Attention: q @ k^T gives [B, N, N] entity-entity scores
        attn_logits = jnp.einsum("bnh,bmh->bnm", q, k) / jnp.sqrt(self.hidden_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # [B, N, N]

        # Weighted combination over entities -> per-entity
        entity_context = jnp.einsum("bnm,btmh->btnh", attn_weights, v)  # [B, T, N, H]

        # Score from context
        p_c_given_t = self.score_proj(entity_context)[..., 0]  # [B, T, N]
        p_c_given_t = jax.nn.sigmoid(p_c_given_t)

        return p_c_given_t

    def _compute_p_c_marginal(self,
                               h_seq: jnp.ndarray,
                               residual: jnp.ndarray,
                               onset_scores: jnp.ndarray,
                               ) -> jnp.ndarray:
        """Path B: p(c) time-free entity scoring.

        Aggregates per-entity signals across the entire time window:
        - Average residual magnitude
        - Total state deviation
        - Peak onset score

        Args:
            h_seq: [B, T, N, det_dim]
            residual: [B, T, N]
            onset_scores: [B, T, N]

        Returns:
            p_c_marginal: [B, N] time-free entity scores.
        """
        B, T, N, D = h_seq.shape

        # Aggregate features per entity
        avg_residual = jnp.mean(residual, axis=1)     # [B, N]
        max_residual = jnp.max(residual, axis=1)       # [B, N]
        avg_onset = jnp.mean(onset_scores, axis=1)     # [B, N]
        max_onset = jnp.max(onset_scores, axis=1)      # [B, N]
        state_mean = jnp.mean(h_seq, axis=1)           # [B, N, D]
        state_std = jnp.std(h_seq, axis=1)             # [B, N, D]

        # Concatenate all entity-level features
        feats = jnp.concatenate([
            avg_residual[..., None],
            max_residual[..., None],
            avg_onset[..., None],
            max_onset[..., None],
            state_mean,
            state_std,
        ], axis=-1)  # [B, N, 4 + 2*D]

        # Encode
        entity_emb = self.entity_encoder(feats)        # [B, N, H]

        # Score
        p_c_marginal = self.score_proj(entity_emb)[..., 0]  # [B, N]
        p_c_marginal = jax.nn.sigmoid(p_c_marginal)

        return p_c_marginal

    def __call__(self,
                 h_seq: jnp.ndarray,
                 residual: jnp.ndarray,
                 onset_scores: jnp.ndarray,
                 edge_index: Optional[jnp.ndarray] = None,
                 ) -> Dict[str, jnp.ndarray]:
        """Compute dual-path component scores.

        Args:
            h_seq: [B, T, N, det_dim] RSSM deterministic states.
            residual: [B, T, N] forecast residuals.
            onset_scores: [B, T, N] onset likelihood.
            edge_index: [2, E] optional relation edges.

        Returns:
            Dict with:
                component_score: [B, N] final entity root cause scores.
                p_c_given_t: [B, T, N] time-conditional scores.
                p_c_marginal: [B, N] time-free scores.
                alpha: [B] fusion weights.
        """
        B, T, N = residual.shape

        # Path A: Time-conditional
        p_c_given_t = self._compute_p_c_given_t(h_seq, onset_scores)

        # Path B: Time-free
        p_c_marginal = self._compute_p_c_marginal(h_seq, residual, onset_scores)

        # Fusion gate: per-batch learnable alpha
        # Use residual statistics to determine which path to trust more
        gate_input = jnp.stack([
            jnp.mean(residual, axis=(1, 2)),   # [B]
            jnp.max(residual, axis=(1, 2)),    # [B]
            jnp.std(residual, axis=(1, 2)),    # [B]
        ], axis=-1)  # [B, 3]

        alpha_logit = self.fusion_dense2(
            nn.elu(self.fusion_dense1(gate_input)))  # [B, 1]
        alpha_logit = jnp.squeeze(alpha_logit, axis=-1)  # [B] or []
        if alpha_logit.ndim == 0:
            alpha_logit = alpha_logit[None]  # ensure [B]
        alpha = jax.nn.sigmoid(alpha_logit)  # [B]

        # Fuse
        # p_c_given_t: [B, T, N] -> time-weighted average -> [B, N]
        # Weight by onset_scores to emphasize likely onset times
        onset_weights = jax.nn.softmax(
            jnp.sum(onset_scores, axis=-1), axis=-1)  # [B, T] normalize over time
        p_c_t_pooled = jnp.einsum("bt,btn->bn", onset_weights, p_c_given_t)  # [B, N]

        component_score = (
            alpha[:, None] * p_c_t_pooled +
            (1.0 - alpha[:, None]) * p_c_marginal
        )

        return {
            "component_score": component_score,           # [B, N]
            "p_c_given_t": p_c_given_t,                   # [B, T, N]
            "p_c_marginal": p_c_marginal,                  # [B, N]
            "alpha": alpha,                                # [B]
        }
