"""Loss functions for RCAWorld-Foundation training.

Supports multiple training stages:
  - World model pretraining (NLL + KL on normal data)
  - Onset head finetuning (onset time classification)
  - Component head finetuning (entity ranking)
  - Joint training with consistency constraints
"""

from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def gaussian_nll(mu: jnp.ndarray,
                 log_sigma: jnp.ndarray,
                 target: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
    """Gaussian negative log-likelihood.

    Args:
        mu: [..., D] predicted means.
        log_sigma: [..., D] predicted log standard deviations.
        target: [..., D] true values.
        mask: [..., D] or broadcastable feature validity mask.

    Returns:
        nll: [...] scalar per batch element.
    """
    sigma = jnp.exp(log_sigma) + 1e-6
    nll = ((target - mu) ** 2) / (2 * sigma ** 2) + log_sigma + 0.5 * jnp.log(2 * jnp.pi)
    if mask is not None:
        nll = nll * mask
        return jnp.sum(nll) / (jnp.sum(mask) + 1e-8)
    return jnp.mean(nll)


def categorical_kl_divergence(post_logits: jnp.ndarray,
                               prior_logits: jnp.ndarray,
                               stoch_dim: int,
                               stoch_classes: int,
                               free_bits: float = 1.0,
                               ) -> jnp.ndarray:
    """KL divergence between categorical posterior and prior.

    Args:
        post_logits: [B, T, N, stoch_dim, stoch_classes] or [B, N, stoch_dim*stoch_classes]
        prior_logits: same shape as post_logits
        stoch_dim: Number of categorical variables.
        stoch_classes: Number of categories per variable.
        free_bits: Minimum KL per latent dimension.

    Returns:
        kl: scalar mean KL divergence.
    """
    if post_logits.ndim == 5:
        # Already [B, T, N, S, C]
        post_probs = jax.nn.softmax(post_logits, axis=-1)
        prior_probs = jax.nn.softmax(prior_logits, axis=-1)
        log_post = jax.nn.log_softmax(post_logits, axis=-1)
        log_prior = jax.nn.log_softmax(prior_logits, axis=-1)
    elif post_logits.ndim == 3:
        # [B, N, S*C] -> [B, N, S, C]
        B, N = post_logits.shape[0], post_logits.shape[1]
        post_logits = post_logits.reshape(B, N, stoch_dim, stoch_classes)
        prior_logits = prior_logits.reshape(B, N, stoch_dim, stoch_classes)
        post_probs = jax.nn.softmax(post_logits, axis=-1)
        prior_probs = jax.nn.softmax(prior_logits, axis=-1)
        log_post = jax.nn.log_softmax(post_logits, axis=-1)
        log_prior = jax.nn.log_softmax(prior_logits, axis=-1)
    else:
        raise ValueError(f"Unexpected logits shape: {post_logits.shape}")

    kl = jnp.sum(post_probs * (log_post - log_prior), axis=-1)
    kl = jnp.maximum(kl, free_bits)
    return jnp.mean(kl)


def onset_loss(onset_scores: jnp.ndarray,
               true_onset_times: jnp.ndarray,
               onset_masks: jnp.ndarray,
               window_radius: int = 3,
               ) -> jnp.ndarray:
    """Onset detection loss.

    Penalizes the model when onset scores are high far from the true onset
    time, or low near the true onset time.

    Args:
        onset_scores: [B, T, N] predicted onset probabilities.
        true_onset_times: [B, N] integer timestep indices of true onset.
        onset_masks: [B, N] 1.0 if label is available, 0.0 otherwise.
        window_radius: Tolerance window around true onset for soft labels.

    Returns:
        loss: scalar mean onset loss.
    """
    B, T, N = onset_scores.shape

    # Create soft target labels around true onset times
    t_idx = jnp.arange(T)[None, :, None]  # [1, T, 1]
    true_t = true_onset_times[:, None, :]  # [B, 1, N]
    distance = jnp.abs(t_idx - true_t)  # [B, T, N]
    soft_target = jnp.exp(-distance ** 2 / (2 * window_radius ** 2))  # [B, T, N]

    # Mask out unlabeled entities
    soft_target = soft_target * onset_masks[:, None, :]

    # Binary cross-entropy between onset_scores and soft_target
    eps = 1e-7
    bce = -(soft_target * jnp.log(onset_scores + eps) +
            (1 - soft_target) * jnp.log(1 - onset_scores + eps))
    bce = jnp.mean(bce, axis=1)  # [B, N]
    bce = bce * onset_masks  # [B, N]
    return jnp.sum(bce) / (jnp.sum(onset_masks) + 1e-8)


def component_ranking_loss(component_scores: jnp.ndarray,
                            true_component_idx: jnp.ndarray,
                            component_masks: jnp.ndarray,
                            margin: float = 0.1,
                            ) -> jnp.ndarray:
    """Pairwise ranking loss for root cause component localization.

    Encourages the true component to score higher than all others.

    Args:
        component_scores: [B, N] per-entity component scores.
        true_component_idx: [B] index of the true root cause entity.
        component_masks: [B] 1.0 if label is available.
        margin: Minimum score gap between true and false components.

    Returns:
        loss: scalar mean ranking loss.
    """
    B, N = component_scores.shape

    # Gather true component score
    true_scores = component_scores[jnp.arange(B), true_component_idx]  # [B]

    # Pairwise margin: max(0, margin + score_false - score_true)
    score_diff = true_scores[:, None] - component_scores  # [B, N]
    loss_per_pair = jax.nn.relu(margin - score_diff)  # [B, N]

    # Don't penalize the true component itself
    not_true = jnp.ones((B, N), dtype=jnp.float32)
    not_true = not_true.at[jnp.arange(B), true_component_idx].set(0.0)
    loss_per_pair = loss_per_pair * not_true

    loss = jnp.mean(loss_per_pair, axis=1)  # [B]
    loss = loss * component_masks  # [B]
    return jnp.sum(loss) / (jnp.sum(component_masks) + 1e-8)


def consistency_loss(p_c_t: jnp.ndarray,
                      p_c: jnp.ndarray,
                      alpha: jnp.ndarray,
                      ) -> jnp.ndarray:
    """Consistency constraint between p(c|t) and p(c).

    Penalizes large divergence between the two paths' predictions,
    encouraging the model to learn coherent representations.

    Args:
        p_c_t: [B, T, N] time-conditional component scores.
        p_c: [B, N] time-free component scores.
        alpha: [B] current fusion weight.

    Returns:
        loss: scalar consistency loss.
    """
    # Average p_c_t over time, compare to p_c
    p_c_t_avg = jnp.mean(p_c_t, axis=1)  # [B, N]
    # KL-like divergence
    eps = 1e-7
    divergence = (p_c_t_avg * jnp.log((p_c_t_avg + eps) / (p_c + eps)) +
                  p_c * jnp.log((p_c + eps) / (p_c_t_avg + eps)))
    return jnp.mean(jnp.abs(divergence)) * 0.1


def world_model_loss(outputs: Dict,
                     targets: Dict,
                     obs_mask: Optional[jnp.ndarray] = None,
                     kl_weight: float = 0.1,
                     onset_weight: float = 1.0,
                     component_weight: float = 1.0,
                     consistency_weight: float = 0.01,
                     ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Combined world model loss.

    Args:
        outputs: Dict from RCAWorldFoundation.__call__().
        targets: Dict with 'obs' (observations), and optionally
                 'onset_times', 'onset_masks', 'true_component_idx',
                 'component_masks'.
        obs_mask: [N, D] valid feature mask.
        kl_weight: Weight for KL divergence.
        onset_weight: Weight for onset loss.
        component_weight: Weight for component ranking loss.
        consistency_weight: Weight for consistency loss.

    Returns:
        total_loss: scalar.
        loss_components: Dict of individual loss terms.
    """
    losses = {}

    # 1. Prediction loss (Gaussian NLL of next-step observations)
    pred_mu = outputs["pred_mu"]          # [B, T, N, D]
    pred_logsigma = outputs["pred_logsigma"]
    obs_true = targets["obs"]             # [B, T, N, D]
    losses["nll"] = gaussian_nll(pred_mu, pred_logsigma, obs_true, obs_mask)

    # 2. KL divergence (prior vs posterior)
    losses["kl"] = categorical_kl_divergence(
        outputs["post_logits"],
        outputs["prior_logits"],
        outputs["z_seq"].shape[3],   # stoch_dim
        outputs["z_seq"].shape[4],   # stoch_classes
        free_bits=1.0,
    )

    total = losses["nll"] + kl_weight * losses["kl"]

    # 3. Onset loss (if labels available and onset head active)
    if ("onset" in outputs and
            "onset_times" in targets and
            targets.get("onset_times") is not None):
        losses["onset"] = onset_loss(
            outputs["onset"]["onset_score"],
            targets["onset_times"],
            targets.get("onset_masks", jnp.ones_like(
                targets["onset_times"])),
        )
        total = total + onset_weight * losses["onset"]

    # 4. Component ranking loss
    if ("component" in outputs and
            "true_component_idx" in targets and
            targets.get("true_component_idx") is not None):
        losses["component"] = component_ranking_loss(
            outputs["component"]["component_score"],
            targets["true_component_idx"],
            targets.get("component_masks",
                        jnp.ones(targets["true_component_idx"].shape[0])),
        )
        total = total + component_weight * losses["component"]

    # 5. Consistency loss between dual paths
    if ("component" in outputs and
            "p_c_given_t" in outputs["component"] and
            "p_c_marginal" in outputs["component"]):
        losses["consistency"] = consistency_loss(
            outputs["component"]["p_c_given_t"],
            outputs["component"]["p_c_marginal"],
            outputs["component"]["alpha"],
        )
        total = total + consistency_weight * losses["consistency"]

    return total, losses
