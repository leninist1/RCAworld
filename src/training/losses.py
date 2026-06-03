"""Loss functions for Graph-RSSM training.

- Gaussian NLL for metrics prediction
- MSE for edge feature prediction
- KL divergence for RSSM regularization
- Masked reconstruction loss (Phase 0 optional)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple


def gaussian_nll(
    pred_mu: jnp.ndarray,
    pred_logsigma: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian negative log-likelihood loss.

    Args:
        pred_mu: [batch, T, N, D] predicted mean
        pred_logsigma: [batch, T, N, D] predicted log std
        target: [batch, T, N, D] ground truth

    Returns:
        scalar loss (mean over batch, T, N, D)
    """
    sigma = jnp.exp(pred_logsigma) + 1e-4
    nll = ((target - pred_mu) ** 2) / (2 * sigma ** 2) + pred_logsigma + 0.5 * jnp.log(2 * jnp.pi)
    return nll.mean()


def gaussian_nll_per_node(
    pred_mu: jnp.ndarray,
    pred_logsigma: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian NLL per node (service), averaged over features.

    Returns:
        [batch, T, N] array of per-node per-timestep losses
    """
    sigma = jnp.exp(pred_logsigma) + 1e-4
    nll = ((target - pred_mu) ** 2) / (2 * sigma ** 2) + pred_logsigma + 0.5 * jnp.log(2 * jnp.pi)
    # Average over feature dim (axis=-1), keep batch, time, node
    return nll.mean(axis=-1)  # [batch, T, N]


def edge_mse(
    pred_edges: jnp.ndarray,
    target_edges: jnp.ndarray,
) -> jnp.ndarray:
    """MSE loss for edge feature prediction.

    Args:
        pred_edges: [batch, T, E, D_edge]
        target_edges: [batch, T, E, D_edge]

    Returns:
        scalar loss
    """
    return ((pred_edges - target_edges) ** 2).mean()


def kl_categorical(
    post_logits: jnp.ndarray,
    prior_logits: jnp.ndarray,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """KL divergence between two categorical distributions.

    KL(post || prior) = sum_i post_i * (log post_i - log prior_i)

    Args:
        post_logits: [batch, T, N, D_stoch, C]
        prior_logits: [batch, T, N, D_stoch, C]
        free_bits: minimum KL per stochastic dimension

    Returns:
        scalar loss (mean KL)
    """
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)
    log_post = jax.nn.log_softmax(post_logits, axis=-1)
    log_prior = jax.nn.log_softmax(prior_logits, axis=-1)

    # KL per category: sum over classes
    kl = (post_probs * (log_post - log_prior)).sum(axis=-1)  # [batch, T, N, D_stoch]

    # Free bits: allow KL up to free_bits before penalizing
    if free_bits > 0:
        kl = jnp.maximum(kl, free_bits)

    # Average over all dims
    return kl.mean()


def graph_rssm_loss(
    model_output: Dict,
    node_targets: jnp.ndarray,
    edge_targets: jnp.ndarray,
    kl_weight: float = 1.0,
    edge_weight: float = 0.5,
    free_bits: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute total Graph-RSSM training loss.

    Args:
        model_output: dict from GraphRSSM forward pass
        node_targets: [batch, T-1, N, D_node] ground truth metrics
        edge_targets: [batch, T-1, E, D_edge] ground truth edge features
        kl_weight: weight for KL term
        edge_weight: weight for edge prediction term
        free_bits: free bits for KL

    Returns:
        total_loss: scalar
        loss_dict: individual loss components
    """
    # Metric loss: predict next metrics, compare with ground truth
    metric_loss = gaussian_nll(
        model_output["metrics_mu"],
        model_output["metrics_logsigma"],
        node_targets,
    )

    # Edge loss
    edge_loss = edge_mse(model_output["edge_preds"], edge_targets)

    # KL loss between posterior and prior
    kl_loss = kl_categorical(
        model_output["post_logits"][:, 1:, :, :, :],  # skip first step (no prior)
        model_output["prior_logits"][:, 1:, :, :, :],
        free_bits=free_bits,
    )

    total = metric_loss + edge_weight * edge_loss + kl_weight * kl_loss

    loss_dict = {
        "metric_loss": metric_loss,
        "edge_loss": edge_loss,
        "kl_loss": kl_loss,
        "total_loss": total,
    }
    return total, loss_dict
