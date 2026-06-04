"""Pretraining pipeline for RCAWorld-Foundation.

Trains the world model on normal (non-fault) data using the ELBO objective:
  L = NLL(obs_pred, obs_true) + beta * KL(post || prior)

The model learns normal system dynamics from unlabeled normal windows.
Onset and component heads are only trained when labels are provided.
"""

from typing import Optional, Dict, Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
from tqdm import tqdm

from ..models.world_model import RCAWorldFoundation
from .losses import world_model_loss


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics."""
    batch_stats: Optional[Dict] = None


def create_train_state(model: RCAWorldFoundation,
                       rng_key,
                       example_input: Dict,
                       learning_rate: float = 1e-3,
                       ) -> TrainState:
    """Initialize model parameters and optimizer.

    Args:
        model: RCAWorldFoundation instance.
        rng_key: PRNG key for parameter initialization.
        example_input: Dict with 'x' [1,T,N,D], 'type_idx' [N], etc.
        learning_rate: Adam learning rate.

    Returns:
        TrainState with initialized params and optimizer.
    """
    init_rng, sample_rng = jax.random.split(rng_key)

    variables = model.init(
        init_rng,
        x=example_input["x"],
        type_idx=example_input["type_idx"],
        rng=sample_rng,
        obs_mask=example_input.get("obs_mask"),
        edge_index=example_input.get("edge_index"),
        ctx_seq=example_input.get("ctx_seq"),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def train_step(state: TrainState,
               batch: Dict,
               rng_key,
               kl_weight: float = 0.1,
               onset_weight: float = 1.0,
               component_weight: float = 1.0,
               consistency_weight: float = 0.01,
               ) -> tuple:
    """Single training step.

    Args:
        state: TrainState.
        batch: Dict with 'x', 'type_idx', and optional labels.
        rng_key: PRNG key for stochastic sampling.
        kl_weight, onset_weight, component_weight, consistency_weight:
            Loss weighting hyperparameters.

    Returns:
        (new_state, loss, loss_components)
    """
    sample_rng = jax.random.fold_in(rng_key, 0)

    def loss_fn(params):
        outputs = state.apply_fn(
            {"params": params},
            batch["x"],
            batch["type_idx"],
            sample_rng,
            obs_mask=batch.get("obs_mask"),
            edge_index=batch.get("edge_index"),
            ctx_seq=batch.get("ctx_seq"),
            use_posterior=True,
        )

        # Build targets
        targets = {
            "obs": batch["x"][:, 1:, :, :],  # next-step targets
        }
        if "onset_times" in batch:
            targets["onset_times"] = batch["onset_times"]
            targets["onset_masks"] = batch.get(
                "onset_masks",
                jnp.ones_like(batch["onset_times"]),
            )
        if "true_component_idx" in batch:
            targets["true_component_idx"] = batch["true_component_idx"]
            targets["component_masks"] = batch.get(
                "component_masks",
                jnp.ones(batch["true_component_idx"].shape[0]),
            )

        total_loss, loss_components = world_model_loss(
            outputs=outputs,
            targets=targets,
            obs_mask=batch.get("obs_mask"),
            kl_weight=kl_weight,
            onset_weight=onset_weight,
            component_weight=component_weight,
            consistency_weight=consistency_weight,
        )
        return total_loss, (outputs, loss_components)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (outputs, loss_components)), grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, loss_components


def pretrain_normal(model: RCAWorldFoundation,
                    train_data: Any,
                    val_data: Optional[Any] = None,
                    num_epochs: int = 80,
                    batch_size: int = 16,
                    learning_rate: float = 1e-3,
                    kl_weight: float = 0.1,
                    rng_key: int = 42,
                    window_size: int = 24,
                    ) -> TrainState:
    """Pretrain the world model on normal (non-fault) data.

    Args:
        model: RCAWorldFoundation instance.
        train_data: Training dataset (events or EventBatch).
        val_data: Optional validation dataset.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        kl_weight: KL divergence weight.
        rng_key: Random seed.
        window_size: Sliding window size for training.

    Returns:
        Trained TrainState.
    """
    rng = jax.random.PRNGKey(rng_key)

    # Build example input for initialization
    # This requires the data to be in EventBatch format
    if hasattr(train_data, "event_tensor"):
        nodes = jnp.array(train_data.event_tensor[:window_size+1])  # [T+1, N, D]
    else:
        nodes = jnp.array(train_data)[:window_size+1]

    N = nodes.shape[1]
    example_input = {
        "x": nodes[None, :, :, :],  # [1, T+1, N, D]
        "type_idx": jnp.zeros(N, dtype=jnp.int32),
    }

    state = create_train_state(model, rng, example_input, learning_rate)

    # Simple training loop
    # In practice, build batches from sliding windows
    losses_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Build batches (simplified: single batch per epoch)
        rng, step_rng = jax.random.split(rng)

        # For now, use the same data as single batch
        # TODO: Implement proper batching with sliding windows
        batch = {
            "x": jnp.array(nodes)[None, :, :, :],  # [1, T+1, N, D]
            "type_idx": jnp.zeros(N, dtype=jnp.int32),
        }

        state, loss, loss_components = train_step(
            state, batch, step_rng, kl_weight=kl_weight)
        epoch_loss = float(loss)
        losses_history.append(epoch_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            nll = float(loss_components.get("nll", 0))
            kl = float(loss_components.get("kl", 0))
            print(f"Epoch {epoch:3d}/{num_epochs}  "
                  f"loss={epoch_loss:.4f}  nll={nll:.4f}  kl={kl:.4f}")

    return state
