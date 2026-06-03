"""Phase 2: Train Graph-RSSM on Nezha Online Boutique normal data.

Validates RQ1: Does the model learn normal microservice dynamics?
"""
import os
import sys

# Ruff: allow re-imports
# ruff: noqa: F811

import jax
import jax.numpy as jnp
import numpy as np
import h5py
from flax.training import train_state, orbax_utils
import optax
from tqdm import tqdm
import orbax.checkpoint as ocp
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models.graph_rssm import GraphRSSM
from training.losses import graph_rssm_loss

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase2")
os.makedirs(CKPT_DIR, exist_ok=True)


def load_dataset(path: str, split: str = "train"):
    """Load window data from HDF5."""
    with h5py.File(path, "r") as f:
        group = f[split]
        nodes = group["nodes"][:]  # [n_windows, window_size, N, D_node]
        edges = group["edges"][:]  # [n_windows, window_size, E, D_edge]
        ctx = group["ctx"][:]      # [n_windows, window_size, D_ctx]
    return nodes, edges, ctx


def main():
    print(f"Device: {jax.devices()[0]}")

    # Load dataset
    print("Loading dataset...")
    train_nodes, train_edges, train_ctx = load_dataset(DATA_PATH, "train")
    val_nodes, val_edges, val_ctx = load_dataset(DATA_PATH, "val")

    with h5py.File(DATA_PATH, "r") as f:
        edge_index = f["edge_index"][:]
        N = f.attrs["N_services"]
        E = f.attrs["N_edges"]
        N_node = f.attrs["N_node_features"]
        N_edge = f.attrs["N_edge_features"]
        N_ctx = f.attrs["N_context_features"]
        window_size = f.attrs["window_size"]
        service_names = [s.decode() if isinstance(s, bytes) else s for s in f["service_names"][:]]

    print(f"System: {' '.join(service_names)}")
    print(f"  N={N}, E={E}, D_node={N_node}, D_edge={N_edge}, D_ctx={N_ctx}")
    print(f"  Train windows: {len(train_nodes)}, Val windows: {len(val_nodes)}")
    print(f"  Window size: {window_size}")

    # Create model
    model = GraphRSSM(
        node_dim=N_node,
        edge_dim=N_edge,
        ctx_dim=N_ctx,
        graph_hidden=128,
        num_heads=4,
        det_dim=256,
        stoch_dim=32,
        stoch_classes=32,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, rng = jax.random.split(rng)

    # Initialize with first batch
    b = 1
    b_nodes = jnp.array(train_nodes[:b])
    b_edges = jnp.array(train_edges[:b])
    b_ctx = jnp.array(train_ctx[:b])
    b_edge_idx = jnp.array(edge_index, dtype=jnp.int32)

    params = model.init(init_rng, rng, b_nodes, b_edges, b_edge_idx, b_ctx)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model params: {n_params:,}")

    # Optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=3e-3,
        warmup_steps=200,
        decay_steps=2000,
        end_value=1e-4,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(schedule, weight_decay=1e-5),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    # Training step
    @jax.jit
    def train_step(s, rng_key, bn, be, bw):
        def loss_fn(p):
            out = s.apply_fn(p, rng_key, bn, be,
                             jnp.array(edge_index, dtype=jnp.int32), bw,
                             use_posterior=True)
            nt = bn[:, 1:, :, :]
            et = be[:, 1:, :, :]
            total, ld = graph_rssm_loss(out, nt, et, kl_weight=0.1, edge_weight=0.3,
                                         free_bits=1.0)
            return total, (out, ld)
        (loss, (_, ld)), grads = jax.value_and_grad(loss_fn, has_aux=True)(s.params)
        return s.apply_gradients(grads=grads), loss, ld

    @jax.jit
    def eval_step(s, rng_key, bn, be, bw):
        out = s.apply_fn(s.params, rng_key, bn, be,
                         jnp.array(edge_index, dtype=jnp.int32), bw,
                         use_posterior=True)
        nt = bn[:, 1:, :, :]
        et = be[:, 1:, :, :]
        _, ld = graph_rssm_loss(out, nt, et, kl_weight=0.1, edge_weight=0.3,
                                 free_bits=1.0)
        return ld

    # Training loop
    batch_size = 16
    n_epochs = 100
    steps_per_epoch = max(1, len(train_nodes) // batch_size)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_metric": [], "val_edge": [], "val_kl": []}

    print(f"\nTraining: {n_epochs} epochs, {steps_per_epoch} steps/epoch, batch={batch_size}")
    for epoch in range(n_epochs):
        # Shuffle
        idxs = np.random.permutation(len(train_nodes))
        train_nodes_epoch = train_nodes[idxs]
        train_edges_epoch = train_edges[idxs]
        train_ctx_epoch = train_ctx[idxs]

        epoch_losses = []
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, len(train_nodes_epoch))
            if start >= end:
                break

            bn = jnp.array(train_nodes_epoch[start:end])
            be = jnp.array(train_edges_epoch[start:end])
            bw = jnp.array(train_ctx_epoch[start:end])

            rng, srng = jax.random.split(rng)
            state, loss, ld = train_step(state, srng, bn, be, bw)
            epoch_losses.append(loss)

        avg_train = np.mean(epoch_losses)
        history["train_loss"].append(avg_train)

        # Validation
        val_batch = min(batch_size, len(val_nodes))
        vidx = np.random.choice(len(val_nodes), val_batch, replace=False)
        vn = jnp.array(val_nodes[vidx])
        ve = jnp.array(val_edges[vidx])
        vw = jnp.array(val_ctx[vidx])

        rng, erng = jax.random.split(rng)
        vld = eval_step(state, erng, vn, ve, vw)
        val_loss = float(vld["total_loss"])
        history["val_loss"].append(val_loss)
        history["val_metric"].append(float(vld["metric_loss"]))
        history["val_edge"].append(float(vld["edge_loss"]))
        history["val_kl"].append(float(vld["kl_loss"]))

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | train: {avg_train:.4f} | val: {val_loss:.4f} | "
                  f"metric: {vld['metric_loss']:.4f} | edge: {vld['edge_loss']:.4f} | "
                  f"kl: {vld['kl_loss']:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {"params": state.params, "opt_state": state.opt_state}
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            orbax_checkpointer.save(os.path.join(CKPT_DIR, "best"), ckpt,
                                     force=True)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Final train loss: {avg_train:.4f}")

    # Save final checkpoint
    ckpt = {"params": state.params, "opt_state": state.opt_state}
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    orbax_checkpointer.save(os.path.join(CKPT_DIR, "final"), ckpt, force=True)

    # RQ1 metrics
    print("\n" + "=" * 60)
    print("RQ1: Model learns normal dynamics")
    print("=" * 60)
    print(f"Final metric NLL: {history['val_metric'][-1]:.4f}")
    print(f"Final edge MSE: {history['val_edge'][-1]:.4f}")
    print(f"Final KL: {history['val_kl'][-1]:.4f}")
    print(f"Train loss reduction: {history['train_loss'][0]:.2f} -> {history['train_loss'][-1]:.2f}")

    # Quick prediction test
    vn_sample = jnp.array(val_nodes[:1])
    ve_sample = jnp.array(val_edges[:1])
    vw_sample = jnp.array(val_ctx[:1])

    rng, prng = jax.random.split(rng)
    out = state.apply_fn(state.params, prng, vn_sample, ve_sample,
                         jnp.array(edge_index, dtype=jnp.int32), vw_sample,
                         use_posterior=True)

    # Per-service prediction error
    from training.losses import gaussian_nll_per_node
    nt = vn_sample[:, 1:, :, :]
    per_node_nll = gaussian_nll_per_node(out["metrics_mu"], out["metrics_logsigma"], nt)
    # per_node_nll: [1, T-1, N]
    avg_per_node = np.array(per_node_nll[0].mean(axis=0))

    print("\nPer-service prediction NLL (lower = better):")
    for i, svc in enumerate(service_names):
        print(f"  {svc:<25s}: {avg_per_node[i]:.4f}")


if __name__ == "__main__":
    main()
