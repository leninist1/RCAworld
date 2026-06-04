"""Phase A pretraining: OB Day1 normal data → convergence test.

Verifies that RCAWorldFoundation can learn on real data before moving to
Onset Head and Component evaluation.
"""
import os, sys, time, argparse
import numpy as np
import jax, jax.numpy as jnp
import optax
from flax.training import train_state as flax_train_state
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
jax.config.update('jax_platform_name', 'cpu')

from foundation.models import RCAWorldFoundation
from foundation.training import world_model_loss, create_train_state, train_step


def load_ob_windows(h5_path: str, window_size: int = 23):
    """Load OB Day1 sliding windows from HDF5.

    Returns:
        nodes: [num_windows, window_size+1, N, D]
        type_idx: [N]
        obs_mask: [N, D]
    """
    with h5py.File(h5_path, 'r') as f:
        svc_names = [s.decode() for s in f['service_names'][:]]
        nodes_all = f['train/nodes'][:]
        ctx_all = f['train/ctx'][:]
        n_mean = f['stats/node_mean'][:]
        n_std = f['stats/node_std'][:]

    # Normalize
    nodes_norm = (nodes_all - n_mean) / (n_std + 1e-6)
    nodes_norm = np.nan_to_num(nodes_norm, nan=0.0)

    B, T_full, N, D = nodes_norm.shape
    print(f"  HDF5 data: {B} windows, T={T_full}, N={N}, D={D}")

    # Build sliding windows of size window_size+1 (for prediction target)
    windows = []
    for b in range(B):
        seq = nodes_norm[b]  # [T_full, N, D]
        for start in range(0, T_full - window_size, max(1, window_size // 2)):
            end = start + window_size + 1  # +1 for target
            if end <= T_full:
                windows.append(seq[start:end])

    windows = np.stack(windows, axis=0)  # [num_windows, T+1, N, D]

    # Type index (all SERVICE for OB)
    type_idx = np.zeros(N, dtype=np.int32)

    # Observation mask (all features valid)
    obs_mask = np.ones((N, D), dtype=np.float32)

    print(f"  Built {len(windows)} sub-windows of size {window_size}")
    return windows, type_idx, obs_mask, svc_names, N, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window_size', type=int, default=23)
    parser.add_argument('--h5', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                            "data", "processed", "cross_system.h5"))
    args = parser.parse_args()

    print("=" * 60)
    print("Phase A: OB Day1 Pretraining")
    print("=" * 60)

    # ── Load data ──
    windows, type_idx, obs_mask, svc_names, N, D_orig = load_ob_windows(
        args.h5, args.window_size)
    num_windows = len(windows)

    # Pad features to model's max_obs_dim
    MAX_D = 16
    if D_orig < MAX_D:
        pad = MAX_D - D_orig
        windows = np.pad(windows, ((0, 0), (0, 0), (0, 0), (0, pad)),
                         mode='constant')
        obs_mask = np.pad(obs_mask, ((0, 0), (0, pad)), mode='constant')

    # Train/val split (90/10)
    np.random.seed(42)
    idx = np.random.permutation(num_windows)
    split = int(0.9 * num_windows)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_data = windows[train_idx]
    val_data = windows[val_idx]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # ── Model init ──
    rng = jax.random.PRNGKey(42)
    x_example = jnp.array(train_data[:1])
    type_idx_j = jnp.array(type_idx)
    obs_mask_j = jnp.array(obs_mask)

    model = RCAWorldFoundation(
        common_dim=128,
        max_obs_dim=MAX_D,
        num_entity_types=3,
        det_dim=256,
        stoch_dim=32,
        stoch_classes=32,
        use_onset_head=False,
        use_component_head=False,
    )

    print("Initializing model...")
    init_rng, train_rng = jax.random.split(rng)
    example = {"x": x_example, "type_idx": type_idx_j, "obs_mask": obs_mask_j}
    state = create_train_state(model, init_rng, example, learning_rate=args.lr)
    n_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"  {n_params:,} parameters")

    # ── Training ──
    print(f"\nTraining ({args.epochs} epochs)...")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        epoch_train_loss = 0.0
        num_batches = 0
        for i in range(0, len(train_data), args.batch_size):
            batch_x = jnp.array(train_data[i:i + args.batch_size])
            batch = {"x": batch_x, "type_idx": type_idx_j, "obs_mask": obs_mask_j}
            train_rng, step_rng = jax.random.split(train_rng)
            state, loss, comps = train_step(
                state, batch, step_rng, kl_weight=0.1)
            epoch_train_loss += float(loss)
            num_batches += 1
        avg_train_loss = epoch_train_loss / max(1, num_batches)
        train_losses.append(avg_train_loss)

        # Val
        epoch_val_loss = 0.0
        num_val = 0
        for i in range(0, len(val_data), args.batch_size):
            batch_x = jnp.array(val_data[i:i + args.batch_size])
            batch = {"x": batch_x, "type_idx": type_idx_j, "obs_mask": obs_mask_j}
            val_rng, _ = jax.random.split(train_rng)
            out = model.apply(
                {"params": state.params}, batch_x, type_idx_j, val_rng,
                obs_mask=obs_mask_j, use_posterior=True)
            targets = {"obs": batch_x[:, 1:, :, :]}
            total_loss, _ = world_model_loss(out, targets, obs_mask=obs_mask_j)
            epoch_val_loss += float(total_loss)
            num_val += 1
        avg_val_loss = epoch_val_loss / max(1, num_val)
        val_losses.append(avg_val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"({elapsed:.1f}s)")

    # ── Summary ──
    print(f"\nLoss trajectory:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final:   {train_losses[-1]:.4f}")
    print(f"  Delta:   {train_losses[0] - train_losses[-1]:.4f}")
    if train_losses[-1] < train_losses[0]:
        print("  CONVERGED ✓")
    else:
        print("  WARNING: Loss did not decrease")

    return state, train_losses, val_losses


if __name__ == "__main__":
    main()
