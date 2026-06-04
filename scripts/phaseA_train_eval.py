"""Phase A: OB Day1 full training → Bank zero-shot evaluation.

Strict constraint: OpenRCA data (Bank/Market/Telecom) used for EVALUATION ONLY.
Training ONLY on Nezha OB Day1 normal data.

Matches original training hyperparameters:
  - AdamW, warmup_cosine: init=1e-4, peak=3e-3, warmup=200, decay=2000
  - clip_by_global_norm(10.0), weight_decay=1e-5
  - batch_size=16, kl_weight=0.1, free_bits=1.0
  - window_size=24 (original HDF5 windows)
"""
import os, sys, time, argparse
import numpy as np
import jax, jax.numpy as jnp
import optax
from flax.training import train_state as flax_train_state
import orbax.checkpoint as ocp
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 1 (free)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

from foundation.models import RCAWorldFoundation
from foundation.training import world_model_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "hipster_dataset.h5")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phaseA")
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"

os.makedirs(CKPT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_ob_data(h5_path: str, max_obs_dim: int = 16):
    """Load OB Day1 HDF5, normalize, pad features."""
    with h5py.File(h5_path, 'r') as f:
        nodes = f['train/nodes'][:]      # [B, 24, N, D]
        val_nodes = f['val/nodes'][:]     # [B, 24, N, D]
        n_mean = f['stats/node_mean'][:]
        n_std = f['stats/node_std'][:]
        svc_names = [s.decode() for s in f['service_names'][:]]
        N = f.attrs['N_services']
        window_size = f.attrs['window_size']

    # Normalize
    nodes_n = (nodes - n_mean) / (n_std + 1e-6)
    val_n = (val_nodes - n_mean) / (n_std + 1e-6)
    nodes_n = np.nan_to_num(nodes_n, nan=0.0)
    val_n = np.nan_to_num(val_n, nan=0.0)

    _, _, _, D_orig = nodes_n.shape

    # Pad features
    if D_orig < max_obs_dim:
        pad = max_obs_dim - D_orig
        nodes_n = np.pad(nodes_n, ((0,0),(0,0),(0,0),(0,pad)), mode='constant')
        val_n = np.pad(val_n, ((0,0),(0,0),(0,0),(0,pad)), mode='constant')
        obs_mask = np.zeros((N, max_obs_dim), dtype=np.float32)
        obs_mask[:, :D_orig] = 1.0
    else:
        obs_mask = np.ones((N, max_obs_dim), dtype=np.float32)

    type_idx = np.zeros(N, dtype=np.int32)

    print(f"  OB Day1: {len(nodes_n)} train, {len(val_n)} val windows")
    print(f"  N={N}, window={window_size}, D_orig={D_orig}→D={max_obs_dim}")
    return nodes_n, val_n, type_idx, obs_mask, svc_names, N, D_orig


# ═══════════════════════════════════════════════════════════════════════
# Model & training
# ═══════════════════════════════════════════════════════════════════════

def create_model(max_obs_dim=16, num_types=3):
    return RCAWorldFoundation(
        common_dim=128,
        max_obs_dim=max_obs_dim,
        num_entity_types=num_types,
        det_dim=256,
        stoch_dim=32,
        stoch_classes=32,
        use_graph=False,
        use_onset_head=True,
        use_component_head=True,
    )


def create_optimizer():
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=3e-3,
        warmup_steps=200, decay_steps=2000, end_value=1e-4)
    return optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(schedule, weight_decay=1e-5),
    )


def save_checkpoint(state, path):
    ckpt = {"params": state.params, "opt_state": state.opt_state}
    ocp.PyTreeCheckpointer().save(path, ckpt, force=True)


def load_checkpoint(state, path):
    restored = ocp.PyTreeCheckpointer().restore(path)
    return state.replace(params=restored["params"], opt_state=restored.get("opt_state", state.opt_state))


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_bank_for_eval(bank_dir, max_days=1, target_dim=8):
    """Parse Bank container data and map to OB 8-feature space.

    Returns tensor normalized with OB stats so model can evaluate zero-shot.
    target_dim: number of OB features (8 for the original model).
    """
    from foundation.adapters import OpenRCABankAdapter
    adapter = OpenRCABankAdapter(
        max_days=max_days, max_container_events=100000, max_container_rows=1000000)

    # Use ALL entities (containers only) for entity discovery
    entities_all = adapter.discover_entities(bank_dir)
    cont_entities = [e for e in entities_all if e.entity_type.value == "container"]

    # Extract raw events, then build a dense tensor in OB feature space
    events = adapter.extract_events(bank_dir, cont_entities)

    # Map Bank KPI names → OB feature indices [0..7]
    # Same mapping as original phase567_evaluate.py
    kpi_to_ob_feat = {
        "cpu": 0, "jvm_cpu": 0,
        "mem": 1, "mem_usage": 1, "jvm_mem": 1,
        "net_rx": 2,
        "net_tx": 3,
        "disk_io": 4, "mysql_io": 4,
        "threads": 5, "sessions": 5,
        "fgc": 6,
        # feature 7 (success_rate) has no Bank equivalent, stays zero
    }

    # Collect timestamps and entities
    all_ts = sorted(set(ev.timestamp for ev in events))
    entity_ids = [e.entity_id for e in cont_entities]
    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    N = len(entity_ids)
    T = len(all_ts)
    ts_to_idx = {t: i for i, t in enumerate(all_ts)}

    # Build tensor in OB 8-feature space
    tensor = np.zeros((T, N, target_dim), dtype=np.float32)
    counts = np.zeros((T, N, target_dim), dtype=np.float32)

    for ev in events:
        if ev.entity_id not in eid_to_idx:
            continue
        feat_name = ev.feature_name
        if feat_name not in kpi_to_ob_feat:
            continue
        ob_idx = kpi_to_ob_feat[feat_name]
        t = ts_to_idx.get(ev.timestamp)
        n = eid_to_idx[ev.entity_id]
        if t is None:
            continue
        tensor[t, n, ob_idx] += ev.value
        counts[t, n, ob_idx] += 1

    # Average duplicate entries, forward fill
    for n in range(N):
        for d in range(target_dim):
            cnt = counts[:, n, d]
            mask = cnt > 0
            if mask.any():
                tensor[mask, n, d] /= cnt[mask]
            last = 0.0
            for t in range(T):
                if counts[t, n, d] > 0:
                    last = tensor[t, n, d]
                else:
                    tensor[t, n, d] = last

    # Labels
    labels = adapter.extract_labels(bank_dir, cont_entities)
    label_idx, label_info = [], []
    for lb in labels:
        matched_idx = -1
        if lb.component in eid_to_idx:
            matched_idx = eid_to_idx[lb.component]
        else:
            comp_l = lb.component.lower().replace("_","").replace("-","").replace(" ","")
            for eid in entity_ids:
                eid_l = eid.lower().replace("_","").replace("-","").replace(" ","")
                if comp_l in eid_l or eid_l in comp_l:
                    matched_idx = eid_to_idx[eid]
                    break
        label_idx.append(matched_idx)
        label_info.append(lb)

    return cont_entities, tensor, np.array(label_idx), label_info


def compute_residuals_bank(params, model, tensor, type_idx, obs_mask, ws=23):
    """Sliding-window residual computation on Bank data."""
    T, N, D = tensor.shape
    residuals = np.full((T, N), np.nan, dtype=np.float32)
    windows, positions = [], []
    for start in range(0, T - ws, max(1, ws // 2)):
        end = start + ws + 1
        if end > T: break
        windows.append(tensor[start:end])
        positions.append((start + 1, end))
    if not windows: return residuals

    windows = np.stack(windows, axis=0)
    t_idx = jnp.array(type_idx)
    o_mask = jnp.array(obs_mask) if obs_mask is not None else None
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), 8):
        bx = jnp.array(windows[i:i+8])
        out = model.apply({"params": params}, bx, t_idx, rng,
                          obs_mask=o_mask, use_posterior=True)
        res_b = np.array(out["residual"])
        for j, (s, e) in enumerate(positions[i:i+8]):
            ns = min(res_b.shape[1], e - s)
            residuals[s:e] = res_b[j, :ns]
    return np.nan_to_num(residuals, nan=0.0)


def compute_onset_bank(params, model, tensor, type_idx, obs_mask, ws=23):
    """Sliding-window onset score computation."""
    T, N, D = tensor.shape
    onset_scores = np.full((T, N), np.nan, dtype=np.float32)
    windows, positions = [], []
    for start in range(0, T - ws, max(1, ws // 2)):
        end = start + ws + 1
        if end > T: break
        windows.append(tensor[start:end])
        positions.append((start + 1, end))
    if not windows: return onset_scores

    windows = np.stack(windows, axis=0)
    t_idx = jnp.array(type_idx)
    o_mask = jnp.array(obs_mask) if obs_mask is not None else None
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), 8):
        bx = jnp.array(windows[i:i+8])
        out = model.apply({"params": params}, bx, t_idx, rng,
                          obs_mask=o_mask, use_posterior=True)
        onset_b = np.array(out["onset"]["onset_score"])
        for j, (s, e) in enumerate(positions[i:i+8]):
            ns = min(onset_b.shape[1], e - s)
            onset_scores[s:e] = onset_b[j, :ns]
    return np.nan_to_num(onset_scores, nan=0.0)


def evaluate_ranking(residuals, labels_idx, method="oracle",
                     onset_scores=None, pre_win=5, post_win=5):
    """Rank entities within a selected window.

    method: "oracle"=residual peak, "onset"=onset score peak, "full"=all time
    """
    T, N = residuals.shape
    valid = [i for i, idx in enumerate(labels_idx) if 0 <= idx < N]
    if not valid: return {"top1": 0, "top3": 0, "mrr": 0, "n": 0}

    ranks = []
    for li in valid:
        rc = labels_idx[li]
        if method == "oracle":
            peak = int(np.nanargmax(residuals[:, rc]))
        elif method == "onset" and onset_scores is not None:
            peak = int(np.nanargmax(onset_scores[:, rc]))
        else:
            peak = T // 2  # center

        fs = max(0, peak - pre_win)
        fe = min(T, peak + post_win)
        win_res = residuals[fs:fe]
        if win_res.size == 0: ranks.append(1); continue
        avg = np.nanmean(win_res, axis=0)
        rank = int(np.argsort(-avg).tolist().index(rc)) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float32)
    n = len(ranks)
    return {
        "top1": float(np.mean(ranks <= 1)), "top3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)), "n": n
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_only', action='store_true', help='Skip training, eval from checkpoint')
    parser.add_argument('--eval_bank', action='store_true', default=True)
    parser.add_argument('--eval_market', action='store_true')
    parser.add_argument('--eval_telecom', action='store_true')
    args = parser.parse_args()

    MAX_D = 16
    NUM_TYPES = 3

    # ── Load OB Day1 data ──
    print("=" * 60)
    print("Loading OB Day1 training data...")
    train_data, val_data, type_idx, obs_mask, svc_names, N_svc, D_orig = \
        load_ob_data(DATA_PATH, MAX_D)
    type_idx_j = jnp.array(type_idx)
    obs_mask_j = jnp.array(obs_mask)

    # ── Create model ──
    print("Creating model...")
    model = create_model(MAX_D, NUM_TYPES)
    rng = jax.random.PRNGKey(42)
    init_rng, train_rng = jax.random.split(rng)

    ex_x = jnp.array(train_data[:1])  # [1, 24, N, D]
    variables = model.init(init_rng, ex_x, type_idx_j, train_rng,
                           obs_mask=obs_mask_j)
    n_params = sum(p.size for p in jax.tree.leaves(variables["params"]))
    print(f"  {n_params:,} params")

    tx = create_optimizer()
    state = flax_train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)

    # ── Training ──
    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"Training {args.epochs} epochs on OB Day1 (NORMAL DATA ONLY)")
        print(f"{'='*60}")

        # JIT-compiled train step
        @jax.jit
        def train_step_jit(p, rng_k, bx):
            def loss_fn(pp):
                out = model.apply({"params": pp}, bx, type_idx_j, rng_k,
                                  obs_mask=obs_mask_j, use_posterior=True)
                targets = {"obs": bx[:, 1:, :, :]}
                total, _ = world_model_loss(out, targets, obs_mask=obs_mask_j,
                                            kl_weight=0.1)
                return total
            loss, grads = jax.value_and_grad(loss_fn)(p)
            return loss, grads

        @jax.jit
        def val_step_jit(p, rng_k, bx):
            out = model.apply({"params": p}, bx, type_idx_j, rng_k,
                              obs_mask=obs_mask_j, use_posterior=True)
            targets = {"obs": bx[:, 1:, :, :]}
            total, _ = world_model_loss(out, targets, obs_mask=obs_mask_j,
                                        kl_weight=0.1)
            return total

        steps_per_epoch = max(1, len(train_data) // args.batch_size)
        best_val = float('inf')
        history = {"train": [], "val": []}

        for epoch in range(args.epochs):
            t0 = time.time()
            idxs = np.random.permutation(len(train_data))
            train_shuf = train_data[idxs]
            epoch_losses = []

            for step in range(steps_per_epoch):
                start = step * args.batch_size
                end = min(start + args.batch_size, len(train_shuf))
                if start >= end: break
                bx = jnp.array(train_shuf[start:end])
                train_rng, srng = jax.random.split(train_rng)
                loss, grads = train_step_jit(state.params, srng, bx)
                state = state.apply_gradients(grads=grads)
                epoch_losses.append(float(loss))

            avg_train = np.mean(epoch_losses)
            history["train"].append(avg_train)

            # Validation
            vidx = np.random.choice(len(val_data), min(16, len(val_data)), replace=False)
            vx = jnp.array(val_data[vidx])
            train_rng, vrng = jax.random.split(train_rng)
            val_loss = float(val_step_jit(state.params, vrng, vx))
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(state, os.path.join(CKPT_DIR, "best"))

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                      f"train={avg_train:.4f}  val={val_loss:.4f}  "
                      f"({time.time()-t0:.1f}s)")

        save_checkpoint(state, os.path.join(CKPT_DIR, "final"))
        print(f"\n  Best val: {best_val:.4f}")
        print(f"  Train loss: {history['train'][0]:.2f} → {history['train'][-1]:.2f}")

    else:
        # Load checkpoint
        ckpt_path = os.path.join(CKPT_DIR, "best")
        if os.path.exists(ckpt_path):
            state = load_checkpoint(state, ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}")

    # ── Bank Evaluation ──
    if args.eval_bank:
        print(f"\n{'='*60}")
        print("BANK ZERO-SHOT EVALUATION (no Bank finetuning)")
        print(f"{'='*60}")

        bank_dir = os.path.join(OPENRCA, "Bank", "Bank")
        # Parse with feature mapping to OB 8-dim space
        entities, tensor_bank, labels_idx, label_info = parse_bank_for_eval(bank_dir)
        N = len(entities)
        T, D_bank = tensor_bank.shape[0], tensor_bank.shape[2]
        valid = sum(1 for idx in labels_idx if 0 <= idx < N)
        print(f"  Entities: {N} (containers), T={T}, D_mapped={D_bank} (OB space)")
        print(f"  Labels: {len(labels_idx)} total, {valid} mapped")

        # Load OB normalization stats (shape: [1, N_svc, D] -> average over services)
        with h5py.File(DATA_PATH, 'r') as f:
            ob_mean_raw = f['stats/node_mean'][:]  # [1, 10, 8]
            ob_std_raw = f['stats/node_std'][:]    # [1, 10, 8]
        ob_mean = ob_mean_raw.mean(axis=1)  # [1, 8]
        ob_std = ob_std_raw.mean(axis=1)    # [1, 8]
        D_orig = ob_mean.shape[-1]  # 8

        # Normalize with OB stats, pad to MAX_D
        ob_mean_b = ob_mean.reshape(1, 1, D_bank)
        ob_std_b = ob_std.reshape(1, 1, D_bank)
        tensor_norm = (tensor_bank - ob_mean_b) / (ob_std_b + 1e-6)
        tensor_norm = np.nan_to_num(tensor_norm, nan=0.0)
        pad = MAX_D - D_bank
        tensor_norm = np.pad(tensor_norm, ((0,0),(0,0),(0,pad)), mode='constant')
        b_obs_mask = np.zeros((N, MAX_D), dtype=np.float32)
        b_obs_mask[:, :D_bank] = 1.0
        b_type_idx = np.zeros(N, dtype=np.int32)

        # Compute residuals and onset scores
        print("  Computing residuals...")
        t0 = time.time()
        residuals = compute_residuals_bank(
            state.params, model, tensor_norm, b_type_idx, b_obs_mask, ws=23)
        print(f"    {time.time()-t0:.1f}s")

        print("  Computing onset scores...")
        t0 = time.time()
        onset_all = compute_onset_bank(
            state.params, model, tensor_norm, b_type_idx, b_obs_mask, ws=23)
        print(f"    {time.time()-t0:.1f}s")

        # Evaluate
        oracle_r = evaluate_ranking(residuals, labels_idx, "oracle")
        onset_r = evaluate_ranking(residuals, labels_idx, "onset", onset_scores=onset_all)
        full_r = evaluate_ranking(residuals, labels_idx, "full")

        print(f"\n  {'Method':<18s} {'Top-1':>8s} {'Top-3':>8s} {'MRR':>8s} {'N':>5s}")
        print(f"  {'Oracle (upper)':<18s} {oracle_r['top1']:>7.1%} {oracle_r['top3']:>7.1%} {oracle_r['mrr']:>8.3f} {oracle_r['n']:>5d}")
        print(f"  {'Onset (new)':<18s} {onset_r['top1']:>7.1%} {onset_r['top3']:>7.1%} {onset_r['mrr']:>8.3f} {onset_r['n']:>5d}")
        print(f"  {'Full sequence':<18s} {full_r['top1']:>7.1%} {full_r['top3']:>7.1%} {full_r['mrr']:>8.3f} {full_r['n']:>5d}")
        print(f"  {'Orig clean eval':<18s} {'38%':>8s} {'50%':>8s} {'0.50':>8s} {16:>5d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
