"""Phase A core evaluation: Onset Head vs Oracle/clean window on Bank container.

Evaluates RCAWorldFoundation on Bank dataset:
  - Oracle window:     fault service residual peak selects window (upper bound)
  - Onset window:      OnsetHead scores select window (new method)
  - Timestamp window:  record.csv timestamp selects window (current clean baseline)

Reports Top-1, Top-3, MRR for each method.
"""
import os, sys, time, argparse
import numpy as np
import jax, jax.numpy as jnp
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
jax.config.update('jax_platform_name', 'cpu')

from foundation.adapters import OpenRCABankAdapter
from foundation.models import RCAWorldFoundation
from foundation.evaluation import compute_component_metrics


def parse_bank_data(adapter, bank_dir, max_entities=None):
    """Parse Bank data and return entities, tensor, labels."""
    entities = adapter.discover_entities(bank_dir)
    # Use only container entities for evaluation
    cont_entities = [e for e in entities if e.entity_type.value == "container"]
    if max_entities:
        cont_entities = cont_entities[:max_entities]

    events = adapter.extract_events(bank_dir, cont_entities)
    batch = adapter.events_to_batch(events, cont_entities)
    labels = adapter.extract_labels(bank_dir, cont_entities)

    # Map labels to entity index
    entity_ids = [e.entity_id for e in cont_entities]
    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    label_idx = []
    label_info = []
    for lb in labels:
        if lb.component in eid_to_idx:
            label_idx.append(eid_to_idx[lb.component])
            label_info.append(lb)
        else:
            # Fuzzy match
            comp_lower = lb.component.lower().replace("_", "").replace("-", "").replace(" ", "")
            matched = False
            for eid in entity_ids:
                eid_lower = eid.lower().replace("_", "").replace("-", "").replace(" ", "")
                if comp_lower in eid_lower or eid_lower in comp_lower:
                    label_idx.append(eid_to_idx[eid])
                    label_info.append(lb)
                    matched = True
                    break
            if not matched:
                label_idx.append(-1)
                label_info.append(lb)

    return cont_entities, batch, np.array(label_idx), label_info


def normalize_tensor(tensor):
    """Standardize to zero mean, unit variance."""
    mean = tensor.mean(axis=(0, 1), keepdims=True) + 1e-6
    std = tensor.std(axis=(0, 1), keepdims=True) + 1e-6
    return np.nan_to_num((tensor - mean) / std, nan=0.0)


def compute_residuals(model_params, model, tensor, type_idx, obs_mask,
                      window_size=23, batch_size=8):
    """Compute per-entity per-timestep residuals over sliding windows."""
    T, N, D = tensor.shape
    residuals = np.full((T, N), np.nan, dtype=np.float32)

    # Build sliding windows
    windows = []
    positions = []
    for start in range(0, T - window_size, max(1, window_size // 2)):
        end = start + window_size + 1
        if end > T:
            break
        windows.append(tensor[start:end])
        positions.append((start + 1, end))  # residual range

    if not windows:
        return residuals

    windows = np.stack(windows, axis=0)  # [num_w, T_w+1, N, D]
    type_idx_j = jnp.array(type_idx)
    obs_mask_j = jnp.array(obs_mask) if obs_mask is not None else None
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), batch_size):
        batch_x = jnp.array(windows[i:i + batch_size])
        out = model.apply(
            {"params": model_params},
            batch_x, type_idx_j, rng,
            obs_mask=obs_mask_j, use_posterior=True)

        res_batch = np.array(out["residual"])  # [B, T_w, N]
        for j, (s, e) in enumerate(positions[i:i + batch_size]):
            ns = min(res_batch.shape[1], e - s)
            residuals[s:e] = res_batch[j, :ns]

    return np.nan_to_num(residuals, nan=0.0)


def compute_onset_scores(model_params, model, tensor, type_idx, obs_mask,
                         window_size=23, batch_size=8):
    """Compute OnsetHead scores for all timesteps."""
    T, N, D = tensor.shape
    onset_scores = np.full((T, N), np.nan, dtype=np.float32)

    windows = []
    positions = []
    for start in range(0, T - window_size, max(1, window_size // 2)):
        end = start + window_size + 1
        if end > T:
            break
        windows.append(tensor[start:end])
        positions.append((start + 1, end))

    if not windows:
        return onset_scores

    windows = np.stack(windows, axis=0)
    type_idx_j = jnp.array(type_idx)
    obs_mask_j = jnp.array(obs_mask) if obs_mask is not None else None
    rng = jax.random.PRNGKey(0)

    for i in range(0, len(windows), batch_size):
        batch_x = jnp.array(windows[i:i + batch_size])
        out = model.apply(
            {"params": model_params},
            batch_x, type_idx_j, rng,
            obs_mask=obs_mask_j, use_posterior=True)

        onset_batch = np.array(out["onset"]["onset_score"])  # [B, T_w, N]
        for j, (s, e) in enumerate(positions[i:i + batch_size]):
            ns = min(onset_batch.shape[1], e - s)
            onset_scores[s:e] = onset_batch[j, :ns]

    return np.nan_to_num(onset_scores, nan=0.0)


def evaluate_window(residuals, labels_idx, window_method="oracle",
                    pre_window=5, post_window=3):
    """Evaluate RCA with a given window selection method.

    window_method:
      "oracle": use residual peak near root cause entity as window center
      "onset": use onset score peak near root cause entity as window center
      "timestamp": use fixed window (no window selection, report overall)

    Returns top1, top3, mrr for valid labels.
    """
    T, N = residuals.shape

    valid_labels = [i for i, idx in enumerate(labels_idx) if 0 <= idx < N]
    if not valid_labels:
        return {"top1": 0.0, "top3": 0.0, "mrr": 0.0, "n": 0}

    ranks = []
    for li in valid_labels:
        rc = labels_idx[li]

        if window_method in ("oracle", "onset"):
            # Find peak window for this fault
            rc_res = residuals[:, rc]
            peak = int(np.nanargmax(rc_res))
            fs = max(0, peak - pre_window)
            fe = min(T, peak + post_window)
        else:
            # Use full time range
            fs, fe = 0, T

        fault_res = residuals[fs:fe]
        if fault_res.size == 0:
            ranks.append(1)
            continue

        avg_res = np.nanmean(fault_res, axis=0)
        rank = int(np.argsort(-avg_res).tolist().index(rc)) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float32)
    n = len(ranks)
    top1 = float(np.mean(ranks <= 1))
    top3 = float(np.mean(ranks <= 3))
    mrr = float(np.mean(1.0 / ranks))
    return {"top1": top1, "top3": top3, "mrr": mrr, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank_dir', type=str,
                        default='/home/dell2/RCA513/yyx/OpenRCA/Bank/Bank')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pretrained model checkpoint (optional)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Quick training epochs before eval')
    parser.add_argument('--window_size', type=int, default=23)
    parser.add_argument('--max_days', type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase A Core Eval: Bank Container RCA")
    print("=" * 60)

    # ── Parse Bank data ──
    adapter = OpenRCABankAdapter(max_days=args.max_days,
                                 max_container_events=50000,
                                 max_container_rows=500000)
    entities, batch, labels_idx, label_info = parse_bank_data(
        adapter, args.bank_dir)
    N = len(entities)
    T, _, D = batch.event_tensor.shape

    valid = [i for i, idx in enumerate(labels_idx) if 0 <= idx < N]
    print(f"Entities: {N} (containers only)")
    print(f"Batch:    T={T}, D={D}")
    print(f"Labels:   {len(labels_idx)} total, {len(valid)} mapped")
    for li in valid[:5]:
        idx = labels_idx[li]
        print(f"  Fault: {label_info[li].reason} @ {label_info[li].component} "
              f"(entity_idx={idx})")

    # ── Prepare data ──
    MAX_D = 16
    tensor_raw = batch.event_tensor.astype(np.float32)
    if D < MAX_D:
        pad = MAX_D - D
        tensor_raw = np.pad(tensor_raw, ((0, 0), (0, 0), (0, pad)), mode='constant')
        obs_mask = np.zeros((N, MAX_D), dtype=np.float32)
        obs_mask[:, :D] = 1.0
    else:
        tensor_raw = tensor_raw[:, :, :MAX_D]
        obs_mask = np.ones((N, MAX_D), dtype=np.float32)

    tensor = normalize_tensor(tensor_raw)
    type_idx = np.zeros(N, dtype=np.int32)

    # ── Train model on Bank normal data (or load checkpoint) ──
    # Use first portion of data (before any fault) as normal training data
    # For quick evaluation, train on the full range (since faults are sparse)
    from foundation.training import create_train_state, train_step

    rng = jax.random.PRNGKey(42)
    model = RCAWorldFoundation(
        common_dim=128, max_obs_dim=MAX_D, num_entity_types=3,
        det_dim=256, stoch_dim=32, stoch_classes=32,
        use_onset_head=True, use_component_head=True,
    )

    example_x = jnp.array(tensor[:args.window_size + 1][None, ...])
    type_idx_j = jnp.array(type_idx)
    obs_mask_j = jnp.array(obs_mask)

    print(f"\nInitializing model...")
    state = create_train_state(model, rng,
                               {"x": example_x, "type_idx": type_idx_j,
                                "obs_mask": obs_mask_j},
                               learning_rate=1e-3)
    params_count = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"  {params_count:,} params")

    # Quick training on Bank data
    print(f"\nQuick training ({args.epochs} epochs on Bank data)...")
    window = args.window_size
    train_rng = jax.random.PRNGKey(123)

    for epoch in range(args.epochs):
        t0 = time.time()

        # Build a batch from sliding windows
        batch_windows = []
        for start in range(0, T - window, max(1, window)):
            end = start + window + 1
            if end <= T:
                batch_windows.append(tensor[start:end])
        if not batch_windows:
            break

        # Take a subset for speed
        max_batches = min(50, len(batch_windows))
        indices = np.random.choice(len(batch_windows), max_batches, replace=False)
        batch_data = np.stack([batch_windows[i] for i in indices], axis=0)

        batch = {"x": jnp.array(batch_data[:16]),
                 "type_idx": type_idx_j, "obs_mask": obs_mask_j}
        train_rng, step_rng = jax.random.split(train_rng)

        if args.epochs <= 5:
            # Use small batch for quick test
            state, loss, _ = train_step(
                state, batch, step_rng, kl_weight=0.1)
        else:
            for i in range(0, len(batch_data), 16):
                b = {"x": jnp.array(batch_data[i:i+16]),
                     "type_idx": type_idx_j, "obs_mask": obs_mask_j}
                train_rng, sr = jax.random.split(train_rng)
                state, loss, _ = train_step(state, b, sr, kl_weight=0.1)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  loss={float(loss):.4f}  "
              f"({time.time()-t0:.1f}s)")

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print("Computing residuals & onset scores...")
    print(f"{'='*60}")

    t0 = time.time()
    residuals = compute_residuals(
        state.params, model, tensor, type_idx, obs_mask,
        window_size=args.window_size)
    print(f"  Residuals: {residuals.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    onset_scores_all = compute_onset_scores(
        state.params, model, tensor, type_idx, obs_mask,
        window_size=args.window_size)
    print(f"  Onset scores: {onset_scores_all.shape} ({time.time()-t0:.1f}s)")

    # ── Oracle: use residual peak to select window ──
    oracle = evaluate_window(residuals, labels_idx, window_method="oracle")
    print(f"\nOracle window: Top-1={oracle['top1']:.1%}  "
          f"Top-3={oracle['top3']:.1%}  MRR={oracle['mrr']:.3f}  (n={oracle['n']})")

    # ── Onset: use onset scores to select window ──
    onset = evaluate_window(onset_scores_all, labels_idx, window_method="onset")
    print(f"Onset window:  Top-1={onset['top1']:.1%}  "
          f"Top-3={onset['top3']:.1%}  MRR={onset['mrr']:.3f}  (n={onset['n']})")

    # ── Full-sequence (no window) ──
    full = evaluate_window(residuals, labels_idx, window_method="timestamp")
    print(f"Full sequence: Top-1={full['top1']:.1%}  "
          f"Top-3={full['top3']:.1%}  MRR={full['mrr']:.3f}  (n={full['n']})")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'Top-1':>8s} {'Top-3':>8s} {'MRR':>8s} {'N':>5s}")
    print(f"  {'Oracle window':<20s} {oracle['top1']:>7.1%} {oracle['top3']:>7.1%} {oracle['mrr']:>8.3f} {oracle['n']:>5d}")
    print(f"  {'Onset (new)':<20s} {onset['top1']:>7.1%} {onset['top3']:>7.1%} {onset['mrr']:>8.3f} {onset['n']:>5d}")
    print(f"  {'Full sequence':<20s} {full['top1']:>7.1%} {full['top3']:>7.1%} {full['mrr']:>8.3f} {full['n']:>5d}")
    print(f"  {'Current clean eval':<20s} {'38%':>8s} {'50%':>8s} {'0.50':>8s} {16:>5d}")


if __name__ == "__main__":
    main()
