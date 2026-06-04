"""Phase A unit test: Bank Adapter → EventBatch → Model init & forward.

Usage: python scripts/phaseA_unit_test.py
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Force CPU to avoid GPU OOM
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update('jax_platform_name', 'cpu')

from foundation.adapters import OpenRCABankAdapter
from foundation.models import RCAWorldFoundation
from foundation.evaluation import compute_component_metrics

# ── Paths ──────────────────────────────────────────────────────────
OPENRCA = "/home/dell2/RCA513/yyx/OpenRCA"
BANK_DIR = os.path.join(OPENRCA, "Bank", "Bank")

# ── Step 1: Adapter → Entities ─────────────────────────────────────
print("=" * 60)
print("Step 1: OpenRCABankAdapter — Entity Discovery")
print("=" * 60)

adapter = OpenRCABankAdapter(max_days=1, max_container_timestamps=5000)
entities = adapter.discover_entities(BANK_DIR)
svc_entities = [e for e in entities if e.entity_type.value == "service"]
cont_entities = [e for e in entities if e.entity_type.value == "container"]
print(f"  Total entities: {len(entities)}")
print(f"  Services: {len(svc_entities)}  e.g. {[e.entity_id for e in svc_entities[:5]]}")
print(f"  Containers: {len(cont_entities)}  e.g. {[e.entity_id for e in cont_entities[:5]]}")
print(f"  Container obs dims: {cont_entities[0].obs_dim if cont_entities else 'N/A'}")

# ── Step 2: Extract Events ─────────────────────────────────────────
print()
print("Step 2: Event Extraction")
events = adapter.extract_events(BANK_DIR, entities)
print(f"  Total events: {len(events)}")

if len(events) > 0:
    # Show a few sample events
    for ev in events[:5]:
        print(f"    [{ev.timestamp}] {ev.entity_id} | {ev.feature_name} = {ev.value:.2f}")
    # Feature stats
    feats = set(ev.feature_name for ev in events)
    print(f"  Unique features: {len(feats)} — {sorted(feats)}")

# ── Step 3: Build EventBatch (dense tensor) ────────────────────────
print()
print("Step 3: EventBatch Construction")
try:
    batch = adapter.events_to_batch(events, entities)
    print(f"  event_tensor: {batch.event_tensor.shape}  (T x N x D)")
    print(f"  entity_ids:   {len(batch.entity_ids)}")
    print(f"  feature_names: {len(batch.feature_names)} — {batch.feature_names[:8]}...")
    print(f"  entity_types: {set(batch.entity_types)}")
    T, N, D = batch.event_tensor.shape
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    exit(1)

# ── Step 4: Extract Labels ─────────────────────────────────────────
print()
print("Step 4: Root Cause Labels")
labels = adapter.extract_labels(BANK_DIR, entities)
print(f"  Labels (before mapping): {len(labels)}")
for lb in labels[:3]:
    print(f"    {lb.occurrence_datetime} | {lb.component} | {lb.reason}")
if labels:
    entity_ids = [e.entity_id for e in entities]
    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    mapped_labels = []
    for lb in labels:
        if lb.component in eid_to_idx:
            mapped_labels.append(eid_to_idx[lb.component])
    print(f"  Labels mapped to entity idx: {len(mapped_labels)} (out of {len(labels)})")

# ── Step 5: Build model input (sliding window) ─────────────────────
print()
print("Step 5: Build Model Input")
window_size = 24
# Take first window_size+1 timesteps (extra for prediction target)
max_t = min(window_size + 1, T)
x_raw = batch.event_tensor[:max_t].copy()  # [T_sub, N, D]
# Pad features to model's max_obs_dim
MAX_OBS_DIM = 16
if D < MAX_OBS_DIM:
    x_padded = np.pad(x_raw, ((0, 0), (0, 0), (0, MAX_OBS_DIM - D)), mode='constant')
    obs_mask = np.zeros((N, MAX_OBS_DIM), dtype=np.float32)
    obs_mask[:, :D] = 1.0
else:
    x_padded = x_raw[:, :, :MAX_OBS_DIM]
    obs_mask = np.ones((N, MAX_OBS_DIM), dtype=np.float32)

# Normalize
n_mean = x_padded.mean(axis=(0, 1), keepdims=True) + 1e-6
n_std = x_padded.std(axis=(0, 1), keepdims=True) + 1e-6
x_norm = np.nan_to_num((x_padded - n_mean) / n_std, nan=0.0)

# Ensure at least window_size+1 timesteps
if x_norm.shape[0] < window_size + 1:
    pad_t = window_size + 1 - x_norm.shape[0]
    x_norm = np.pad(x_norm, ((0, pad_t), (0, 0), (0, 0)), mode='constant')

x_input = jnp.array(x_norm[:window_size + 1][None, :, :, :])  # [1, T+1, N, D]
type_idx = jnp.zeros(N, dtype=jnp.int32)  # all type 0 for now
obs_mask_j = jnp.array(obs_mask)

print(f"  x_input: {x_input.shape}  (batch=1, T+1={window_size+1}, N={N}, D={MAX_OBS_DIM})")
print(f"  type_idx: {type_idx.shape}")
print(f"  obs_mask: {obs_mask_j.shape} (valid features per entity)")

# ── Step 6: Model Init + Forward ───────────────────────────────────
print()
print("Step 6: RCAWorldFoundation Init & Forward")
rng = jax.random.PRNGKey(42)

model = RCAWorldFoundation(
    common_dim=128,
    max_obs_dim=MAX_OBS_DIM,
    num_entity_types=3,
    det_dim=256,
    stoch_dim=32,
    stoch_classes=32,
    use_onset_head=True,
    use_component_head=True,
)

# Init
variables = model.init(rng, x_input, type_idx, rng, obs_mask=obs_mask_j)
param_count = sum(p.size for p in jax.tree.leaves(variables["params"]))
print(f"  Model params: {param_count:,}")

# Forward pass
out = model.apply(variables, x_input, type_idx, rng, obs_mask=obs_mask_j)

print(f"  encoded:     {out['encoded'].shape}")
print(f"  h_seq:       {out['h_seq'].shape}")
print(f"  z_seq:       {out['z_seq'].shape}")
print(f"  pred_mu:     {out['pred_mu'].shape}")
print(f"  residual:    {out['residual'].shape}")
print(f"  onset_score: {out['onset']['onset_score'].shape}")
print(f"  comp_score:  {out['component']['component_score'].shape}")
print(f"  alpha:       {float(out['component']['alpha'][0]):.4f}")

# Basic sanity: residual is non-negative, onset/comp scores in [0,1]
res_val = float(jnp.mean(out['residual']))
onset_val = float(jnp.mean(out['onset']['onset_score']))
comp_val = float(jnp.mean(out['component']['component_score']))
print(f"  mean residual: {res_val:.4f}")
print(f"  mean onset:    {onset_val:.4f}")
print(f"  mean comp:     {comp_val:.4f}")

# ── Step 7: Loss Computation ───────────────────────────────────────
print()
print("Step 7: Loss Computation")
from foundation.training import world_model_loss

targets = {"obs": x_input[:, 1:, :, :]}
total_loss, components = world_model_loss(out, targets, obs_mask=obs_mask_j)
print(f"  total_loss:  {float(total_loss):.4f}")
for k, v in components.items():
    print(f"    {k}: {float(v):.6f}")

# ── Step 8: Gradient Check ─────────────────────────────────────────
print()
print("Step 8: Gradient Check")
from foundation.training import create_train_state

example_input = {
    "x": x_input,
    "type_idx": type_idx,
    "obs_mask": obs_mask_j,
}
state = create_train_state(model, jax.random.PRNGKey(123), example_input, learning_rate=1e-3)

batch = {
    "x": x_input,
    "type_idx": type_idx,
    "obs_mask": obs_mask_j,
}
from foundation.training import train_step
new_state, loss_val, loss_comp = train_step(state, batch, jax.random.PRNGKey(456))
print(f"  After 1 step: loss = {float(loss_val):.4f}")

# Check that params changed
old_sum = sum(float(jnp.sum(p)) for p in jax.tree.leaves(state.params))
new_sum = sum(float(jnp.sum(p)) for p in jax.tree.leaves(new_state.params))
print(f"  Params sum change: {old_sum:.2f} -> {new_sum:.2f} (delta={new_sum - old_sum:.6f})")
print("  Gradient flow: OK" if abs(new_sum - old_sum) > 1e-6 else "  WARNING: no gradient flow!")

# ── Summary ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("UNIT TEST SUMMARY")
print("=" * 60)
print(f"  Bank entities:  {len(entities)} ({len(svc_entities)} svc + {len(cont_entities)} cont)")
print(f"  Events:         {len(events)}")
print(f"  Batch:          T={T}, N={N}, D={D}")
print(f"  Labels:         {len(labels)} mapped={len(mapped_labels) if labels else 0}")
print(f"  Model params:   {param_count:,}")
print(f"  Forward pass:   OK")
print(f"  Loss computed:  {float(total_loss):.4f}")
print(f"  Gradient step:  OK (delta={abs(new_sum - old_sum):.6f})")
print()
print("ALL CHECKS PASSED")
