"""Phase 0.8 validation: prediction residuals for fault detection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from models.graph_rssm import GraphRSSM
from training.losses import graph_rssm_loss, gaussian_nll_per_node
from data.synthetic import SyntheticMicroserviceSystem
import optax
from flax.training import train_state
from tqdm import tqdm

print(f'Device: {jax.devices()[0]}')

# Generate data
system = SyntheticMicroserviceSystem(seed=42)
normal_eps, fault_eps = system.generate_dataset(normal_episodes=8, fault_episodes=5, steps_per_episode=400)

# Create training windows from normal episodes
window_sz = 24
def make_windows(eps):
    wins = []
    for ep in eps:
        T = ep['node_features'].shape[0]
        for s in range(0, T - window_sz, window_sz // 3):
            wins.append({'n': ep['node_features'][s:s+window_sz],
                         'e': ep['edge_features'][s:s+window_sz],
                         'w': ep['workload_context'][s:s+window_sz]})
    return wins

train_wins = make_windows(normal_eps)
print(f'Train windows: {len(train_wins)}')

edge_idx = system.edge_index

# Create model
model = GraphRSSM(
    node_dim=6, edge_dim=3, ctx_dim=3,
    graph_hidden=64, num_heads=4,
    det_dim=128, stoch_dim=16, stoch_classes=16,
)

rng = jax.random.PRNGKey(42)
init_rng, rng = jax.random.split(rng)

b0 = train_wins[0]
params = model.init(init_rng, rng,
    jnp.array(b0['n'][None]), jnp.array(b0['e'][None]),
    jnp.array(edge_idx, dtype=jnp.int32), jnp.array(b0['w'][None]))

state = train_state.TrainState.create(apply_fn=model.apply, params=params,
    tx=optax.adam(1e-3))

# JIT-compiled training step
@jax.jit
def train_step(s, rng_key, bn, be, bw):
    def l_fn(p):
        out = s.apply_fn(p, rng_key, bn, be,
                         jnp.array(edge_idx, dtype=jnp.int32), bw,
                         use_posterior=True)
        nt = bn[:, 1:, :, :]
        et = be[:, 1:, :, :]
        total, _ = graph_rssm_loss(out, nt, et, kl_weight=0.1, edge_weight=0.3)
        return total
    l, g = jax.value_and_grad(l_fn)(s.params)
    return s.apply_gradients(grads=g), l

# JIT-compiled residual computation
@jax.jit
def compute_residuals(p, rng_key, bn, be, bw):
    out = model.apply(p, rng_key, bn, be,
                      jnp.array(edge_idx, dtype=jnp.int32), bw,
                      use_posterior=True)
    nt = bn[:, 1:, :, :]
    res = gaussian_nll_per_node(out['metrics_mu'], out['metrics_logsigma'], nt)
    return res  # [T-1, N]

# Train
bs = 16
n_steps = 1500
print('Training...')
for step in tqdm(range(n_steps)):
    idxs = np.random.choice(len(train_wins), bs, replace=True)
    bn = np.stack([train_wins[i]['n'] for i in idxs])
    be = np.stack([train_wins[i]['e'] for i in idxs])
    bw = np.stack([train_wins[i]['w'] for i in idxs])
    rng, srng = jax.random.split(rng)
    state, loss = train_step(state, srng, jnp.array(bn), jnp.array(be), jnp.array(bw))

print(f'Final train loss: {loss:.4f}')

# ---- Phase 0.8: Residual Analysis ----
print('\n' + '='*60)
print('Phase 0.8: Prediction Residual Analysis')
print('='*60)

# Get normal baseline residuals (per-timestep, per-node)
normal_res_all = []
for _ in range(5):
    ep = normal_eps[np.random.randint(len(normal_eps))]
    ws = np.random.randint(0, ep['node_features'].shape[0] - window_sz)
    n_nodes = ep['node_features'][ws:ws+window_sz][None]
    n_edges = ep['edge_features'][ws:ws+window_sz][None]
    n_wl = ep['workload_context'][ws:ws+window_sz][None]
    nres = np.array(compute_residuals(state.params, rng, jnp.array(n_nodes), jnp.array(n_edges), jnp.array(n_wl)))
    # nres: [1, T-1, N]
    normal_res_all.append(nres.reshape(-1))  # flatten all timesteps and nodes

normal_baseline = np.concatenate(normal_res_all)
threshold = np.mean(normal_baseline) + 3 * np.std(normal_baseline)
print(f'Normal residual baseline: mean={np.mean(normal_baseline):.4f}, std={np.std(normal_baseline):.4f}')
print(f'Anomaly threshold (μ+3σ): {threshold:.4f}')

# Test on fault episodes
results = []
for i, fep in enumerate(fault_eps):
    fi = fep['fault_info']
    # Use window around fault start
    ws = max(0, fi['fault_start'] - window_sz // 2)
    we = min(fep['node_features'].shape[0], ws + window_sz)
    if we - ws < window_sz:
        ws = max(0, we - window_sz)

    f_nodes = fep['node_features'][ws:we][None]
    f_edges = fep['edge_features'][ws:we][None]
    f_wl = fep['workload_context'][ws:we][None]

    fault_res = np.array(compute_residuals(state.params, rng, jnp.array(f_nodes), jnp.array(f_edges), jnp.array(f_wl)))
    # fault_res: [1, T-1, N]
    fault_res = fault_res[0]  # remove batch dim → [T-1, N]

    root_cause = fi['root_cause']
    fault_type = fi['fault_type']
    rc_name = system.services[root_cause].name

    # Average residual per node over fault window
    avg_res = np.mean(fault_res, axis=0)  # [N]

    # First time each node crosses threshold
    first_anomaly = np.full(8, 999)
    for t in range(fault_res.shape[0]):
        for n in range(8):
            if first_anomaly[n] == 999 and fault_res[t, n] > threshold:
                first_anomaly[n] = t

    rank = np.argsort(-avg_res)
    rc_rank = np.where(rank == root_cause)[0][0] + 1
    top1_correct = (rank[0] == root_cause)

    results.append({
        'fault': f'{fault_type}@{rc_name}',
        'rc_rank': rc_rank,
        'top1': top1_correct,
        'rc_first': first_anomaly[root_cause],
        'min_first': min(first_anomaly),
        'rc_earliest': first_anomaly[root_cause] == min(first_anomaly),
    })

    print(f'\nFault: {fault_type} on {rc_name} (svc {root_cause})')
    print(f'  Root cause rank by avg residual: {rc_rank}/8 {"✓" if top1_correct else "✗"}')
    print(f'  Root cause earliest anomalous: {"Yes" if first_anomaly[root_cause] <= min(first_anomaly) else "No"}')
    for r in rank[:3]:
        svc_name = system.services[r].name
        tag = ' ← ROOT CAUSE' if r == root_cause else ''
        print(f'    #{np.where(rank==r)[0][0]+1}: {svc_name}: avg_res={avg_res[r]:.3f}, first_anom_step={first_anomaly[r]}{tag}')

# Summary
print('\n' + '='*60)
print('Summary')
print('='*60)
top1_count = sum(1 for r in results if r['top1'])
earliest_count = sum(1 for r in results if r['rc_earliest'])
print(f'Top-1 accuracy: {top1_count}/{len(results)} ({100*top1_count/len(results):.0f}%)')
print(f'Root cause earliest anomaly: {earliest_count}/{len(results)} ({100*earliest_count/len(results):.0f}%)')
mean_rank = np.mean([r['rc_rank'] for r in results])
print(f'Mean rank of root cause: {mean_rank:.1f}')
