"""Hierarchical RSSM for heterogeneous entity dynamics.

Follows the same Flax pattern as the working src/models/rssm.py:
setup() for all submodules, no @nn.compact on methods,
nn.Sequential for MLPs.
"""

from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class RSSMState:
    def __init__(self, h, z, z_logits):
        self.h = h
        self.z = z
        self.z_logits = z_logits


class HierarchicalRSSM(nn.Module):
    latent_dim: int = 128
    det_dim: int = 256
    stoch_dim: int = 32
    stoch_classes: int = 32
    num_entity_types: int = 10
    use_graph: bool = False

    def setup(self):
        self.transition = nn.Sequential([
            nn.Dense(self.det_dim),
            nn.tanh,
        ])
        self.prior_net = nn.Sequential([
            nn.Dense(self.det_dim),
            nn.elu,
            nn.Dense(self.stoch_dim * self.stoch_classes),
        ])
        self.post_net = nn.Sequential([
            nn.Dense(self.det_dim),
            nn.elu,
            nn.Dense(self.stoch_dim * self.stoch_classes),
        ])
        self.type_trans_emb = self.param(
            "type_trans_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_entity_types, self.det_dim),
            jnp.float32,
        )

    @staticmethod
    def initial_state(batch_size: int, N: int, det_dim: int,
                      stoch_dim: int, stoch_classes: int) -> RSSMState:
        h = jnp.zeros((batch_size, N, det_dim), dtype=jnp.float32)
        z = jnp.zeros((batch_size, N, stoch_dim, stoch_classes), dtype=jnp.float32)
        z_logits = jnp.zeros((batch_size, N, stoch_dim, stoch_classes), dtype=jnp.float32)
        return RSSMState(h=h, z=z, z_logits=z_logits)

    def one_step(self, rng, prev_state, obs_t, type_idx, ctx_t, use_posterior):
        batch, N = obs_t.shape[0], obs_t.shape[1]
        S, C = self.stoch_dim, self.stoch_classes

        z_flat = prev_state.z.reshape(batch, N, -1)

        # Type modulation on deterministic state
        type_mod = self.type_trans_emb[type_idx]  # [N, det_dim]
        type_mod = type_mod[None, :, :]           # [1, N, det_dim]
        h_mod = prev_state.h + 0.1 * jnp.broadcast_to(type_mod, (batch, N, self.det_dim))

        # Transition
        trans_feats = [h_mod, z_flat]
        if ctx_t is not None:
            ctx_exp = jnp.broadcast_to(ctx_t[:, None, :], (batch, N, ctx_t.shape[-1]))
            trans_feats.append(ctx_exp)
        trans_in = jnp.concatenate(trans_feats, axis=-1)
        h_t = self.transition(trans_in)

        # Prior
        prior_logits = self.prior_net(h_t)
        prior_logits = prior_logits.reshape(batch, N, S, C)

        # Posterior
        if use_posterior:
            post_in = jnp.concatenate([h_t, obs_t], axis=-1)
            post_logits = self.post_net(post_in)
            post_logits = post_logits.reshape(batch, N, S, C)
            rng, sample_rng = jax.random.split(rng)
            z_t = self._sample(sample_rng, post_logits)
        else:
            post_logits = prior_logits
            rng, sample_rng = jax.random.split(rng)
            z_t = self._sample(sample_rng, prior_logits)

        new_state = RSSMState(h=h_t, z=z_t, z_logits=post_logits)
        return new_state, prior_logits, post_logits

    def _sample(self, rng, logits):
        batch, N, S, C = logits.shape
        logits_2d = logits.reshape(-1, C)
        sample_1d = jax.random.categorical(rng, logits_2d)
        sample_oh = jax.nn.one_hot(sample_1d, C)
        sample = sample_oh.reshape(batch, N, S, C)
        probs = jax.nn.softmax(logits)
        return sample + probs - jax.lax.stop_gradient(probs)

    def __call__(self,
                 rng,
                 observations,
                 type_idx,
                 contexts=None,
                 initial_state=None,
                 use_posterior=True,
                 ) -> Dict:
        batch, T, N, _ = observations.shape
        S, C = self.stoch_dim, self.stoch_classes

        if initial_state is None:
            state = self.initial_state(batch, N, self.det_dim, S, C)
        else:
            state = initial_state

        h_list, z_list, prior_list, post_list = [], [], [], []

        for t in range(T):
            obs_t = observations[:, t, :, :]
            ctx_t = contexts[:, t, :] if contexts is not None else None

            rng, step_rng = jax.random.split(rng)
            state, prior, post = self.one_step(
                step_rng, state, obs_t, type_idx, ctx_t, use_posterior)

            h_list.append(state.h)
            z_list.append(state.z)
            prior_list.append(prior)
            post_list.append(post)

        result = {
            "h_seq": jnp.stack(h_list, axis=1),
            "z_seq": jnp.stack(z_list, axis=1),
            "prior_logits": jnp.stack(prior_list, axis=1),
            "post_logits": jnp.stack(post_list, axis=1),
            "final_state": state,
        }
        return result

    def get_latent_shift(self, h_seq: jnp.ndarray) -> jnp.ndarray:
        diff = h_seq[:, 1:, :, :] - h_seq[:, :-1, :, :]
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
