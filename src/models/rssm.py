"""RSSM (Recurrent State-Space Model) core for microservice dynamics.

Per-node deterministic + stochastic latent state dynamics.
Uses JIT-traced for-loop for unrolling (avoids nn.scan compatibility issues).
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, NamedTuple, Optional


class RSSMState(NamedTuple):
    h: jnp.ndarray        # deterministic state [batch, N, D_det]
    z: jnp.ndarray        # stochastic state [batch, N, D_stoch, C]
    z_logits: jnp.ndarray  # logits [batch, N, D_stoch, C]


class RSSM(nn.Module):
    det_dim: int = 256
    stoch_dim: int = 32
    stoch_classes: int = 32

    def setup(self):
        z_dim = self.stoch_dim * self.stoch_classes

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

    @staticmethod
    def get_initial_state(batch: int, N: int, det_dim: int, stoch_dim: int, stoch_classes: int) -> RSSMState:
        return RSSMState(
            h=jnp.zeros((batch, N, det_dim)),
            z=jnp.zeros((batch, N, stoch_dim, stoch_classes)),
            z_logits=jnp.zeros((batch, N, stoch_dim, stoch_classes)),
        )

    def one_step(
        self,
        rng: jax.random.PRNGKey,
        prev_state: RSSMState,
        obs_t: jnp.ndarray,
        ctx_t: jnp.ndarray,
        use_posterior: bool,
    ) -> Tuple[RSSMState, jnp.ndarray, jnp.ndarray]:
        """Single RSSM step.

        Args:
            rng: PRNG key
            prev_state: previous RSSMState
            obs_t: observation [batch, N, D_obs]
            ctx_t: workload context [batch, D_ctx]
            use_posterior: use posterior (training) or prior (inference)
        Returns:
            new_state: updated RSSMState
            prior_logits: [batch, N, D_stoch, C]
            post_logits: [batch, N, D_stoch, C]
        """
        batch, N = obs_t.shape[0], obs_t.shape[1]
        D_ctx = ctx_t.shape[-1]

        z_flat = prev_state.z.reshape(batch, N, -1)
        ctx_exp = jnp.broadcast_to(ctx_t[:, None, :], (batch, N, D_ctx))

        # h_t = tanh(W @ [h_{t-1}, z_{t-1}, ctx_t])
        trans_in = jnp.concatenate([prev_state.h, z_flat, ctx_exp], axis=-1)
        h_t = self.transition(trans_in)

        # Prior: h_t -> logits
        prior_logits = self.prior_net(h_t)
        prior_logits = prior_logits.reshape(batch, N, self.stoch_dim, self.stoch_classes)

        # Posterior
        if use_posterior:
            post_in = jnp.concatenate([h_t, obs_t], axis=-1)
            post_logits = self.post_net(post_in)
            post_logits = post_logits.reshape(batch, N, self.stoch_dim, self.stoch_classes)

            rng, sample_rng = jax.random.split(rng)
            z_t = self._sample(sample_rng, post_logits)
        else:
            post_logits = prior_logits
            rng, sample_rng = jax.random.split(rng)
            z_t = self._sample(sample_rng, prior_logits)

        new_state = RSSMState(h=h_t, z=z_t, z_logits=post_logits)
        return new_state, prior_logits, post_logits

    def _sample(self, rng: jax.random.PRNGKey, logits: jnp.ndarray) -> jnp.ndarray:
        """Sample one-hot z with straight-through gradient."""
        batch, N, S, C = logits.shape
        logits_2d = logits.reshape(-1, C)
        sample_1d = jax.random.categorical(rng, logits_2d)
        sample_oh = jax.nn.one_hot(sample_1d, C)
        sample = sample_oh.reshape(batch, N, S, C)

        probs = jax.nn.softmax(logits)
        return sample + probs - jax.lax.stop_gradient(probs)

    def __call__(
        self,
        rng: jax.random.PRNGKey,
        observations: jnp.ndarray,
        contexts: jnp.ndarray,
        initial_state: Optional[RSSMState] = None,
        use_posterior: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, RSSMState]:
        """Unroll RSSM over time.

        Args:
            rng: PRNG key
            observations: [batch, T, N, D_obs]
            contexts: [batch, T, D_ctx]
            initial_state: optional initial RSSMState
            use_posterior: use posterior during unroll
        Returns:
            h_seq: [batch, T, N, D_det]
            z_seq: [batch, T, N, D_stoch, C]
            prior_seq: [batch, T, N, D_stoch, C]
            post_seq: [batch, T, N, D_stoch, C]
            final_state: RSSMState
        """
        batch, T, N, D_obs = observations.shape
        S, C = self.stoch_dim, self.stoch_classes

        if initial_state is None:
            state = self.get_initial_state(batch, N, self.det_dim, S, C)
        else:
            state = initial_state

        h_list, z_list, prior_list, post_list = [], [], [], []

        for t in range(T):
            obs_t = observations[:, t, :, :]
            ctx_t = contexts[:, t, :]

            rng, step_rng = jax.random.split(rng)
            state, prior, post = self.one_step(step_rng, state, obs_t, ctx_t, use_posterior)

            h_list.append(state.h)
            z_list.append(state.z)
            prior_list.append(prior)
            post_list.append(post)

        h_seq = jnp.stack(h_list, axis=1)
        z_seq = jnp.stack(z_list, axis=1)
        prior_seq = jnp.stack(prior_list, axis=1)
        post_seq = jnp.stack(post_list, axis=1)

        return h_seq, z_seq, prior_seq, post_seq, state
