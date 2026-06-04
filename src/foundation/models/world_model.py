"""RCAWorld-Foundation World Model: unified diagnostic world model.

Phase A implementation: encodes heterogeneous entity observations into a
common latent space, learns RSSM dynamics, detects onset times, and localizes
root cause entities via parallel p(c|t) + p(c) paths.
"""

from typing import Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .entity_tokenizer import MetricEncoder, TypeAwareDecoder
from .hierarchical_rssm import HierarchicalRSSM, RSSMState
from .onset_head import OnsetHead
from .parallel_component_head import ParallelComponentHead


class RCAWorldFoundation(nn.Module):
    common_dim: int = 128
    max_obs_dim: int = 16
    num_entity_types: int = 10
    det_dim: int = 256
    stoch_dim: int = 32
    stoch_classes: int = 32
    use_graph: bool = False
    use_onset_head: bool = True
    use_component_head: bool = True
    onset_hidden_dim: int = 128
    comp_hidden_dim: int = 128

    def setup(self):
        self.encoder = MetricEncoder(
            common_dim=self.common_dim,
            max_obs_dim=self.max_obs_dim,
            num_entity_types=self.num_entity_types,
            name="encoder",
        )
        self.decoder = TypeAwareDecoder(
            common_dim=self.common_dim,
            max_obs_dim=self.max_obs_dim,
            num_entity_types=self.num_entity_types,
            name="decoder",
        )
        self.rssm = HierarchicalRSSM(
            latent_dim=self.common_dim,
            det_dim=self.det_dim,
            stoch_dim=self.stoch_dim,
            stoch_classes=self.stoch_classes,
            num_entity_types=self.num_entity_types,
            use_graph=self.use_graph,
            name="rssm",
        )
        self.state_proj = nn.Dense(self.common_dim, name="state_proj")

        if self.use_onset_head:
            self.onset_head = OnsetHead(
                hidden_dim=self.onset_hidden_dim, name="onset_head")
        if self.use_component_head:
            self.component_head = ParallelComponentHead(
                hidden_dim=self.comp_hidden_dim, name="component_head")

    def encode_observations(self, x, type_idx, obs_mask=None):
        B, T, N, _ = x.shape
        x_flat = x.reshape(B * T, N, self.max_obs_dim)
        encoded_flat = self.encoder(x_flat, type_idx, obs_mask)
        return encoded_flat.reshape(B, T, N, self.common_dim)

    def decode_predictions(self, state_for_pred, type_idx):
        lead_dims = state_for_pred.shape[:-2]
        N_h = state_for_pred.shape[-2:]
        flat = state_for_pred.reshape(-1, *N_h)
        mu_f, ls_f = self.decoder(flat, type_idx)
        mu = mu_f.reshape(*lead_dims, N_h[0], self.max_obs_dim)
        ls = ls_f.reshape(*lead_dims, N_h[0], self.max_obs_dim)
        return mu, ls

    def compute_state_for_prediction(self, h_seq, z_seq):
        # h_seq[t] is state after processing obs[t]; it predicts obs[t+1].
        # h_seq has T entries; z_seq has T entries.
        # We need T predictions for x_target (obs[1:]), which also has T entries.
        # Thus no slicing needed.
        h_pred = h_seq  # [B, T, N, det_dim]
        z_flat = z_seq.reshape(*z_seq.shape[:2], z_seq.shape[2], -1)
        combined = jnp.concatenate([h_pred, z_flat], axis=-1)
        B, T, N, _ = combined.shape
        combined_flat = combined.reshape(B * T * N, -1)
        proj_flat = self.state_proj(combined_flat)
        return proj_flat.reshape(B, T, N, self.common_dim)

    def compute_residual(self, obs_true, obs_pred_mu, obs_pred_logsigma, obs_mask=None):
        sigma = jnp.exp(obs_pred_logsigma) + 1e-6
        nll = ((obs_true - obs_pred_mu) ** 2) / (2 * sigma ** 2) + obs_pred_logsigma
        if obs_mask is not None:
            m = obs_mask[None, None, :, :]
            nll = nll * m
            valid = jnp.sum(m, axis=-1, keepdims=True) + 1e-8
            nll = jnp.sum(nll, axis=-1) / valid[..., 0]
        else:
            nll = jnp.mean(nll, axis=-1)
        return nll

    def __call__(self, x, type_idx, rng, obs_mask=None, edge_index=None,
                 ctx_seq=None, use_posterior=True):
        x_rssm = x[:, :-1, :, :]    # [B, T, N, D]
        x_target = x[:, 1:, :, :]   # [B, T, N, D]
        ctx_rssm = ctx_seq[:, :-1] if ctx_seq is not None else None

        encoded = self.encode_observations(x_rssm, type_idx, obs_mask)

        # RSSM: (rng, obs, type_idx, contexts)
        rssm_out = self.rssm(rng, encoded, type_idx, ctx_rssm,
                             use_posterior=use_posterior)

        # Predict future
        state_for_pred = self.compute_state_for_prediction(
            rssm_out["h_seq"], rssm_out["z_seq"])
        pred_mu, pred_logsigma = self.decode_predictions(
            state_for_pred, type_idx)

        residual = self.compute_residual(
            x_target, pred_mu, pred_logsigma, obs_mask)

        result = {
            "encoded": encoded,
            "h_seq": rssm_out["h_seq"],
            "z_seq": rssm_out["z_seq"],
            "prior_logits": rssm_out["prior_logits"],
            "post_logits": rssm_out["post_logits"],
            "pred_mu": pred_mu,
            "pred_logsigma": pred_logsigma,
            "residual": residual,
            "final_state": rssm_out["final_state"],
        }

        if self.use_onset_head:
            onset_out = self.onset_head(
                residual=residual,
                h_seq=rssm_out["h_seq"],
                edge_index=edge_index,
            )
            result["onset"] = onset_out

        if self.use_component_head:
            onset_scores = result.get("onset", {}).get(
                "onset_score", jnp.ones_like(residual))
            h_for_comp = rssm_out["h_seq"]   # [B, T, N, det_dim]
            component_out = self.component_head(
                h_seq=h_for_comp,
                residual=residual,
                onset_scores=onset_scores,
                edge_index=edge_index,
            )
            result["component"] = component_out

        return result

    def diagnose(self, x, type_idx, rng, obs_mask=None, edge_index=None,
                 ctx_seq=None):
        out = self(x, type_idx, rng, obs_mask, edge_index, ctx_seq,
                   use_posterior=True)
        diagnosis = {"residual": out["residual"]}

        if "onset" in out:
            onset_scores = out["onset"]["onset_score"]
            diagnosis["onset_scores"] = onset_scores
            diagnosis["onset_times"] = jnp.argmax(onset_scores, axis=1)

        if "component" in out:
            comp = out["component"]
            diagnosis["component_scores"] = comp["component_score"]
            diagnosis["entity_ranking"] = jnp.argsort(
                -comp["component_score"], axis=-1)
            diagnosis["p_c_given_t"] = comp["p_c_given_t"]
            diagnosis["p_c_marginal"] = comp["p_c_marginal"]
            diagnosis["alpha"] = comp["alpha"]

        return diagnosis
