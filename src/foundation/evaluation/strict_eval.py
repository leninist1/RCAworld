"""Phase A-Hardening: Strict evaluation harness.

Key rules (enforced):
  1. Candidate generation MUST NOT read GT component (rc).
  2. Time localization MUST NOT read GT onset time.
  3. One episode per fault query (query-context window).
  4. prior-only as primary result; posterior only as ablation.
  5. OpenRCA data for EVALUATION ONLY — zero training.

Joint scoring approach:
  S[t, c] = lambda_r * R[t,c] + lambda_s * DeltaZ[t,c] + lambda_e * EarlyRise[t,c]
  (t_hat, c_hat) = argmax_{t,c} S[t,c]

Then:
  p(c | O) = sum_t S[t,c]     (marginal component score)
  p(t | O) = sum_c S[t,c]     (marginal onset score)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QueryEpisode:
    """One fault query with its time context window (NO gt labels in scoring)."""
    query_id: str
    system: str
    # Data within the query's time window [T, N, D]
    tensor: np.ndarray            # [T, N, D] normalized observations
    entity_ids: List[str]         # [N]
    entity_types: List[str]       # [N] e.g. 'container', 'database'
    type_idx: np.ndarray          # [N] integer type indices
    obs_mask: np.ndarray          # [N, D] valid feature mask
    # Ground truth (ONLY used for metric computation, NEVER for candidate gen)
    gt_component: str = ""
    gt_component_idx: int = -1
    gt_onset_ts: float = 0.0


@dataclass
class JointScores:
    """Joint time-component scores for one episode."""
    S: np.ndarray          # [T, N] joint score matrix
    residual: np.ndarray   # [T, N] prediction residuals
    shift: np.ndarray      # [T, N] latent state shift
    early_rise: np.ndarray # [T, N] temporal early-rise
    onset_raw: np.ndarray  # [T, N] raw onset head output

    @property
    def component_score(self) -> np.ndarray:
        """Marginal: p(c | O) = sum_t S[t,c]"""
        return self.S.sum(axis=0)  # [N]

    @property
    def onset_score(self) -> np.ndarray:
        """Marginal: p(t | O) = sum_c S[t,c]"""
        return self.S.sum(axis=1)  # [T]

    @property
    def top_component(self) -> int:
        return int(np.argmax(self.component_score))

    @property
    def top_onset(self) -> int:
        return int(np.argmax(self.onset_score))

    @property
    def joint_argmax(self) -> Tuple[int, int]:
        """(t_hat, c_hat) = argmax_{t,c} S[t,c]"""
        idx = int(np.argmax(self.S))
        T, N = self.S.shape
        return (idx // N, idx % N)


def compute_joint_scores(
    residuals: np.ndarray,
    latents: np.ndarray,
    model_onset_scores: np.ndarray,
    lambda_residual: float = 1.0,
    lambda_shift: float = 0.5,
    lambda_early: float = 0.3,
    lambda_onset: float = 0.0,  # 0 = don't use model onset head (prior-only safe)
    temporal_smooth_window: int = 3,
) -> JointScores:
    """Compute joint S[t,c] score matrix WITHOUT reading GT.

    Args:
        residuals:    [T, N] prediction residuals (per timestep, per entity)
        latents:      [T, N, D] latent states or None (for shift computation)
        model_onset_scores: [T, N] raw onset head output (may be None)
        lambda_*:     Weights for each signal
        temporal_smooth_window: Window size for smoothing

    Returns:
        JointScores with S[t,c], component_score, onset_score, etc.
    """
    T, N = residuals.shape

    # 1. Residual signal (normalize to [0,1] per entity)
    R = residuals.copy()
    R_max = np.max(R, axis=0, keepdims=True) + 1e-8
    R = R / R_max
    R = np.clip(R, 0, 1)

    # 2. Latent shift signal
    if latents is not None and latents.shape[0] >= 2:
        diff = np.sqrt(np.sum((latents[1:] - latents[:-1]) ** 2, axis=-1))  # [T-1, N]
        # Pad first timestep
        Z = np.zeros((T, N), dtype=np.float32)
        Z[1:] = diff
        Z_max = np.max(Z, axis=0, keepdims=True) + 1e-8
        Z = Z / Z_max
    else:
        Z = np.zeros((T, N), dtype=np.float32)

    # 3. Temporal early-rise signal (exponential moving average of positive changes)
    E = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        for t in range(1, T):
            delta = max(0, R[t, n] - R[t-1, n])
            E[t, n] = 0.7 * E[t-1, n] + 0.3 * delta
    E_max = np.max(E, axis=0, keepdims=True) + 1e-8
    E = E / E_max

    # 4. Temporal smoothing (simple moving average)
    def smooth(x, w):
        if w <= 1:
            return x
        k = np.ones(w) / w
        out = np.zeros_like(x)
        for n in range(N):
            out[:, n] = np.convolve(x[:, n], k, mode='same')
        return out

    R_s = smooth(R, temporal_smooth_window)
    Z_s = smooth(Z, temporal_smooth_window)
    E_s = smooth(E, temporal_smooth_window)

    # 5. Joint score (prior-only: no model onset head)
    S = (lambda_residual * R_s +
         lambda_shift * Z_s +
         lambda_early * E_s)

    if model_onset_scores is not None and lambda_onset > 0:
        O_s = smooth(model_onset_scores, temporal_smooth_window)
        O_max = np.max(O_s, axis=0, keepdims=True) + 1e-8
        O_s = O_s / O_max
        S = S + lambda_onset * O_s

    return JointScores(
        S=S,
        residual=R_s,
        shift=Z_s,
        early_rise=E_s,
        onset_raw=model_onset_scores if model_onset_scores is not None else np.zeros((T, N)),
    )


def evaluate_joint(
    joint: JointScores,
    gt_component_idx: int,
    gt_onset_ts: float,
    timestamps: np.ndarray,
    onset_tolerance_steps: int = 3,
) -> Dict:
    """Evaluate joint (t,c) prediction against ground truth.

    Returns component rank, time error, and joint accuracy.
    """
    T, N = joint.S.shape

    # Component ranking (marginal)
    comp_score = joint.component_score  # [N]
    comp_rank = int(np.sum(comp_score > comp_score[gt_component_idx])) + 1

    # Onset time (marginal)
    onset_score = joint.onset_score  # [T]
    pred_t = int(np.argmax(onset_score))
    gt_t = int(np.argmin(np.abs(timestamps - gt_onset_ts))) if len(timestamps) > 0 else 0
    time_error = abs(pred_t - gt_t)
    time_hit = time_error <= onset_tolerance_steps

    # Joint argmax
    jt, jc = joint.joint_argmax
    comp_joint_hit = (jc == gt_component_idx)
    time_joint_hit = (abs(jt - gt_t) <= onset_tolerance_steps)
    joint_hit = comp_joint_hit and time_joint_hit

    return {
        "component_rank": comp_rank,
        "component_top1": comp_rank == 1,
        "component_top3": comp_rank <= 3,
        "time_error": time_error,
        "time_hit": time_hit,
        "joint_component_hit": comp_joint_hit,
        "joint_time_hit": time_joint_hit,
        "joint_hit": joint_hit,
        "reciprocal_rank": 1.0 / max(1, comp_rank),
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregate per-query metrics into summary statistics."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "component_top1": np.mean([r["component_top1"] for r in results]),
        "component_top3": np.mean([r["component_top3"] for r in results]),
        "mrr": np.mean([r["reciprocal_rank"] for r in results]),
        "avg_rank": np.mean([r["component_rank"] for r in results]),
        "time_hit_rate": np.mean([r["time_hit"] for r in results]),
        "time_mae": np.mean([r["time_error"] for r in results]),
        "joint_hit_rate": np.mean([r["joint_hit"] for r in results]),
        "joint_component_hit": np.mean([r["joint_component_hit"] for r in results]),
        "joint_time_hit": np.mean([r["joint_time_hit"] for r in results]),
    }
