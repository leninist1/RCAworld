"""Phase A-Hardening: Strict evaluation harness with calibrated joint scoring.

Key rules (enforced):
  1. Candidate generation MUST NOT read GT component (rc).
  2. Time localization MUST NOT read GT onset time.
  3. One episode per fault query (query-context window).
  4. prior-only as primary result; posterior only as ablation.
  5. OpenRCA data for EVALUATION ONLY — zero training.

Scoring methods:
  A. Residual-only heuristic:        S[t,c] = R_norm[t,c]
  B. Residual + latent shift + early-rise heuristic:  S[t,c] = weighted sum
  C. Calibrated joint score (MAD-based):  cross-entity comparable z-scores
  D. Learned Onset Head:              S[t,c] = onset_head(residual, h_seq)[t,c]
  E. Learned Onset + Component Head:  S[t,c] = component_head(h_seq, residual, onset)[t,c]

Normalization approaches:
  - Per-entity max (OLD):  R[t,c] / max_t(R[t,c])  — destroys cross-entity comparison
  - Calibrated z-score (NEW):  (R[t,c] - median_c) / (MAD_c + epsilon)
    where median_c and MAD_c are estimated from burn-in (normal) data.
    This preserves which entity's anomaly is truly significant.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


@dataclass
class QueryEpisode:
    """One fault query with its time context window (NO gt labels in scoring)."""
    query_id: str
    system: str
    tensor: np.ndarray            # [T, N, D] normalized observations
    entity_ids: List[str]         # [N]
    entity_types: List[str]       # [N] e.g. 'container', 'database'
    type_idx: np.ndarray          # [N] integer type indices
    obs_mask: np.ndarray          # [T, N, D] valid feature mask (time-varying)
    timestamps: np.ndarray        # [T] timestamp grid

    query_window_start_ts: float = 0.0
    query_window_end_ts: float = 0.0

    gt_component: str = ""
    gt_component_idx: int = -1
    gt_onset_ts: float = 0.0

    @property
    def query_window_mask(self) -> np.ndarray:
        """[T] bool mask: True for timesteps inside the query observation window."""
        return (self.timestamps >= self.query_window_start_ts) & \
               (self.timestamps <= self.query_window_end_ts)


@dataclass
class JointScores:
    """Joint time-component scores for one episode."""
    S: np.ndarray          # [T, N] joint score matrix
    residual: np.ndarray   # [T, N] prediction residuals (raw or normalized)
    shift: np.ndarray      # [T, N] latent state shift
    early_rise: np.ndarray # [T, N] temporal early-rise
    onset_raw: np.ndarray  # [T, N] raw onset head output
    comp_raw: np.ndarray   # [N] raw component head output (optional)

    query_window_mask: Optional[np.ndarray] = None

    @property
    def component_score(self) -> np.ndarray:
        """Marginal: p(c | O) = max_t S[t,c] inside query window."""
        if self.query_window_mask is not None:
            # Only score within query observation window
            S_window = self.S.copy()
            S_window[~self.query_window_mask] = -np.inf
            return np.max(S_window, axis=0)  # [N]
        return np.max(self.S, axis=0)  # [N]

    @property
    def onset_score(self) -> np.ndarray:
        """Marginal: p(t | O) = sum_c S[t,c] inside query window.

        FIXED: time predictions are now constrained to the query observation
        window only. Burn-in timesteps are masked to -inf.
        """
        scores = self.S.sum(axis=1)  # [T]
        if self.query_window_mask is not None:
            scores = np.where(self.query_window_mask, scores, -np.inf)
        return scores

    @property
    def top_component(self) -> int:
        return int(np.argmax(self.component_score))

    @property
    def top_onset(self) -> int:
        return int(np.argmax(self.onset_score))

    @property
    def joint_argmax(self) -> Tuple[int, int]:
        """(t_hat, c_hat) = argmax_{t,c} S[t,c] inside query window."""
        if self.query_window_mask is not None:
            S_valid = self.S.copy()
            S_valid[~self.query_window_mask] = -np.inf
            idx = int(np.argmax(S_valid))
        else:
            idx = int(np.argmax(self.S))
        T, N = self.S.shape
        return (idx // N, idx % N)

    def decode_multiple_faults(self, max_faults: int = 1,
                                min_time_distance: int = 5) -> List[Tuple[int, int, float]]:
        """Decode multiple fault predictions with Non-Maximum Suppression.

        For multi-fault queries, returns a list of (t, c, score) tuples,
        each being a local maximum after suppressing previous peaks.

        Args:
            max_faults: Maximum number of faults to return.
            min_time_distance: Minimum timestep distance between faults.

        Returns:
            List of (time_idx, component_idx, score) sorted by time.
        """
        S = self.S.copy()
        if self.query_window_mask is not None:
            S[~self.query_window_mask] = -np.inf

        candidates = []
        for _ in range(max_faults):
            idx = int(np.argmax(S))
            score = S.flat[idx]
            if not np.isfinite(score) or score <= 0:
                break
            t, c = idx // S.shape[1], idx % S.shape[1]
            candidates.append((t, c, float(score)))

            # Suppress nearby time region
            left = max(0, t - min_time_distance)
            right = min(S.shape[0], t + min_time_distance + 1)
            S[left:right, :] = -np.inf

        return sorted(candidates, key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════
# Scoring methods
# ═══════════════════════════════════════════════════════════════════════

def compute_mad_normalization(
    values: np.ndarray,
    burn_in_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-entity median and MAD for z-score normalization.

    Args:
        values: [T, N] raw scores (e.g., residuals).
        burn_in_mask: [T] bool mask for burn-in (normal) period.

    Returns:
        median: [N] per-entity medians.
        mad: [N] per-entity MAD (scaled to match std of normal).
    """
    if burn_in_mask is not None and burn_in_mask.any():
        data = values[burn_in_mask, :]  # [T_norm, N]
    else:
        data = values

    median = np.median(data, axis=0)  # [N]
    mad = np.median(np.abs(data - median), axis=0) * 1.4826  # scale to ~std
    return median, mad


def normalize_zscore(values: np.ndarray, median: np.ndarray,
                     mad: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Compute z-score: (x - median) / (mad + epsilon).

    Args:
        values: [T, N] raw scores.
        median: [N] per-entity medians.
        mad: [N] per-entity MADs (scaled).

    Returns:
        z_scores: [T, N] normalized scores, clipped to [0, +inf).
    """
    z = (values - median[None, :]) / (mad[None, :] + epsilon)
    return np.maximum(z, 0)  # Only positive deviations (anomalies)


def compute_residual_only(
    residuals: np.ndarray,
    burn_in_mask: Optional[np.ndarray] = None,
) -> JointScores:
    """Method A: Residual-only heuristic scoring.

    Args:
        residuals: [T, N] prediction residuals.
        burn_in_mask: [T] bool mask for burn-in period.

    Returns:
        JointScores with S = calibrated residual z-scores.
    """
    T, N = residuals.shape

    # Calibrated z-score normalization
    median, mad = compute_mad_normalization(residuals, burn_in_mask)
    R_norm = normalize_zscore(residuals, median, mad)

    return JointScores(
        S=R_norm,
        residual=R_norm,
        shift=np.zeros((T, N)),
        early_rise=np.zeros((T, N)),
        onset_raw=np.zeros((T, N)),
        comp_raw=np.zeros(N),
    )


def compute_residual_shift_earlyrise(
    residuals: np.ndarray,
    latents: np.ndarray,
    burn_in_mask: Optional[np.ndarray] = None,
    lambda_residual: float = 1.0,
    lambda_shift: float = 0.5,
    lambda_early: float = 0.3,
    temporal_smooth_window: int = 3,
    model_onset_scores: Optional[np.ndarray] = None,
    lambda_onset: float = 0.0,
) -> JointScores:
    """Method B: Residual + latent shift + early-rise heuristic scoring.

    Uses calibrated z-score normalization (cross-entity comparable)
    instead of per-entity max normalization.

    Args:
        residuals: [T, N] prediction residuals.
        latents: [T, N, D] latent states.
        burn_in_mask: [T] bool mask for burn-in (normal) period.
        lambda_*: Weights for each signal.
        temporal_smooth_window: Smoothing window size.
        model_onset_scores: [T, N] learned onset head output (optional).
        lambda_onset: Weight for learned onset scores.

    Returns:
        JointScores with S[t,c].
    """
    T, N = residuals.shape

    # 1. Calibrated residual signal
    res_median, res_mad = compute_mad_normalization(residuals, burn_in_mask)
    R = normalize_zscore(residuals, res_median, res_mad)

    # 2. Latent shift signal (calibrated)
    if latents is not None and latents.shape[0] >= 2:
        diff = np.sqrt(np.sum((latents[1:] - latents[:-1]) ** 2, axis=-1))  # [T-1, N]
        Z_full = np.zeros((T, N), dtype=np.float32)
        Z_full[1:] = diff
        z_median, z_mad = compute_mad_normalization(Z_full, burn_in_mask)
        Z = normalize_zscore(Z_full, z_median, z_mad)
    else:
        Z = np.zeros((T, N), dtype=np.float32)

    # 3. Temporal early-rise signal (EMA of positive changes)
    E = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        for t in range(1, T):
            delta = max(0, R[t, n] - R[t-1, n])
            E[t, n] = 0.7 * E[t-1, n] + 0.3 * delta
    e_median, e_mad = compute_mad_normalization(E, burn_in_mask)
    E_norm = normalize_zscore(E, e_median, e_mad)

    # 4. Temporal smoothing
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
    E_s = smooth(E_norm, temporal_smooth_window)

    # 5. Weighted sum
    S = (lambda_residual * R_s +
         lambda_shift * Z_s +
         lambda_early * E_s)

    # 6. Optional learned onset head blending
    onset_raw_out = np.zeros((T, N))
    if model_onset_scores is not None and lambda_onset > 0:
        o_median, o_mad = compute_mad_normalization(model_onset_scores, burn_in_mask)
        O = normalize_zscore(model_onset_scores, o_median, o_mad)
        O_s = smooth(O, temporal_smooth_window)
        S += lambda_onset * O_s
        onset_raw_out = model_onset_scores

    return JointScores(
        S=S,
        residual=R_s,
        shift=Z_s,
        early_rise=E_s,
        onset_raw=onset_raw_out,
        comp_raw=np.zeros(N),
    )


def compute_learned_onset_only(
    residuals: np.ndarray,
    h_states: np.ndarray,
    onset_scores: np.ndarray,
    burn_in_mask: Optional[np.ndarray] = None,
    lambda_residual: float = 0.3,
    lambda_onset: float = 1.0,
) -> JointScores:
    """Method D: Learned Onset Head scoring.

    Args:
        residuals: [T, N] prediction residuals.
        h_states: [T, N, D] latent states.
        onset_scores: [T, N] learned onset head output.
        burn_in_mask: [T] bool mask for burn-in period.
        lambda_residual: Weight for calibrated residual.
        lambda_onset: Weight for learned onset scores.

    Returns:
        JointScores with S[t,c].
    """
    T, N = residuals.shape

    # Calibrated residual
    res_median, res_mad = compute_mad_normalization(residuals, burn_in_mask)
    R = normalize_zscore(residuals, res_median, res_mad)

    # Learned onset scores (normalized)
    o_median, o_mad = compute_mad_normalization(onset_scores, burn_in_mask)
    O = normalize_zscore(onset_scores, o_median, o_mad)

    S = lambda_residual * R + lambda_onset * O

    return JointScores(
        S=S,
        residual=R,
        shift=np.zeros((T, N)),
        early_rise=np.zeros((T, N)),
        onset_raw=onset_scores,
        comp_raw=np.zeros(N),
    )


# ═══════════════════════════════════════════════════════════════════════
# Scoring dispatcher (NO GT access)
# ═══════════════════════════════════════════════════════════════════════

def compute_joint_scores(
    residuals: np.ndarray,
    latents: np.ndarray,
    model_onset_scores: Optional[np.ndarray] = None,
    model_component_scores: Optional[np.ndarray] = None,
    method: str = "calibrated",
    burn_in_mask: Optional[np.ndarray] = None,
    lambda_residual: float = 1.0,
    lambda_shift: float = 0.5,
    lambda_early: float = 0.3,
    lambda_onset: float = 0.0,
    temporal_smooth_window: int = 3,
) -> JointScores:
    """Compute joint S[t,c] score matrix WITHOUT reading GT.

    Args:
        residuals:    [T, N] prediction residuals (per timestep, per entity).
        latents:      [T, N, D] latent states.
        model_onset_scores: [T, N] learned onset head output (optional).
        model_component_scores: [N] learned component head output (optional).
        method: Scoring method:
            "residual_only"          — Method A
            "residual_shift_early"   — Method B (calibrated z-score)
            "calibrated"             — Method B (same as above, default)
            "legacy_per_entity_max"  — Old per-entity max normalization (for comparison)
            "onset_head"             — Method D: learned onset head
        burn_in_mask: [T] bool mask for burn-in (normal) period.
        lambda_*: Weights for each signal.
        temporal_smooth_window: Smoothing window size.

    Returns:
        JointScores with S[t,c], component_score, onset_score, etc.
    """
    if method == "legacy_per_entity_max":
        return _compute_joint_legacy(
            residuals=residuals,
            latents=latents,
            model_onset_scores=model_onset_scores,
            lambda_residual=lambda_residual,
            lambda_shift=lambda_shift,
            lambda_early=lambda_early,
            lambda_onset=lambda_onset,
            temporal_smooth_window=temporal_smooth_window,
        )

    if method == "residual_only":
        return compute_residual_only(residuals, burn_in_mask)

    if method == "onset_head" and model_onset_scores is not None:
        return compute_learned_onset_only(
            residuals=residuals,
            h_states=latents,
            onset_scores=model_onset_scores,
            burn_in_mask=burn_in_mask,
            lambda_residual=lambda_residual,
            lambda_onset=lambda_onset,
        )

    # Default: calibrated residual + shift + early-rise
    return compute_residual_shift_earlyrise(
        residuals=residuals,
        latents=latents,
        burn_in_mask=burn_in_mask,
        lambda_residual=lambda_residual,
        lambda_shift=lambda_shift,
        lambda_early=lambda_early,
        temporal_smooth_window=temporal_smooth_window,
        model_onset_scores=model_onset_scores,
        lambda_onset=lambda_onset,
    )


def _compute_joint_legacy(
    residuals: np.ndarray,
    latents: np.ndarray,
    model_onset_scores: Optional[np.ndarray] = None,
    lambda_residual: float = 1.0,
    lambda_shift: float = 0.5,
    lambda_early: float = 0.3,
    lambda_onset: float = 0.0,
    temporal_smooth_window: int = 3,
) -> JointScores:
    """Legacy scoring method using per-entity max normalization.

    This is the OLD method preserved for comparison. It normalizes each entity
    by its own maximum, which destroys cross-entity comparability. Use
    `method="calibrated"` instead for production evaluation.

    Args:
        residuals:    [T, N] prediction residuals.
        latents:      [T, N, D] latent states.
        model_onset_scores: [T, N] raw onset head output.
        lambda_*:     Weights for each signal.
        temporal_smooth_window: Smoothing window size.

    Returns:
        JointScores with S[t,c], component_score, onset_score.
    """
    T, N = residuals.shape

    R = residuals.copy()
    R_max = np.max(R, axis=0, keepdims=True) + 1e-8
    R = R / R_max
    R = np.clip(R, 0, 1)

    if latents is not None and latents.shape[0] >= 2:
        diff = np.sqrt(np.sum((latents[1:] - latents[:-1]) ** 2, axis=-1))
        Z = np.zeros((T, N), dtype=np.float32)
        Z[1:] = diff
        Z_max = np.max(Z, axis=0, keepdims=True) + 1e-8
        Z = Z / Z_max
    else:
        Z = np.zeros((T, N), dtype=np.float32)

    E = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        for t in range(1, T):
            delta = max(0, R[t, n] - R[t-1, n])
            E[t, n] = 0.7 * E[t-1, n] + 0.3 * delta
    E_max = np.max(E, axis=0, keepdims=True) + 1e-8
    E = E / E_max

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
        comp_raw=np.zeros(N),
    )


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (GT ONLY used here, NOT in candidate generation)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_joint(
    joint: JointScores,
    gt_component_idx: int,
    gt_onset_ts: float,
    timestamps: np.ndarray,
    onset_tolerance_steps: int = 3,
) -> Dict:
    """Evaluate joint (t,c) prediction against ground truth.

    GT ONLY used in metric computation, NEVER in candidate generation.
    """
    T, N = joint.S.shape

    # Component ranking (marginal within query window)
    comp_score = joint.component_score  # [N]
    comp_rank = int(np.sum(comp_score > comp_score[gt_component_idx])) + 1
    comp_rank = min(comp_rank, N)

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
    """Aggregate per-root-cause metrics into summary statistics."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    total_queries = len(set(r.get("num_faults_in_query", 0) for r in results))
    # sum num_faults is not correct — just aggregate available fields
    agg = {
        "n": n,
        "component_top1": np.mean([r["component_top1"] for r in results]),
        "component_top3": np.mean([r["component_top3"] for r in results]),
        "mrr": np.mean([r["reciprocal_rank"] for r in results]),
        "avg_rank": np.mean([r["component_rank"] for r in results]),
        "time_hit_rate": np.mean([r["time_hit"] for r in results]),
        "time_mae": np.mean([r["time_error"] for r in results]),
        "time_mae_min": np.mean([r.get("time_error_min", r.get("time_error", 0) * 2) for r in results]),
        "time_hit_5min": np.mean([r.get("time_hit_5min", False) for r in results]),
        "time_hit_10min": np.mean([r.get("time_hit_10min", False) for r in results]),
        "time_hit_15min": np.mean([r.get("time_hit_15min", False) for r in results]),
        "joint_hit_rate": np.mean([r["joint_hit"] for r in results]),
        "joint_component_hit": np.mean([r["joint_component_hit"] for r in results]),
        "joint_time_hit": np.mean([r["joint_time_hit"] for r in results]),
    }
    return agg
