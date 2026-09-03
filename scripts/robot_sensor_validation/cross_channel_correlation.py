# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/cross_channel_correlation.py
================================================================
Cross-channel correlation detector -- a classical, no-retraining
alternative to arXiv:2505.05811's deep-learning insight (a real robot
fault shows up as a breakdown in the normal correlation between sensor
channels that usually co-vary, validated there via Mahalanobis SVDD +
cross-attention on real audio-IMU data, F1=92.3%). This module tests
the SAME insight with a much lighter mechanism: rolling pairwise
Pearson correlation across channels, causal reference window, same
radius/ref_mult convention as arbiter.py/cusum.py.

Multi-channel signals (a robot arm's N joints, an IMU's 3+ axes)
normally co-vary during real coordinated motion. A single channel
going rogue (stuck sensor, jammed joint, decoupled axis) should show up
as a drop in its average pairwise correlation with the rest, even if
its own univariate statistics (mean, variance) look unremarkable --
exactly the class of fault a single-channel detector (classify_segments,
cusum_detector) structurally cannot see.
"""
import numpy as np


def _window_causal(X: np.ndarray, i: int, span: int) -> np.ndarray:
    lo = max(0, i - span)
    return X[lo:i]


def cross_channel_correlation_detector(
    X: np.ndarray, radius: int = 10, ref_mult: int = 3,
    drop_threshold: float = 0.4, eps: float = 1e-9,
    vel_threshold: float = None, min_active_frac: float = 0.5,
) -> np.ndarray:
    """
    X: (n_samples, n_channels).
    For each point i (once span=radius*ref_mult samples are available),
    compute each channel's mean pairwise Pearson correlation with every
    OTHER channel over the causal reference window. A channel is
    flagged at point i if its own correlation drops more than
    `drop_threshold` below the window's mean cross-channel correlation
    (i.e., it decorrelated from the group, not just noisier overall).

    vel_threshold : float, optional
        If set, gates the reference window to only ACTIVE frames (mean
        |velocity| across channels > vel_threshold) before computing
        correlation -- the opposite use of the velocity-gating idea
        from stable_frame_filter.py (there: keep only STABLE/slow
        frames; here: keep only ACTIVE/moving frames, since the
        correlation between near-constant, mostly-noise channels is
        unstable and not a meaningful "normal co-variation" baseline).
        1.0 (this default when set) matches Experiment 43's own reused
        VEL_THRESHOLD, not a new number invented for this check.
    min_active_frac : float
        Minimum fraction of the reference window that must be active
        (after gating) for a correlation estimate to be trusted; below
        this, the point is skipped (not flagged) rather than judged on
        too few active samples.

    Returns per_channel_flagged: bool array (n_samples, n_channels).
    """
    X = np.asarray(X, dtype=np.float64)
    n, c = X.shape
    span = radius * ref_mult
    flagged = np.zeros((n, c), dtype=bool)

    for i in range(n):
        w = _window_causal(X, i, span)
        if w.shape[0] < max(8, c + 2):
            continue
        if not np.all(np.isfinite(w)):
            continue

        if vel_threshold is not None:
            vel = np.abs(np.diff(w, axis=0, prepend=w[:1]))
            active_mask = vel.mean(axis=1) > vel_threshold
            if active_mask.mean() < min_active_frac or active_mask.sum() < max(8, c + 2):
                continue
            w = w[active_mask]

        stds = w.std(axis=0)
        if np.any(stds < eps):
            continue
        corr = np.corrcoef(w, rowvar=False)
        if not np.all(np.isfinite(corr)):
            continue
        # per-channel mean correlation with every OTHER channel
        off_diag_sum = corr.sum(axis=1) - 1.0
        per_channel_corr = off_diag_sum / (c - 1)
        group_mean_corr = per_channel_corr.mean()
        flagged[i] = (group_mean_corr - per_channel_corr) > drop_threshold

    return flagged
