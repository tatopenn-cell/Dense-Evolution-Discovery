# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/ewma_batch.py
=================================================
Batch/array EWMA detector, ported from ewma_river.py's streaming logic
into dense_armor.utility.cusum.cusum_detector's own conventions (causal
sliding window, radius*ref_mult span, robust or classical center/scale)
-- for validating EWMA's real speed advantage (found on synthetic data
against river's ADWIN) against Dense-Armor's own real agent telemetry,
before considering any promotion. Not yet part of dense_armor.
"""
from typing import Tuple

import numpy as np


def _window_causal(x: np.ndarray, i: int, span: int) -> np.ndarray:
    lo = max(0, i - span)
    return x[lo:i]


def _center_scale(w: np.ndarray, robust: bool) -> Tuple[float, float]:
    if robust:
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        return med, 1.4826 * mad
    return float(np.mean(w)), float(np.std(w))


def ewma_detector(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    lam: float = 0.2, L: float = 5.0, eps: float = 1e-9,
    robust: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """EWMA control chart (Roberts 1959), same causal-window convention as
    cusum_detector's 'adaptive' mode: center/scale recomputed every step
    from a sliding causal window (span=radius*ref_mult). z_t is the EWMA
    of the raw signal itself (not of a standardized residual), so it
    tracks the window's own center as the window moves -- consistent
    with cusum_detector's adaptive philosophy.

    Returns (flagged, z): flagged is True where |z_t - center_t| exceeds
    L * sigma_z_t (exact time-varying EWMA variance, resetting the
    effective sample count after every warmup gap, not carried over).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    flagged = np.zeros(n, dtype=bool)
    z_arr = np.zeros(n, dtype=np.float64)
    span = radius * ref_mult
    z = None
    t = 0

    for i in range(n):
        if not np.isfinite(x[i]):
            z = None
            t = 0
            continue
        w = _window_causal(x, i, span)
        if w.size < 4 or not np.all(np.isfinite(w)):
            z = None
            t = 0
            continue
        center, scale = _center_scale(w, robust)
        if not np.isfinite(scale) or scale < eps:
            z = None
            t = 0
            continue
        if z is None:
            z = center
            t = 0
        t += 1
        z = lam * x[i] + (1.0 - lam) * z
        z_arr[i] = z
        var_factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t))
        sigma_z = scale * np.sqrt(var_factor) if var_factor > 0 else 0.0
        if sigma_z < eps:
            continue
        flagged[i] = abs(z - center) > L * sigma_z

    return flagged, z_arr
