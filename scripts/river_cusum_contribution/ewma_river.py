# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/ewma_river.py
=================================================
EWMA (Exponentially Weighted Moving Average) control chart, streaming,
matching river's `base.DriftDetector` interface -- Roberts (1959),
listed in issue #1914's Family 1 alongside CUSUM as a highest-value,
lowest-risk gap. Same fixed-baseline-from-warmup convention as
cusum_river.py and shewhart_river.py.

z_t = lambda*x_t + (1-lambda)*z_{t-1}, z_0 = baseline_center. Flags when
|z_t - baseline_center| exceeds L * sigma_z_t, where sigma_z_t uses the
EXACT time-varying EWMA variance (not the steady-state approximation),
so the control limit doesn't start artificially tight right after
warmup: sigma_z_t = baseline_scale * sqrt(lambda/(2-lambda) *
(1-(1-lambda)^(2t))).

Unlike CUSUM, drift_detected is NOT force-reset after a flag -- z_t
keeps evolving and can legitimately stay flagged for several consecutive
points during a genuinely sustained shift (the standard EWMA control-
chart convention; a forced reset would defeat the point of a
persistent, decaying-memory statistic).
"""
from __future__ import annotations

import math

from river import base, stats


class EWMA(base.DriftDetector):
    """EWMA control chart (Roberts, 1959).

    Parameters
    ----------
    warmup_period
        Number of initial points used to estimate the fixed baseline
        mean/std.
    lam
        Smoothing factor (0, 1]. Smaller = longer memory, more
        sensitive to small sustained shifts, slower to react; larger =
        shorter memory, closer to Shewhart's single-point behavior.
        Classical textbook range 0.05-0.25; NOT assumed without checking
        (see `L`).
    L
        Control limit, in EWMA-sigma units (NOT baseline-sigma units --
        the EWMA statistic has smaller variance than a raw point by
        construction). Checked directly (benchmark_harness.py) before
        shipping a default, same discipline as cusum_river.py/
        shewhart_river.py: with lam=0.2, the textbook-adjacent L=3.0
        gives an 85.5% stream-level false-alarm rate on a 1000-sample
        stable N(0,1) series (200 trials, shift=1.0-sigma) -- far too
        high for a practical monitoring horizon. Swept L from 2.5 to
        8.0: F1 peaks at L=5.0 (this default) -- 8.0% false-alarm rate,
        16.5% missed-detection rate, F1=0.684 for a 1-sigma shift. Not
        claimed optimal (lam itself was not swept), just verified rather
        than assumed.
    robust
        If True, use median/1.4826*MAD instead of mean/std for the
        baseline.

    Examples
    --------
    >>> import random
    >>> rng = random.Random(12345)
    >>> chart = EWMA(warmup_period=30)
    >>> data_stream = [rng.gauss(0, 1) for _ in range(500)] + [rng.gauss(1.0, 1) for _ in range(500)]
    >>> for i, val in enumerate(data_stream):
    ...     chart.update(val)
    ...     if chart.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val:.3f}")
    ...         break
    Change detected at index 587, input value: 1.840

    References
    ----------
    S. W. Roberts. 1959. Control Chart Tests Based on Geometric Moving Averages.
    Technometrics 1, 3, 239-250.
    """

    def __init__(self, warmup_period: int = 30, lam: float = 0.2, L: float = 5.0, robust: bool = False):
        super().__init__()
        self.warmup_period = warmup_period
        self.lam = lam
        self.L = L
        self.robust = robust
        self._reset()

    def _reset(self):
        super()._reset()
        self._buffer: list[float] = []
        self._baseline_center: float | None = None
        self._baseline_scale: float | None = None
        self._z: float | None = None
        self._t = 0  # number of EWMA updates since warmup, for the exact time-varying variance

    def _estimate_baseline(self):
        if self.robust:
            sorted_buf = sorted(self._buffer)
            n = len(sorted_buf)
            med = sorted_buf[n // 2] if n % 2 else 0.5 * (sorted_buf[n // 2 - 1] + sorted_buf[n // 2])
            abs_dev = sorted([abs(v - med) for v in self._buffer])
            mad = abs_dev[n // 2] if n % 2 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
            self._baseline_center = med
            self._baseline_scale = 1.4826 * mad
        else:
            mean = stats.Mean()
            for v in self._buffer:
                mean.update(v)
            var = stats.Var()
            for v in self._buffer:
                var.update(v)
            self._baseline_center = mean.get()
            self._baseline_scale = math.sqrt(var.get()) if var.get() and var.get() > 0 else 0.0

    def update(self, x: int | float) -> None:
        if self._baseline_center is None:
            self._buffer.append(x)
            if len(self._buffer) >= self.warmup_period:
                self._estimate_baseline()
                self._z = self._baseline_center
            self._drift_detected = False
            return

        if self._baseline_scale is None or self._baseline_scale < 1e-9:
            self._drift_detected = False
            return

        self._t += 1
        self._z = self.lam * x + (1.0 - self.lam) * self._z
        var_factor = (self.lam / (2.0 - self.lam)) * (1.0 - (1.0 - self.lam) ** (2 * self._t))
        sigma_z = self._baseline_scale * math.sqrt(var_factor) if var_factor > 0 else 0.0
        if sigma_z < 1e-9:
            self._drift_detected = False
            return
        self._drift_detected = abs(self._z - self._baseline_center) > self.L * sigma_z
