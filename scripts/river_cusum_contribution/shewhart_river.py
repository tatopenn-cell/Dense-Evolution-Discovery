# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/shewhart_river.py
=====================================================
Shewhart control chart, streaming, matching river's `base.DriftDetector`
interface -- the "optional, completes the control-chart set" item from
issue #1914's Family 1. The simplest classical chart: no accumulation,
no memory -- each point is judged independently against a FIXED
baseline (same warmup convention as cusum_river.py). Flags when
|x[i] - baseline_center| > L * baseline_scale.

Because it has no memory, drift_detected is recomputed fresh every
update() -- unlike CUSUM/EWMA there is nothing to "reset" on a flag.
"""
from __future__ import annotations

import math

from river import base, stats


class Shewhart(base.DriftDetector):
    """Shewhart control chart (three-sigma rule and its generalizations).

    Parameters
    ----------
    warmup_period
        Number of initial points used to estimate the fixed baseline
        mean/std.
    L
        Control limit, in baseline-sigma units. NOT the textbook L=3.0
        by default -- checked directly (benchmark_harness.py) before
        shipping a value, same discipline as cusum_river.py's h: L=3.0
        gives an 88.5% stream-level false-alarm rate on a 1000-sample
        stable N(0,1) series (200 trials) -- a single-point three-sigma
        rule is memoryless, so it re-rolls the false-alarm dice every
        step, and over 1000 steps the odds catch up fast. L=4.5 (this
        default) gives a 10.5% stream-level false-alarm rate on the same
        horizon.

        IMPORTANT, STRUCTURAL LIMIT (not a tuning gap): being memoryless,
        Shewhart is only useful for LARGE, abrupt shifts -- at L=4.5,
        F1=0.86-0.88 for a 3-5 sigma shift (200 trials, same horizon),
        but F1 collapses to 0.24-0.33 for a 1-sigma shift (missed-rate
        67.5% at L=4.5, since a single point rarely exceeds 4.5 sigma
        even during a genuinely sustained small shift). This is expected
        -- Shewhart has no mechanism to accumulate small, persistent
        deviations (that's what CUSUM/EWMA are for); it exists in this
        module for completeness (per issue #1914's own "Shewhart control
        chart (optional, completes the control-chart set)"), not as a
        general-purpose recommendation.
    robust
        If True, use median/1.4826*MAD instead of mean/std for the
        baseline.

    Examples
    --------
    >>> import random
    >>> rng = random.Random(12345)
    >>> chart = Shewhart(warmup_period=30)
    >>> data_stream = [rng.gauss(0, 1) for _ in range(500)] + [rng.gauss(3.0, 1) for _ in range(500)]
    >>> for i, val in enumerate(data_stream):
    ...     chart.update(val)
    ...     if chart.drift_detected:
    ...         print(f"Change detected at index {i}, input value: {val:.3f}")
    ...         break
    Change detected at index 500, input value: 5.414

    References
    ----------
    W. A. Shewhart. 1931. Economic Control of Quality of Manufactured Product.
    """

    def __init__(self, warmup_period: int = 30, L: float = 4.5, robust: bool = False):
        super().__init__()
        self.warmup_period = warmup_period
        self.L = L
        self.robust = robust
        self._reset()

    def _reset(self):
        super()._reset()
        self._buffer: list[float] = []
        self._baseline_center: float | None = None
        self._baseline_scale: float | None = None

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
            self._drift_detected = False
            return

        if self._baseline_scale is None or self._baseline_scale < 1e-9:
            self._drift_detected = False
            return

        z = (x - self._baseline_center) / self._baseline_scale
        self._drift_detected = abs(z) > self.L
