# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/cusum_dual_channel.py
=========================================================
A two-channel CUSUM: one arm tuned for small sustained shifts (low k,
more sensitive but noisier), one for medium/large shifts (k=0.5, h=20 --
the single-channel default already calibrated in cusum_river.py), OR'd
together. A real, standard CUSUM technique (running multiple charts with
different k values in parallel to cover a wider range of shift sizes
without one k/h pair having to compromise), not invented for this
project -- analogous in SPIRIT (a second signal that can only help,
never silently degrade the base channel) to this repo's own
jsd_predictive_zne_density_matrix (docs/photonic_predictive_zne.md):
there, a rectified JSD-based nudge fires only when it helps and reduces
exactly to the base method otherwise; here, the small-shift channel
fires independently and never suppresses the large-shift channel's own
detections -- the combination can only add detections (and, honestly,
also add false alarms), never remove any the base channel already had.

NOT claimed to be free -- see benchmark_dual_channel.py for the real
combined false-alarm-rate cost, measured, not assumed.
"""
from __future__ import annotations

from river import base

from cusum_river import CUSUM


class CUSUMDualChannel(base.DriftDetector):
    """Two CUSUM channels in parallel, flags if EITHER fires.

    Parameters
    ----------
    small_kwargs, large_kwargs
        Constructor kwargs forwarded to each channel's own `CUSUM(...)`.
        Defaults calibrated empirically (benchmark_harness.py /
        cusum_river.py's own docstring): small=(k=0.25, h=25.0), tuned
        for a ~0.5-sigma shift; large=(k=0.5, h=20.0), the single-channel
        default, tuned for 1-2-sigma shifts.
    """

    def __init__(self, small_kwargs: dict | None = None, large_kwargs: dict | None = None):
        super().__init__()
        self.small_kwargs = small_kwargs or {"warmup_period": 30, "k": 0.25, "h": 25.0}
        self.large_kwargs = large_kwargs or {"warmup_period": 30, "k": 0.5, "h": 20.0}
        self._reset()

    def _reset(self):
        super()._reset()
        self._small = CUSUM(**self.small_kwargs)
        self._large = CUSUM(**self.large_kwargs)

    def update(self, x: int | float) -> None:
        self._small.update(x)
        self._large.update(x)
        self._drift_detected = bool(self._small.drift_detected or self._large.drift_detected)
