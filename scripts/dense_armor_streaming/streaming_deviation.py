# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/streaming_deviation.py
========================================================
Zero-latency streaming port of the CAUSAL DEVIATION half of Dense-Armor's
`arbiter.classify_segments` -- deliberately NOT the spike-vs-regime label,
which classify_segments computes by looking `radius` points AHEAD of a
deviant run's end (arbiter.py lines 178-186, "persiste" check) and
therefore cannot be zero-latency streaming. That distinction stays a
batch/offline triage question, unchanged. What real-time robotics safety
monitoring actually needs is narrower and simpler: "is this point
deviant right now" -- exactly `classify_segments`' own `deviante` array
(arbiter.py lines 113-132), same causal window (radius*ref_mult),
same robust median/MAD center-scale, same degenerate-baseline handling.

O(span) recomputation per step (median/MAD over a plain buffer), not a
two-heap O(log span) structure: for the window sizes this project
already uses everywhere (10-100 points), a plain buffer recomputation
takes microseconds -- optimizing further would add real complexity for
no measured benefit at real robot control-loop rates (30-100Hz).

Correctness bar: MUST reproduce `classify_segments`' own `deviante`
array bit-for-bit when fed the same series one point at a time -- not
"looks similar", exact equivalence, verified in
validate_streaming_deviation.py.
"""
from collections import deque
from typing import Tuple

import numpy as np


def _robust_center_scale(w) -> Tuple[float, float]:
    arr = np.asarray(w, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, 1.4826 * mad


class StreamingDeviationDetector:
    """Zero-latency causal deviation flag, one point at a time.

    Parameters mirror `classify_segments` exactly (radius, ref_mult,
    n_sigmas, eps) -- same causal-window span (radius*ref_mult), same
    robust center/scale, same degenerate-baseline rule (a truly flat
    reference makes ANY nonzero deviation count as deviant, no sigma
    threshold needed -- see arbiter.py's own docstring for why).

    Examples
    --------
    >>> det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> stream = list(rng.normal(0, 1, 20)) + [50.0]  # one huge outlier at the end
    >>> flags = [det.update(x) for x in stream]
    >>> flags[-1]
    True
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3, n_sigmas: float = 3.0, eps: float = 1e-9):
        self.radius = radius
        self.ref_mult = ref_mult
        self.n_sigmas = n_sigmas
        self.eps = eps
        self._span = radius * ref_mult
        self._buffer = deque(maxlen=self._span)
        self.last_deviation = 0.0  # informational, not part of the correctness contract

    def update(self, x: float) -> bool:
        """Feed one new point. Returns True if IT is deviant against the
        window of points strictly BEFORE it (never including itself) --
        matching `_window_causal`'s own semantics exactly."""
        if len(self._buffer) < 4:
            self._buffer.append(x)
            self.last_deviation = 0.0
            return False

        med, scala = _robust_center_scale(self._buffer)
        scarto = abs(x - med)
        if scala < self.eps:
            self.last_deviation = scarto
            deviante = scarto > self.eps
        else:
            self.last_deviation = scarto / scala
            deviante = self.last_deviation > self.n_sigmas

        self._buffer.append(x)
        return deviante
