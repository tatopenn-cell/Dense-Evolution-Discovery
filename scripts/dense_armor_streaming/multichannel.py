# -*- coding: utf-8 -*-
"""
scripts/dense_armor_streaming/multichannel.py
=================================================
Native multi-channel wrappers for classify_segments and
StreamingDeviationDetector -- ergonomics, not a new algorithm. Every
real robotics experiment in this repo (LeRobot's 6 joints, the UCI HAR
IMU's multiple axes) has needed a hand-written per-channel loop around
a function built for one 1D signal. Real robots always have more than
one channel; this closes that gap by applying the SAME, already-
validated per-channel logic across all channels, returning one array
instead of requiring the caller to loop and stack manually.

Not a new detector: `classify_segments_multichannel`'s output for
channel j is required to be byte-identical to calling
`classify_segments(X[:, j], **kw)` directly -- verified, not assumed,
in validate_multichannel.py.
"""
from typing import List

import numpy as np

from streaming_deviation import StreamingDeviationDetector


def classify_segments_multichannel(X: np.ndarray, classify_segments_fn, **kwargs):
    """X: (n_samples, n_channels). Applies `classify_segments_fn`
    (pass `dense_armor.utility.arbiter.classify_segments` -- not
    imported directly here so this module has no hard Dense-Armor
    dependency) independently to each column.

    Returns (etichette, deviazione, incertezza), each (n_samples,
    n_channels) -- same per-column values `classify_segments` would
    give if called on that column alone, just stacked instead of
    requiring the caller to loop.
    """
    X = np.asarray(X, dtype=np.float64)
    n, c = X.shape
    etichette = np.empty((n, c), dtype=object)
    deviazione = np.zeros((n, c), dtype=np.float64)
    incertezza = np.zeros((n, c), dtype=np.float64)
    for j in range(c):
        e, d, u = classify_segments_fn(X[:, j], **kwargs)
        etichette[:, j] = e
        deviazione[:, j] = d
        incertezza[:, j] = u
    return etichette, deviazione, incertezza


class MultiChannelStreamingDeviationDetector:
    """N independent StreamingDeviationDetector instances, one per
    channel -- each channel's own deviation flag is computed exactly
    as if `StreamingDeviationDetector` were run on that channel alone
    (channels never influence each other's reference window; a robot's
    joints/axes are typically NOT expected to share one baseline)."""

    def __init__(self, n_channels: int, radius: int = 10, ref_mult: int = 3,
                 n_sigmas: float = 3.0, eps: float = 1e-9):
        self.n_channels = n_channels
        self._detectors: List[StreamingDeviationDetector] = [
            StreamingDeviationDetector(radius=radius, ref_mult=ref_mult, n_sigmas=n_sigmas, eps=eps)
            for _ in range(n_channels)
        ]

    def update(self, x_vec) -> np.ndarray:
        """x_vec: length n_channels. Returns bool array (n_channels,)."""
        x_vec = np.asarray(x_vec, dtype=np.float64).ravel()
        if x_vec.size != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x_vec.size}")
        return np.array([det.update(float(v)) for det, v in zip(self._detectors, x_vec)], dtype=bool)
