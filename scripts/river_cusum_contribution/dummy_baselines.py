# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/dummy_baselines.py
======================================================
Trivial baseline detectors to stress-test benchmark_harness.py itself --
prompted directly by a real methodological critique found on river's own
issue #1914/PR #1963 (mateenali66, 2026-08-xx): "A real detector can
land below the data-ignoring one without the report saying so" -- an F1
number is meaningless without checking it beats a detector that ignores
the data entirely. If AlwaysFire or NeverFire score close to CUSUM/ADWIN
here, the harness itself (not any detector) is the thing to fix.
"""
from river import base


class AlwaysFire(base.DriftDetector):
    """Flags drift on every single update -- the maximal-recall,
    minimal-precision extreme."""

    def update(self, x) -> None:
        self._drift_detected = True


class NeverFire(base.DriftDetector):
    """Never flags drift -- the maximal-precision (vacuously), zero-recall
    extreme."""

    def update(self, x) -> None:
        self._drift_detected = False
