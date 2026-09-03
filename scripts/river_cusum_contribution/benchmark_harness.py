# -*- coding: utf-8 -*-
"""
scripts/river_cusum_contribution/benchmark_harness.py
========================================================
A small evaluation harness for streaming drift detectors -- exactly what
river issue #1914 ("Drift module roadmap") calls the structural gap
("we can't regression-test new detectors or substantiate competitiveness
claims"): detection delay, false-alarm rate on stable data, missed-drift
rate, and F1, against a KNOWN change point. Built here, in Discovery,
to validate the CUSUM prototype (cusum_river.py) BEFORE proposing
anything upstream -- per this project's own promotion discipline.

Metrics (matching the issue's own wording):
  - false_alarm_rate: fraction of STABLE-only streams (no real change)
    that raise at least one flag anywhere.
  - detection_delay: for CHANGED streams, (flag_index - true_change_index)
    for the FIRST flag at or after the true change point. None if missed.
  - missed_rate: fraction of CHANGED streams with no flag at or after the
    true change point.
  - f1: computed at the STREAM level (a stream counts as a true positive
    if the change was detected with delay < tolerance, a false negative
    if missed, and false positives come from the stable-stream set) --
    a simple, honest, replicable definition, not claimed to match any
    paper's exact protocol.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """false_alarms/false_alarm_flags: two DIFFERENT counts, kept
    separate on purpose after a real bug was found and fixed here
    (2026-09-03, prompted directly by a methodological critique on
    river's own PR #1963: "a real detector can land below the
    data-ignoring one without the report saying so"). An AlwaysFire
    dummy detector (flags every single point) scored F1=0.667 under the
    FIRST version of this harness -- beating a real CUSUM detector
    (F1=0.512) at a 0.5-sigma shift -- because `false_alarms` only
    counted STREAMS with >=1 flag, not the actual NUMBER of spurious
    flags. A detector firing 1000 times in one stable stream was
    penalized identically to one firing once. `false_alarm_flags`
    (every single spurious flag, sample-level) fixes this; `false_alarms`
    (streams with >=1 flag) is kept for the original, coarser
    per-stream false-alarm RATE metric, not used in f1() anymore."""
    name: str
    n_stable: int
    n_changed: int
    false_alarms: int = 0  # streams (not samples) with >=1 spurious flag
    false_alarm_flags: int = 0  # EVERY spurious flag, across all stable streams AND pre-change points in changed streams
    detections: list = field(default_factory=list)  # delays, one per detected changed-stream
    missed: int = 0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.n_stable if self.n_stable else float("nan")

    @property
    def missed_rate(self) -> float:
        return self.missed / self.n_changed if self.n_changed else float("nan")

    @property
    def mean_delay(self):
        return statistics.mean(self.detections) if self.detections else None

    @property
    def median_delay(self):
        return statistics.median(self.detections) if self.detections else None

    def f1(self, tolerance: int = 200) -> float:
        tp = sum(1 for d in self.detections if d <= tolerance)
        fn = self.missed + sum(1 for d in self.detections if d > tolerance)
        fp = self.false_alarm_flags
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def make_stable_stream(rng: random.Random, n: int, mean: float = 0.0, std: float = 1.0):
    return [rng.gauss(mean, std) for _ in range(n)]


def make_changed_stream(rng: random.Random, n: int, change_at: int, mean_before: float,
                         mean_after: float, std: float = 1.0):
    before = [rng.gauss(mean_before, std) for _ in range(change_at)]
    after = [rng.gauss(mean_after, std) for _ in range(n - change_at)]
    return before + after


def run_detector_factory(detector_factory, stream, warn_only_first_flag_after=None):
    """Feeds a fresh detector instance through `stream`; returns the list of
    indices where drift_detected fired. `detector_factory` is a zero-arg
    callable returning a fresh, unfitted detector (river's own convention:
    a detector accumulates state, so each stream needs its own instance)."""
    det = detector_factory()
    flags = []
    for i, x in enumerate(stream):
        det.update(x)
        if det.drift_detected:
            flags.append(i)
    return flags


def evaluate(name, detector_factory, rng_seed: int, n_trials: int, n_points: int,
             change_at: int, mean_before: float, mean_after: float, std: float = 1.0,
             ) -> BenchmarkResult:
    rng = random.Random(rng_seed)
    result = BenchmarkResult(name=name, n_stable=n_trials, n_changed=n_trials)

    for _ in range(n_trials):
        stream = make_stable_stream(rng, n_points, mean_before, std)
        flags = run_detector_factory(detector_factory, stream)
        if flags:
            result.false_alarms += 1
        result.false_alarm_flags += len(flags)  # EVERY flag on a stream with no real change is spurious

    for _ in range(n_trials):
        stream = make_changed_stream(rng, n_points, change_at, mean_before, mean_after, std)
        flags = run_detector_factory(detector_factory, stream)
        pre_change_flags = [f for f in flags if f < change_at]
        post_change_flags = [f for f in flags if f >= change_at]
        result.false_alarm_flags += len(pre_change_flags)  # any flag before the true change is also spurious
        if post_change_flags:
            result.detections.append(post_change_flags[0] - change_at)
        else:
            result.missed += 1

    return result
