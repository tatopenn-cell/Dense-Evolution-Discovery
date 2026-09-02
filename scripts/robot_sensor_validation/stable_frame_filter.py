"""
scripts/robot_sensor_validation/stable_frame_filter.py
==========================================================
`velocity_gated_stable_mask` -- extracted from Experiment 43's inline
logic (a companion reference signal gates which frames of a SEPARATE
analyzed signal are trusted). Lives here in Discovery, not yet
promoted to dense-armor's utility/ package: this project's own
established rule (see how `one_sided_upper_filter` was promoted only
after proving itself on two independent real cases, via a real
ablation) is that a filter earns library-code status only after a
SECOND real case shows it still helps -- not from one experiment
alone. See VALIDATION below for both real cases this function has now
been checked against.

DESIGN CHOICE, made explicit rather than left implicit: this is
"Variant A" -- stability is judged from a REFERENCE signal, separate
from the signal actually being analyzed (e.g. LeRobot's teleoperation
leader's commanded position, separate from the leader-follower tracking
offset being analyzed). This matters: gating on the analyzed signal's
OWN volatility ("Variant B") would be circular for a signal that spikes
BECAUSE of the very confound "stable" is supposed to remove (Experiment
43 found and rejected exactly this trap on the raw tracking-offset
signal). Variant A avoids that by keying stability off an independent
reference the analyst already has a physical reason to trust.

already_rate PARAMETER -- added after a real, honest failure on a
second case: the first version always differentiated `reference`
(treating it as a position-like signal, matching LeRobot's `action`
array). Applied naively to a second real domain (IMU: gating
accelerometer-magnitude analysis by gyroscope-magnitude "how much is
the device rotating"), that was WRONG -- gyroscope magnitude is
ALREADY a rate (angular velocity), so differentiating it again doesn't
mean what it means for a position signal. `already_rate=True` skips
the differentiation and thresholds `reference` directly. This was not
silently patched in: the naive first attempt's wrong result is exactly
what surfaced the need for this parameter, see this project's own
research notes for that discovery process.

VALIDATION, two real, independent physical domains:
  1. already_rate=False (position-like reference): LeRobot SO-101 real
     teleoperation data, Experiment 43 -- gates on the leader's
     commanded joint position, byte-identical output verified after
     this refactor.
  2. already_rate=True (rate-like reference): real UCI HAR IMU data
     (Experiment 41's dataset, subject 17's real WALKING segment,
     50Hz) -- gates accelerometer-magnitude analysis by real gyroscope
     magnitude (25th-percentile threshold, declared before checking
     results). Real, modest, honest effect: accelerometer-magnitude
     std on gated-stable frames drops from 0.208 to 0.169 (-19%)
     versus the full raw signal -- a real, physically sensible but not
     dramatic noise reduction (less real device rotation correlates
     with less incidental translational-acceleration variance), not a
     dramatic win. See scripts/robot_sensor_validation/
     validate_stable_frame_filter_second_case.py for the frozen,
     reproducible numbers.
"""
from typing import Optional

import numpy as np


def velocity_gated_stable_mask(
    reference: np.ndarray, vel_threshold: float, axis: Optional[int] = None,
    already_rate: bool = False,
) -> np.ndarray:
    """Boolean mask, True where `reference` indicates a "slow" or
    "stable" state at that index, trustworthy for judging a companion
    analyzed signal at the same index.

    Parameters
    ----------
    reference : np.ndarray
        The signal stability is judged FROM -- not necessarily the
        same signal being analyzed for anomalies (see module docstring's
        "Variant A" note). 1D (shape (n,)) or 2D (shape (n, d), e.g. one
        column per joint/channel).
    vel_threshold : float
        A frame is "stable" only if EVERY channel's value (already_rate=
        True) or per-step absolute change (already_rate=False) is below
        this threshold.
    axis : int, optional
        Unused for 1D input. For 2D input, the channel axis (default:
        last axis, matching an (n_frames, n_channels) layout).
    already_rate : bool, default False
        False (the original, LeRobot-validated case): `reference` is a
        position/command-like signal -- velocity is computed as its
        own step-to-step absolute difference before thresholding.
        True (added for the IMU/gyroscope case): `reference` is ALREADY
        a rate/velocity-like quantity (e.g. angular velocity magnitude)
        -- thresholded directly, no further differentiation.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True at indices where `reference` indicates a stable/slow state.
    """
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim not in (1, 2):
        raise ValueError(f"reference must be 1D or 2D, got shape {reference.shape}")

    if already_rate:
        rate = np.abs(reference)
    elif reference.ndim == 1:
        rate = np.abs(np.diff(reference, prepend=reference[:1]))
    else:
        rate = np.abs(np.diff(reference, axis=0, prepend=reference[:1]))

    if rate.ndim == 1:
        return rate < vel_threshold
    return np.all(rate < vel_threshold, axis=1)
