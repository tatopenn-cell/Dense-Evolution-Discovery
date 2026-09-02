"""
scripts/robot_sensor_validation/stable_frame_filter.py
==========================================================
`velocity_gated_stable_mask` -- extracted from Experiment 43's inline
logic (a companion reference signal's velocity gates which frames of a
SEPARATE analyzed signal are trusted). Lives here in Discovery, not yet
promoted to dense-armor's utility/ package: validated on exactly ONE
real dataset (LeRobot's leader/follower teleoperation data) so far, and
this project's own established rule (see how `one_sided_upper_filter`
was promoted only after proving itself on two independent real cases,
via a real ablation) is that a filter earns library-code status only
after a SECOND real case shows it still helps -- not from one
experiment alone.

DESIGN CHOICE, made explicit rather than left implicit: this is
"Variant A" -- velocity is computed from a REFERENCE signal (e.g. the
teleoperation leader's commanded position), separate from the signal
actually being analyzed (e.g. the leader-follower tracking offset).
This matters: gating on the offset signal's OWN volatility ("Variant
B") would be a different, not-yet-validated design -- for a tracking-
offset signal specifically, the offset itself spikes DURING fast
motion (the effect Experiment 43 found and rejected as a false-positive
trap), so gating on ITS OWN volatility would be circular (excluding
exactly the frames where the confound is worst is what "stable" is
supposed to achieve, but you'd be using the confounded signal to decide
that). Variant A avoids this by keying stability off an independent
reference the analyst already has a physical reason to trust (the
commanded/leader signal, not the derived offset).
"""
from typing import Optional

import numpy as np


def velocity_gated_stable_mask(
    reference: np.ndarray, vel_threshold: float, axis: Optional[int] = None,
) -> np.ndarray:
    """Boolean mask, True where `reference` was changing slowly enough
    (per-step absolute difference below `vel_threshold`) to trust a
    companion analyzed signal at that same index.

    Parameters
    ----------
    reference : np.ndarray
        The signal velocity is computed FROM -- not necessarily the
        same signal being analyzed for anomalies (see module docstring's
        "Variant A" note). 1D (shape (n,)) or 2D (shape (n, d), e.g. one
        column per joint/channel).
    vel_threshold : float
        A frame is "stable" only if EVERY channel's absolute step
        change is below this threshold (for 2D input) -- matches
        Experiment 43's use (a teleoperated arm's `action` array, all
        joints must have settled).
    axis : int, optional
        Unused for 1D input. For 2D input, the channel axis (default:
        last axis, matching an (n_frames, n_channels) layout).

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True at indices where `reference` was locally stable.
    """
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim == 1:
        vel = np.abs(np.diff(reference, prepend=reference[:1]))
        return vel < vel_threshold
    if reference.ndim == 2:
        vel = np.abs(np.diff(reference, axis=0, prepend=reference[:1]))
        return np.all(vel < vel_threshold, axis=1)
    raise ValueError(f"reference must be 1D or 2D, got shape {reference.shape}")
