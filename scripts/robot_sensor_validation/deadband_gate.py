# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/deadband_gate.py
====================================================
Deadband/backlash gate for real robot joints -- inspired by the real
mechanical backlash phenomenon studied in Lima, Machado & Crisostomo,
"Experimental backlash study in mechanical manipulators", Robotica
29(2):211-219 (2011, DOI:10.1017/S0263574710000056; checked via WebFetch
before citing) -- a mechanical gap in a joint's gearing that briefly
resists motion whenever the commanded direction reverses, until static
friction is overcome. NOT a reimplementation of that paper's own method
(a pseudo-phase-plane + wavelet index) -- a much simpler, classical
signature check: real velocity right after a commanded direction
reversal, compared against the joint's own baseline real-velocity
level.

Checked directly on real SO-101 arm data (lerobot/svla_so101_pickplace,
episode 0) before building anything: 5 of 6 joints show real velocity
dropping to 16-68% of baseline immediately after a commanded reversal
-- joint 5 (the gripper, mechanically different) shows the opposite
pattern, sensibly (an open/close mechanism, not a geared rotary joint).

Two independent signals needed (commanded, actual) -- same pair as
stable_frame_filter.py's velocity_gated_stable_mask, reused here for a
different purpose: not gating for STABILITY (low velocity), but gating
OUT the deadband window so a downstream anomaly detector isn't fooled
by a real, benign, mechanical velocity dip into flagging a false
regime change.
"""
import numpy as np


def deadband_mask(
    commanded: np.ndarray, actual: np.ndarray,
    post_reversal_window: int = 3, drop_ratio: float = 0.5,
) -> np.ndarray:
    """
    commanded, actual : 1D arrays, same length -- e.g. leader/follower
        position for one joint.
    post_reversal_window : how many frames after a commanded direction
        reversal to consider "possibly in deadband".
    drop_ratio : a frame in that window is flagged as deadband if the
        joint's real |velocity| there is below `drop_ratio` times its
        own whole-series mean |velocity|.

    Returns deadband: bool array, True where the frame is plausibly
    inside a real mechanical deadband/backlash event (gate OUT of
    anomaly detection, don't treat as a genuine regime change).
    """
    commanded = np.asarray(commanded, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    n = commanded.size
    cmd_vel = np.diff(commanded, prepend=commanded[:1])
    act_vel = np.diff(actual, prepend=actual[:1])
    baseline = np.abs(act_vel).mean()
    if baseline < 1e-9:
        return np.zeros(n, dtype=bool)

    sign_change = np.zeros(n, dtype=bool)
    sign_change[1:] = np.sign(cmd_vel[1:]) != np.sign(cmd_vel[:-1])

    deadband = np.zeros(n, dtype=bool)
    for i in np.where(sign_change)[0]:
        hi = min(n, i + post_reversal_window)
        window_slow = np.abs(act_vel[i:hi]) < drop_ratio * baseline
        deadband[i:hi] = deadband[i:hi] | window_slow

    return deadband
