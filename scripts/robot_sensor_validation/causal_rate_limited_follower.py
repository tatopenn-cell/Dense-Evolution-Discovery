# -*- coding: utf-8 -*-
"""
scripts/robot_sensor_validation/causal_rate_limited_follower.py (PRIVATE)
==============================================================================
Different damping mechanism, after the causal neighbor-consensus rewrite
of healing_filter proved a structural dead end (see causal_healing_filter.py,
1.7% win rate vs a trivial moving median on real joint commands -- a real
detection-based approach cannot causally tell a real spike from a real
regime change at the instant it happens).

This sidesteps that classification problem entirely: instead of trying to
DETECT whether a deviation is "real", bound how fast the applied command
can physically change, regardless of cause -- the same principle real
robot controllers already use (torque/velocity/acceleration limits), and
the same family of problem solved rigorously by Berscheid & Kroger (2021),
"Jerk-limited Real-time Trajectory Generation with Arbitrary Target
States", Robotics: Science and Systems XVII, arXiv:2105.04830 (fetched
and verified directly before using: causal by construction -- uses only
the current kinematic state, no lookahead -- time-optimal jerk-limited
online trajectory generation, validated on 1e9 real trajectories, ~20us
real compute time per DoF).

THIS implementation is a deliberately SIMPLER special case, not a
reimplementation of Ruckig's full time-optimal jerk synthesis: a causal
velocity+acceleration-limited follower (a classic double-integrator rate
limiter) -- bounds the FIRST two derivatives of the applied command, not
the third (jerk itself is still a step function at each clamp transition).
If this shows real promise, a genuine jerk-limited version (bounding the
THIRD derivative too, closer to Ruckig itself) would be the natural next
step, not attempted here.
"""
import numpy as np


def causal_rate_limited_follower(x_raw: np.ndarray, max_vel: float, max_accel: float, dt: float = 1.0) -> np.ndarray:
    """Causally tracks x_raw (the raw incoming command stream) with an
    applied output whose velocity and acceleration are both bounded --
    uses only the previous applied position/velocity and the CURRENT raw
    target, never future values.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw incoming command stream (e.g. an LLM's per-step target position).
    max_vel, max_accel : float
        Real physical limits, in the same units as x_raw per real dt.
    dt : float
        Real time step between samples.

    Returns
    -------
    np.ndarray
        The applied (damped) command stream, same length as x_raw.
    """
    n = len(x_raw)
    out = np.empty(n)
    pos = float(x_raw[0])
    vel = 0.0
    out[0] = pos
    for i in range(1, n):
        target = float(x_raw[i])
        desired_vel = (target - pos) / dt
        max_dvel = max_accel * dt
        vel = float(np.clip(desired_vel, vel - max_dvel, vel + max_dvel))
        vel = float(np.clip(vel, -max_vel, max_vel))
        pos = pos + vel * dt
        out[i] = pos
    return out
