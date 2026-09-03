"""
Direct unit tests for causal_rate_limited_follower -- the function's own
correctness properties (causality, respects real limits, converges when
unconstrained), separate from tests/test_rate_limiter_real_joint_commands.py's
aggregate real-data benchmark results.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "robot_sensor_validation"))
from causal_rate_limited_follower import causal_rate_limited_follower


def test_causal_output_unaffected_by_truncating_future_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50).cumsum()
    i = 30
    out_full = causal_rate_limited_follower(x, max_vel=1.0, max_accel=0.5)
    out_truncated = causal_rate_limited_follower(x[:i + 1], max_vel=1.0, max_accel=0.5)
    assert np.isclose(out_full[i], out_truncated[i])


def test_velocity_never_exceeds_max_vel():
    x = np.concatenate([np.zeros(5), np.full(20, 1000.0)])  # a huge, instant real jump
    out = causal_rate_limited_follower(x, max_vel=2.0, max_accel=1.0)
    vel = np.diff(out)
    assert np.all(np.abs(vel) <= 2.0 + 1e-9)


def test_acceleration_never_exceeds_max_accel():
    x = np.concatenate([np.zeros(5), np.full(20, 1000.0)])
    out = causal_rate_limited_follower(x, max_vel=100.0, max_accel=0.3)
    vel = np.diff(out)
    accel = np.diff(vel)
    assert np.all(np.abs(accel) <= 0.3 + 1e-9)


def test_converges_to_a_constant_target_when_limits_are_generous():
    x = np.concatenate([np.zeros(5), np.full(50, 10.0)])
    out = causal_rate_limited_follower(x, max_vel=100.0, max_accel=100.0)
    assert np.isclose(out[-1], 10.0, atol=1e-6)


def test_isolated_spike_capped_not_fully_passed_through():
    x = np.zeros(30)
    x[15] = 1000.0  # a single-sample real spike
    out = causal_rate_limited_follower(x, max_vel=1.0, max_accel=0.5)
    assert out[15] < 100.0  # far below the raw spike -- the whole point of rate-limiting


def test_smooth_real_signal_tracked_closely_when_within_limits():
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.sin(t)  # slow enough to stay well within generous limits
    out = causal_rate_limited_follower(x, max_vel=1.0, max_accel=1.0)
    assert np.sqrt(np.mean((out - x) ** 2)) < 0.05
