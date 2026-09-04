"""
Direct unit tests for scripts/trajectory_planning/kinematic_controller.py
-- the closed-form feedforward+proportional kinematic tracking
controller, before any real-data validation. See
test_kinematic_controller.py for the frozen real SO-101/ALOHA
validation, and docs/kinematic_tracking_controller.md for the full
write-up.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "trajectory_planning"))
from kinematic_controller import kinematic_tracking_controller  # noqa: E402


def test_exact_exponential_convergence_constant_reference():
    q = np.array([5.0])
    q_ref = np.array([0.0])
    qd_ref = np.array([0.0])
    kp = 2.0
    dt = 0.001
    errors = []
    for _ in range(5000):
        e = q_ref - q
        errors.append(e[0])
        u = kinematic_tracking_controller(q, q_ref, qd_ref, kp)
        q = q + u * dt
    errors = np.array(errors)
    t = np.arange(len(errors)) * dt
    theory = errors[0] * np.exp(-kp * t)
    assert np.max(np.abs(errors - theory)) < 0.05


def test_exact_convergence_time_varying_reference():
    # The controller must converge to zero tracking error even when the
    # reference itself is moving, not just for a fixed setpoint.
    t = np.linspace(0, 3, 3000)
    dt = t[1] - t[0]
    q_ref_traj = np.sin(t)
    qd_ref_traj = np.cos(t)
    q = np.array([5.0])
    kp = 8.0
    for i in range(len(t)):
        q_ref = np.array([q_ref_traj[i]])
        qd_ref = np.array([qd_ref_traj[i]])
        u = kinematic_tracking_controller(q, q_ref, qd_ref, kp)
        q = q + u * dt
    final_error = abs(q_ref_traj[-1] - q[0])
    assert final_error < 0.01


def test_zero_gain_gives_pure_feedforward():
    q = np.array([0.0, 0.0])
    q_ref = np.array([1.0, 2.0])
    qd_ref = np.array([3.0, 4.0])
    u = kinematic_tracking_controller(q, q_ref, qd_ref, kp=0.0)
    assert np.allclose(u, qd_ref)


def test_works_for_any_number_of_dof():
    n = 14
    q = np.zeros(n)
    q_ref = np.arange(n, dtype=float)
    qd_ref = np.zeros(n)
    u = kinematic_tracking_controller(q, q_ref, qd_ref, kp=1.0)
    assert u.shape == (n,)
    assert np.allclose(u, q_ref)


def test_mismatched_shapes_raise():
    import pytest
    with pytest.raises(AssertionError):
        kinematic_tracking_controller([0.0, 0.0], [1.0], [0.0], kp=1.0)
