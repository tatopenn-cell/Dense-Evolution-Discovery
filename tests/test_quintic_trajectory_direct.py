"""
Direct unit tests for scripts/trajectory_planning/quintic_trajectory.py
-- the closed-form point-to-point trajectory generator, before any
real-data validation. See test_quintic_trajectory.py for the frozen
real SO-101/ALOHA validation, and docs/quintic_trajectory_planner.md
for the full write-up.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "trajectory_planning"))
from quintic_trajectory import quintic_trajectory  # noqa: E402


def test_boundary_conditions_satisfied_exactly():
    q0, qf, T = [0.0], [10.0], 2.0
    v0, a0, vf, af = [1.0], [0.5], [-2.0], [0.3]
    t, q, v, a = quintic_trajectory(q0, qf, T, v0, a0, vf, af, n_samples=200)
    assert abs(q[0, 0] - q0[0]) < 1e-9
    assert abs(v[0, 0] - v0[0]) < 1e-9
    assert abs(a[0, 0] - a0[0]) < 1e-9
    assert abs(q[-1, 0] - qf[0]) < 1e-9
    assert abs(v[-1, 0] - vf[0]) < 1e-9
    assert abs(a[-1, 0] - af[0]) < 1e-6


def test_matches_independent_numeric_differentiation():
    t, q, v, a = quintic_trajectory([0.0], [10.0], 2.0, [1.0], [0.5], [-2.0], [0.3], n_samples=200)
    dt = t[1] - t[0]
    v_numeric = np.gradient(q[:, 0], dt)
    a_numeric = np.gradient(v_numeric, dt)
    assert np.max(np.abs(v[5:-5, 0] - v_numeric[5:-5])) < 0.01
    assert np.max(np.abs(a[5:-5, 0] - a_numeric[5:-5])) < 0.1


def test_default_boundary_conditions_are_zero():
    t, q, v, a = quintic_trajectory([0.0], [1.0], 1.0)
    assert abs(v[0, 0]) < 1e-9
    assert abs(a[0, 0]) < 1e-9
    assert abs(v[-1, 0]) < 1e-9
    assert abs(a[-1, 0]) < 1e-9


def test_works_for_any_number_of_dof():
    n_dof = 14
    q0 = np.zeros(n_dof)
    qf = np.arange(n_dof, dtype=float)
    t, q, v, a = quintic_trajectory(q0, qf, 3.0, n_samples=50)
    assert q.shape == (50, n_dof)
    assert np.allclose(q[0], q0)
    assert np.allclose(q[-1], qf)


def test_stationary_start_end_gives_a_real_trajectory_shape_not_a_straight_line():
    # A minimum-jerk profile between two rest points is NOT linear in time --
    # it must accelerate away from and decelerate into both endpoints.
    t, q, v, a = quintic_trajectory([0.0], [1.0], 1.0, n_samples=101)
    mid = q[50, 0]
    assert abs(mid - 0.5) < 1e-6          # symmetric profile, midpoint at the midpoint
    assert v[25, 0] < v[50, 0]             # still accelerating at t=T/4
    assert v[50, 0] > v[75, 0]             # already decelerating by t=3T/4
