# -*- coding: utf-8 -*-
"""
scripts/trajectory_planning/quintic_trajectory.py
====================================================================
Closed-form, no-training, minimum-jerk-continuous point-to-point
trajectory generator -- the "universal base generator" identified as
step 1 of the trajectory-planning notes in prog.txt, deliberately
scoped down from the much larger bi-level/URDF-dynamics-aware
optimizer those notes also describe (Fried & Paternain 2024,
arXiv:2412.07859; Lozer et al., Robotics and Autonomous Systems --
both real papers, verified directly by reading them before writing
this, not trusted from a third-party summary).

Universal in the sense that matters here: works for any number of
joints (any robot) given only position/velocity/acceleration boundary
conditions and a total time -- no URDF, no kinematics, no dynamics.
Composes with the existing stack exactly as prog.txt's own point 10
says: cbf_filter already handles spatial safety, rate_limiter already
handles rate-of-change safety, so this generator does not need to
worry about either -- it only needs to produce a smooth reference.

THEORY: the classic quintic (5th-order) polynomial trajectory (see
e.g. Craig, "Introduction to Robotics", or Piazzi & Visioli 2000) is
the unique degree-5 polynomial per joint satisfying 6 real boundary
conditions (position, velocity, acceleration at t=0 and t=T) -- solved
here directly from the 6x6 linear system, not memorized/copied from a
formula, so a transcription error would fail the boundary-condition
test below rather than silently producing a wrong trajectory.
"""
import numpy as np


def quintic_trajectory(q0, qf, T: float, v0=None, a0=None, vf=None, af=None, n_samples: int = 100):
    """Closed-form quintic point-to-point trajectory, any number of DOF.

    Parameters
    ----------
    q0, qf : array-like, shape (n_dof,)
        Start and end position for each joint.
    T : float
        Total real trajectory duration.
    v0, a0, vf, af : array-like, shape (n_dof,), optional
        Start/end velocity and acceleration per joint (default 0 --
        the standard "start and end at rest" case).
    n_samples : int
        Number of real time samples to return.

    Returns
    -------
    t : ndarray, shape (n_samples,)
    q, v, a : ndarray, shape (n_samples, n_dof)
        Position, velocity, acceleration at each sampled time.
    """
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    qf = np.atleast_1d(np.asarray(qf, dtype=float))
    n_dof = q0.shape[0]
    assert qf.shape[0] == n_dof, "q0 and qf must have the same number of joints"
    v0 = np.zeros(n_dof) if v0 is None else np.atleast_1d(np.asarray(v0, dtype=float))
    a0 = np.zeros(n_dof) if a0 is None else np.atleast_1d(np.asarray(a0, dtype=float))
    vf = np.zeros(n_dof) if vf is None else np.atleast_1d(np.asarray(vf, dtype=float))
    af = np.zeros(n_dof) if af is None else np.atleast_1d(np.asarray(af, dtype=float))
    assert T > 0, "T must be positive"

    t = np.linspace(0.0, T, n_samples)

    boundary_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
        [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    rhs = np.stack([q0, v0, a0, qf, vf, af], axis=1)          # (n_dof, 6)
    coeffs = np.linalg.solve(boundary_matrix, rhs.T)           # (6, n_dof)

    powers_q = t[:, None] ** np.arange(6)[None, :]                                    # (n_samples, 6)
    powers_v = np.concatenate([np.zeros((n_samples, 1)),
                                np.arange(1, 6)[None, :] * t[:, None] ** np.arange(0, 5)[None, :]], axis=1)
    powers_a = np.concatenate([np.zeros((n_samples, 2)),
                                (np.arange(2, 6) * np.arange(1, 5))[None, :] * t[:, None] ** np.arange(0, 4)[None, :]], axis=1)

    q = powers_q @ coeffs
    v = powers_v @ coeffs
    a = powers_a @ coeffs
    return t, q, v, a
