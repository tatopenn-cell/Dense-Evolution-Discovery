import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics"))

from urdf_dynamics import RigidBodyModel  # noqa: E402
from six_dof_pbc_cbf_controller import solve_control_qp, rotation_error  # noqa: E402

URDF_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics", "urdf")


def test_zero_error_gives_pure_gravity_compensation():
    """
    At the exact desired pose (position and orientation) with zero velocity,
    the solved torque must equal gravity compensation exactly (qdd=0, no
    tracking terms active) -- a real correctness check of the whole 6-DoF
    pipeline (spatial Jacobian, rotation error, task-space Lambda), not just
    "doesn't crash".
    """
    model = RigidBodyModel(os.path.join(URDF_DIR, "GEN3_URDF_V12.urdf"))
    link = "end_effector_link"
    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.zeros(7)
    p, r = model.link_pose(q, link)

    qdd, tau, mu, h = solve_control_qp(model, link, q, qd, p, jnp.zeros(3), jnp.zeros(3),
                                        r, jnp.zeros(3), jnp.zeros(3), eps=0.03)
    grav = model.gravity_forces(q)

    assert np.max(np.abs(np.array(qdd))) < 1e-9
    assert np.max(np.abs(np.array(tau) - np.array(grav))) < 1e-9


def test_rotation_error_zero_iff_equal():
    model = RigidBodyModel(os.path.join(URDF_DIR, "GEN3_URDF_V12.urdf"))
    q = jnp.array([0.1, 0.2, 0.3, -0.5, 0.2, 0.6, -0.1])
    _, r = model.link_pose(q, "end_effector_link")
    assert np.max(np.abs(np.array(rotation_error(r, r)))) < 1e-12


def test_position_and_orientation_converge():
    """
    Real closed-loop check: a real disclosed 10cm/10cm position offset and a
    real disclosed 30-degree orientation offset (about world z) both
    converge under RK4-integrated real dynamics -- not just a single QP call.
    """
    from functools import partial
    import jax

    @partial(jax.jit, static_argnames=("model", "n_substeps"))
    def rk4_n_substeps(model, q, qd, tau, dt, n_substeps):
        def deriv(q, qd):
            return qd, model.forward_dynamics(q, qd, tau)

        def one_step(carry, _):
            q, qd = carry
            k1q, k1v = deriv(q, qd)
            k2q, k2v = deriv(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
            k3q, k3v = deriv(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
            k4q, k4v = deriv(q + dt * k3q, qd + dt * k3v)
            q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            qd_next = qd + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            return (q_next, qd_next), None

        (q_out, qd_out), _ = jax.lax.scan(one_step, (q, qd), None, length=n_substeps)
        return q_out, qd_out

    def axis_angle_to_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

    model = RigidBodyModel(os.path.join(URDF_DIR, "GEN3_URDF_V12.urdf"))
    link = "end_effector_link"
    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.zeros(7)
    p0, r0 = model.link_pose(q, link)
    p_des = np.array(p0) + np.array([0.1, 0.0, 0.1])
    r_des = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), np.deg2rad(30.0)) @ np.array(r0)

    dt_ctrl = 0.01
    n_substeps = 5
    dt_phys = dt_ctrl / n_substeps
    for k in range(1000):
        qdd, tau, mu, h = solve_control_qp(
            model, link, q, qd, jnp.asarray(p_des), jnp.zeros(3), jnp.zeros(3),
            jnp.asarray(r_des), jnp.zeros(3), jnp.zeros(3), eps=0.03)
        q, qd = rk4_n_substeps(model, q, qd, jnp.asarray(tau), dt_phys, n_substeps)

    p, r = model.link_pose(q, link)
    pos_err = float(jnp.linalg.norm(p - jnp.asarray(p_des)))
    rot_err = float(jnp.linalg.norm(rotation_error(r, jnp.asarray(r_des))))
    assert pos_err < 1e-6
    assert rot_err < 1e-4
