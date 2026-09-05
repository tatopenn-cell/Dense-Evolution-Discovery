from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from urdf_dynamics import RigidBodyModel
from six_dof_pbc_cbf_controller import solve_control_qp, rotation_error

jax.config.update("jax_enable_x64", True)


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


if __name__ == "__main__":
    model = RigidBodyModel("urdf/GEN3_URDF_V12.urdf")
    link = "end_effector_link"

    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.zeros(7)

    p0, r0 = model.link_pose(q, link)
    # real disclosed perturbation: target 10cm away in x,z, and rotated 30
    # degrees about the world z-axis from the current orientation.
    p_des = np.array(p0) + np.array([0.1, 0.0, 0.1])
    r_des = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), np.deg2rad(30.0)) @ np.array(r0)

    dt_ctrl = 0.01
    n_substeps = 5
    dt_phys = dt_ctrl / n_substeps
    n_steps = 500

    pos_err_log, rot_err_log = [], []
    for k in range(n_steps):
        qdd, tau, mu, h = solve_control_qp(
            model, link, q, qd, jnp.asarray(p_des), jnp.zeros(3), jnp.zeros(3),
            jnp.asarray(r_des), jnp.zeros(3), jnp.zeros(3), eps=0.03)
        q, qd = rk4_n_substeps(model, q, qd, jnp.asarray(tau), dt_phys, n_substeps)

        p, r = model.link_pose(q, link)
        pos_err_log.append(float(jnp.linalg.norm(p - jnp.asarray(p_des))))
        rot_err_log.append(float(jnp.linalg.norm(rotation_error(r, jnp.asarray(r_des)))))

    pos_err_log = np.array(pos_err_log)
    rot_err_log = np.array(rot_err_log)
    print(f"initial pos error = {pos_err_log[0]:.5f} m, final = {pos_err_log[-1]:.6f} m")
    print(f"initial rot error = {rot_err_log[0]:.5f}, final = {rot_err_log[-1]:.6f}")
    print(f"converged (pos<1mm and rot<0.001) = "
          f"{'YES' if pos_err_log[-1] < 1e-3 and rot_err_log[-1] < 1e-3 else 'no'}")
