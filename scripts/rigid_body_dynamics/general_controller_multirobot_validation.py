from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from dense_armor.utility.trajectory import quintic_trajectory
from urdf_dynamics import RigidBodyModel
from general_pbc_cbf_controller import solve_control_qp

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("model", "n_substeps"))
def _rk4_n_substeps(model, q, qd, tau, dt, n_substeps):
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


def run(model, link_name, q_home, p_home, p_target, eps, tag,
        dt_ctrl=0.01, n_substeps=5, t_traj=3.0, t_hold=2.0):
    n_steps_traj = int(round(t_traj / dt_ctrl)) + 1
    n_steps_hold = int(round(t_hold / dt_ctrl))
    _, p_ref, pd_ref, pdd_ref = quintic_trajectory(p_home, p_target, t_traj, n_samples=n_steps_traj)

    def desired_at_step(k):
        if k < n_steps_traj:
            return p_ref[k], pd_ref[k], pdd_ref[k]
        return p_ref[-1], np.zeros(3), np.zeros(3)

    q, qd = q_home, jnp.zeros(model.n)
    dt_phys = dt_ctrl / n_substeps

    mu_log, err_log = [], []
    n_total = n_steps_traj + n_steps_hold
    for k in range(n_total):
        p_des, pd_des, pdd_des = desired_at_step(k)
        _, tau, mu, h = solve_control_qp(model, link_name, q, qd, p_des, pd_des, pdd_des, eps=eps)

        q, qd = _rk4_n_substeps(model, q, qd, jnp.asarray(tau), dt_phys, n_substeps)

        err = float(jnp.linalg.norm(model.link_position(q, link_name) - jnp.asarray(p_des)))
        mu_log.append(mu)
        err_log.append(err)

    mu_log = np.array(mu_log)
    err_log = np.array(err_log)
    print(f"[{tag}] eps={eps:.3f}  min(mu)={mu_log.min():.5f}  "
          f"final_err={err_log[-1]:.5f} m  "
          f"violated={'YES' if mu_log.min() < eps - 1e-6 else 'no'}")
    return mu_log, err_log


if __name__ == "__main__":
    m6 = RigidBodyModel("urdf/GEN3-6DOF.urdf")
    q_home = jnp.array([0.51570841, 0.7814598, -1.17051634, 0.37482289, -0.25813235, 0.34260431])
    p_home = np.array([0.57535071, -0.33600968, 0.39733997])
    p_target = np.array([0.0, 0.00135062, 1.11510006])
    link = "bracelet_with_vision_link"

    run(m6, link, q_home, p_home, p_target, eps=0.0, tag="Gen3-6DOF no-CBF")
    run(m6, link, q_home, p_home, p_target, eps=0.03, tag="Gen3-6DOF CBF 100Hz")
