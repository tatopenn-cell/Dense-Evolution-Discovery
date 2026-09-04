from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from dense_armor.utility.trajectory import quintic_trajectory
from gen3_dynamics import forward_dynamics, end_effector_position, N
from pbc_singularity_cbf_controller import solve_control_qp

Q_HOME = jnp.array([0.0, 0.3, 0.0, -1.8, 0.0, 1.0, 0.0])

P_HOME = np.array([-0.26951164, -0.02485595, 0.85596968])
P_EXTENDED = np.array([0.0, -0.02485595, 1.18738477])

T_TRAJ = 3.0
T_HOLD = 2.0


@partial(jax.jit, static_argnames=("n_substeps",))
def rk4_n_substeps(q, qd, tau, dt, n_substeps):
    def deriv(q, qd):
        return qd, forward_dynamics(q, qd, tau)

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


def run(eps, tag, dt_ctrl=0.01, n_substeps=5):
    n_steps_traj = int(round(T_TRAJ / dt_ctrl)) + 1
    n_steps_hold = int(round(T_HOLD / dt_ctrl))
    _, p_ref, pd_ref, pdd_ref = quintic_trajectory(P_HOME, P_EXTENDED, T_TRAJ, n_samples=n_steps_traj)

    def desired_at_step(k):
        if k < n_steps_traj:
            return p_ref[k], pd_ref[k], pdd_ref[k]
        return p_ref[-1], np.zeros(3), np.zeros(3)

    q = Q_HOME
    qd = jnp.zeros(7)
    dt_phys = dt_ctrl / n_substeps

    mu_log, err_log = [], []
    n_total = n_steps_traj + n_steps_hold
    for k in range(n_total):
        p_des, pd_des, pdd_des = desired_at_step(k)
        _, tau, mu, h = solve_control_qp(q, qd, p_des, pd_des, pdd_des, eps=eps)

        q, qd = rk4_n_substeps(q, qd, jnp.asarray(tau), dt_phys, n_substeps)

        err = float(jnp.linalg.norm(end_effector_position(q) - jnp.asarray(p_des)))
        mu_log.append(mu)
        err_log.append(err)

    mu_log = np.array(mu_log)
    err_log = np.array(err_log)
    print(f"[{tag}] eps={eps:.3f} dt_ctrl={dt_ctrl:.4f}  min(mu)={mu_log.min():.5f}  "
          f"final_err={err_log[-1]:.5f} m  max_err_during_hold="
          f"{err_log[n_steps_traj:].max():.5f} m  "
          f"violated={'YES' if mu_log.min() < eps - 1e-6 else 'no'}")
    return mu_log, err_log


if __name__ == "__main__":
    run(eps=0.03, tag="with-CBF eps=0.03, 100Hz ctrl", dt_ctrl=0.01, n_substeps=5)
    run(eps=0.03, tag="with-CBF eps=0.03, 500Hz ctrl", dt_ctrl=0.002, n_substeps=1)
    run(eps=0.0, tag="no-CBF (never binds), 100Hz ctrl", dt_ctrl=0.01, n_substeps=5)
