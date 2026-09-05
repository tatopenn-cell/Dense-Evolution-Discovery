from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from urdf_dynamics import RigidBodyModel
from general_pbc_cbf_controller import solve_control_qp

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


if __name__ == "__main__":
    model = RigidBodyModel("urdf/panda.urdf")
    print("real per-joint limits (q_min, q_max, qd_max):")
    for j, qmin, qmax, qdmax in zip(model.dof_joints, np.array(model.q_min), np.array(model.q_max), np.array(model.qd_max)):
        print(f"  {j['name']:20s} q in [{qmin:.4f}, {qmax:.4f}]  qd_max={qdmax:.4f}")

    # panda_joint4's real range is [-3.1416, 0.0] -- start near its q_max=0.0
    # boundary and command a target that would drive it further positive
    # (past the real limit) if nothing enforced it.
    q = jnp.array([0.0, 0.5, 0.0, -0.05, 0.0, 1.5, 0.0, 0.0, 0.0])
    qd = jnp.zeros(9)
    link = "panda_hand"

    from urdf_dynamics import RigidBodyModel as _RBM
    p_home = model.link_position(q, link)
    # target further along +x, +z to pull the arm toward straightening
    # joint4 (its own real q_max=0.0) well past that boundary
    p_target = np.array(p_home) + np.array([0.15, 0.0, 0.1])

    dt_ctrl = 0.01
    n_substeps = 5
    dt_phys = dt_ctrl / n_substeps
    n_steps = 300

    q4_log = []
    for k in range(n_steps):
        qdd, tau, mu, h = solve_control_qp(model, link, q, qd, p_target, np.zeros(3), np.zeros(3), eps=0.03)
        q, qd = rk4_n_substeps(model, q, qd, jnp.asarray(tau), dt_phys, n_substeps)
        q4_log.append(float(q[3]))

    q4_log = np.array(q4_log)
    q4_max_real = float(model.q_max[3])
    print(f"\njoint4 real q_max = {q4_max_real}")
    print(f"joint4 trajectory: max reached = {q4_log.max():.6f}, "
          f"violated = {'YES' if q4_log.max() > q4_max_real + 1e-6 else 'no'}")
