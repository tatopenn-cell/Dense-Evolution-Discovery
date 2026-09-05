from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import osqp
from scipy import sparse

jax.config.update("jax_enable_x64", True)


def manipulability(j):
    return jnp.sqrt(jnp.linalg.det(j @ j.T))


def rotation_error(r, r_des):
    """
    SO(3) attitude error following Lee, Leok & McClamroch (2010), "Geometric
    Tracking Control of a Quadrotor UAV on SE(3)": e_R = 0.5*vee(R_des^T R -
    R^T R_des), smooth everywhere (no log-map singularity at zero error, no
    arccos derivative blowup), vanishes iff r == r_des. Deliberately used
    instead of Kurtz, Wensing & Lin's own RPY-based error (RollPitchYaw of
    the error rotation): RPY has a real gimbal-lock singularity this
    formulation avoids, while keeping the same task-space PBC+CBF structure.
    """
    e_mat = r_des.T @ r - r.T @ r_des
    return 0.5 * jnp.array([e_mat[2, 1], e_mat[0, 2], e_mat[1, 0]])


def _mu_fn(model, link_name):
    return lambda q: manipulability(model.link_spatial_jacobian(q, link_name))


def manipulability_jacobian(model, link_name, q):
    return jax.grad(_mu_fn(model, link_name))(q)


def manipulability_jacobian_dot_times_qd(model, link_name, q, qd):
    jfn = lambda qq: manipulability_jacobian(model, link_name, qq) @ qd
    return jax.jvp(jfn, (q,), (qd,))[1]


def spatial_jacobian_dot_times_qd(model, link_name, q, qd):
    jfn = lambda qq: model.link_spatial_jacobian(qq, link_name) @ qd
    return jax.jvp(jfn, (q,), (qd,))[1]


def lambda_task(model, link_name, q):
    m = model.mass_matrix(q)
    j = model.link_spatial_jacobian(q, link_name)
    return jnp.linalg.inv(j @ jnp.linalg.solve(m, j.T))


def lambda_task_dot_times_qd(model, link_name, q, qd):
    fn = lambda qq: lambda_task(model, link_name, qq)
    return jax.jvp(fn, (q,), (qd,))[1]


@partial(jax.jit, static_argnames=("model", "link_name"))
def _qp_ingredients(model, link_name, q, qd, p_des, pd_des, pdd_des, r_des, w_des, wd_des,
                     kp_task, kd_task, kd_null, eps, ka0, ka1):
    n = model.n
    m = model.mass_matrix(q)
    minv = jnp.linalg.inv(m)
    j = model.link_spatial_jacobian(q, link_name)
    jdot_qd = spatial_jacobian_dot_times_qd(model, link_name, q, qd)

    lam = lambda_task(model, link_name, q)
    lam_dot = lambda_task_dot_times_qd(model, link_name, q, qd)
    jbar = minv @ j.T @ lam

    p, r = model.link_pose(q, link_name)
    e_r = rotation_error(r, r_des)
    x_tilde = jnp.concatenate([e_r, p - p_des])

    xd = j @ qd
    xd_des = jnp.concatenate([w_des, pd_des])
    xd_tilde = xd - xd_des

    xdd_des = jnp.concatenate([wd_des, pdd_des])
    xdd_cmd = xdd_des - kd_task * xd_tilde - kp_task * x_tilde
    qdd_task = jbar @ (xdd_cmd - jdot_qd)
    null_proj = jnp.eye(n) - jbar @ j
    qdd_nom = qdd_task + null_proj @ (-kd_null * qd)

    def vdot_fn(qdd):
        xdd_tilde = j @ qdd + jdot_qd - xdd_des
        return (xd_tilde @ lam @ xdd_tilde
                + 0.5 * xd_tilde @ lam_dot @ xd_tilde
                + kp_task * x_tilde @ xd_tilde)

    zero = jnp.zeros(n)
    a_vdot = jax.grad(vdot_fn)(zero)
    c_vdot = vdot_fn(zero)

    mu = manipulability(j)
    j_mu = manipulability_jacobian(model, link_name, q)
    jdot_mu_qd = manipulability_jacobian_dot_times_qd(model, link_name, q, qd)
    h = mu - eps
    hd = j_mu @ qd

    a1 = a_vdot
    u1 = -c_vdot
    a2 = -j_mu
    u2 = ka0 * h + ka1 * hd + jdot_mu_qd

    qd_min = -model.qd_max
    ah_qd_min = qd - qd_min
    ah_qd_max = model.qd_max - qd
    ah_q_min = qd + (q - model.q_min)
    ah_q_max = (model.q_max - q) - qd
    qdd_lb = jnp.maximum(-ah_qd_min, -ah_q_min)
    qdd_ub = jnp.minimum(ah_qd_max, ah_q_max)

    bias = model.bias_forces(q, qd)
    grav = model.gravity_forces(q)

    return m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h, qdd_lb, qdd_ub


def solve_control_qp(model, link_name, q, qd, p_des, pd_des, pdd_des, r_des, w_des, wd_des,
                      kp_task=50.0, kd_task=20.0, kd_null=5.0,
                      eps=0.035, ka=(100.0, 20.0), w_reg=1e-6):
    """
    Full 6-DoF (position + orientation) generalization of
    general_pbc_cbf_controller.py's 3-DoF (position-only) controller --
    same passivity+singularity-CBF QP (Kurtz, Wensing & Lin 2021), same
    real per-joint limit CBF, but tracking the named link's full pose via
    the spatial (6xN) Jacobian instead of only its position. Attitude error
    uses Lee et al. (2010)'s SO(3) formula (see rotation_error), not
    Kurtz et al.'s own RPY-based error, to avoid gimbal lock.
    """
    n = model.n
    m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h, qdd_lb, qdd_ub = _qp_ingredients(
        model, link_name, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(p_des),
        jnp.asarray(pd_des), jnp.asarray(pdd_des), jnp.asarray(r_des),
        jnp.asarray(w_des), jnp.asarray(wd_des), kp_task, kd_task, kd_null,
        eps, ka[0], ka[1])

    p_mat = sparse.csc_matrix(np.eye(n) * (1.0 + w_reg))
    q_vec = -np.asarray(qdd_nom)

    qdd_lb_raw = np.asarray(qdd_lb)
    qdd_ub_raw = np.asarray(qdd_ub)
    has_real_limits = not (np.all(np.isneginf(qdd_lb_raw)) and np.all(np.isposinf(qdd_ub_raw)))

    rows_a1a2 = [sparse.csc_matrix(np.asarray(a1).reshape(1, n)),
                 sparse.csc_matrix(np.asarray(a2).reshape(1, n))]
    l_a1a2 = [-1e20, -1e20]
    u_a1a2 = [float(u1), float(u2)]

    if has_real_limits:
        qdd_lb_np = np.clip(qdd_lb_raw, -1e20, 1e20)
        qdd_ub_np = np.clip(qdd_ub_raw, -1e20, 1e20)
        identity_n = sparse.csc_matrix(np.eye(n))
        a_full = sparse.vstack(rows_a1a2 + [identity_n]).tocsc()
        l_full = np.concatenate([l_a1a2, qdd_lb_np])
        u_full = np.concatenate([u_a1a2, qdd_ub_np])
    else:
        a_full = sparse.vstack(rows_a1a2).tocsc()
        l_full = np.array(l_a1a2)
        u_full = np.array(u_a1a2)

    prob = osqp.OSQP()
    prob.setup(p_mat, q_vec, a_full, l_full, u_full, verbose=False, polish=True)
    res = prob.solve()

    if res.info.status_val != osqp.constant("OSQP_SOLVED"):
        if has_real_limits:
            a_cbf = sparse.vstack([sparse.csc_matrix(np.asarray(a2).reshape(1, n)),
                                    sparse.csc_matrix(np.eye(n))]).tocsc()
            l_cbf = np.concatenate([[-1e20], qdd_lb_np])
            u_cbf = np.concatenate([[float(u2)], qdd_ub_np])
        else:
            a_cbf = sparse.csc_matrix(np.asarray(a2).reshape(1, n))
            l_cbf = np.array([-1e20])
            u_cbf = np.array([float(u2)])
        prob = osqp.OSQP()
        prob.setup(p_mat, q_vec, a_cbf, l_cbf, u_cbf, verbose=False, polish=True)
        res = prob.solve()

        if has_real_limits and res.info.status_val != osqp.constant("OSQP_SOLVED"):
            # A real second-level infeasibility: the 6-DoF manipulability
            # measure (spatial Jacobian, includes orientation) can be much
            # smaller than the 3-DoF one at the same configuration (a real
            # wrist-singularity-adjacent case, not an artifact), so the CBF's
            # required recovery rate can exceed what the velocity-limit box
            # allows -- CBF+box jointly infeasible. The CBF is the harder
            # safety constraint (prevents an actual kinematic singularity,
            # not just a momentary rate-limit overshoot), so drop the box
            # and keep only the CBF -- feasible in R^n as long as a2 != 0.
            a_cbf2 = sparse.csc_matrix(np.asarray(a2).reshape(1, n))
            l_cbf2 = np.array([-1e20])
            u_cbf2 = np.array([float(u2)])
            prob = osqp.OSQP()
            prob.setup(p_mat, q_vec, a_cbf2, l_cbf2, u_cbf2, verbose=False, polish=True)
            res = prob.solve()

    qdd = np.asarray(res.x)
    tau = np.asarray(m) @ qdd + np.asarray(bias) + np.asarray(grav)
    return qdd, tau, float(mu), float(h)
