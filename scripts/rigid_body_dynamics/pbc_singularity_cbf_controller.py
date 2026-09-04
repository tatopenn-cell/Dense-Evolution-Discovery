from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import osqp
from scipy import sparse

from gen3_dynamics import (
    N, mass_matrix, bias_forces, gravity_forces,
    end_effector_position, end_effector_jacobian, end_effector_jacobian_dot_times_qd,
)

jax.config.update("jax_enable_x64", True)


def manipulability(j):
    return jnp.sqrt(jnp.linalg.det(j @ j.T))


def _mu_of_q(q):
    return manipulability(end_effector_jacobian(q))


def manipulability_jacobian(q):
    return jax.grad(_mu_of_q)(q)


def manipulability_jacobian_dot_times_qd(q, qd):
    jfn = lambda qq: manipulability_jacobian(qq) @ qd
    return jax.jvp(jfn, (q,), (qd,))[1]


def lambda_task(q):
    m = mass_matrix(q)
    j = end_effector_jacobian(q)
    return jnp.linalg.inv(j @ jnp.linalg.solve(m, j.T))


def lambda_task_dot_times_qd(q, qd):
    return jax.jvp(lambda_task, (q,), (qd,))[1]


@partial(jax.jit, static_argnames=())
def _qp_ingredients(q, qd, p_des, pd_des, pdd_des,
                     kp_task, kd_task, kd_null, eps, ka0, ka1):
    """
    All pure-JAX work for one control tick, jit-compiled once and reused on every
    call. Returns plain arrays; the only non-JAX step left to the caller is the
    OSQP solve itself (OSQP is a compiled external solver, not traceable by JAX).
    """
    m = mass_matrix(q)
    minv = jnp.linalg.inv(m)
    j = end_effector_jacobian(q)
    jdot_qd = end_effector_jacobian_dot_times_qd(q, qd)

    lam = lambda_task(q)
    lam_dot = lambda_task_dot_times_qd(q, qd)
    jbar = minv @ j.T @ lam

    x = end_effector_position(q)
    xd = j @ qd
    x_tilde = x - p_des
    xd_tilde = xd - pd_des

    xdd_cmd = pdd_des - kd_task * xd_tilde - kp_task * x_tilde
    qdd_task = jbar @ (xdd_cmd - jdot_qd)
    null_proj = jnp.eye(N) - jbar @ j
    qdd_nom = qdd_task + null_proj @ (-kd_null * qd)

    def vdot_fn(qdd):
        xdd_tilde = j @ qdd + jdot_qd - pdd_des
        return (xd_tilde @ lam @ xdd_tilde
                + 0.5 * xd_tilde @ lam_dot @ xd_tilde
                + kp_task * x_tilde @ xd_tilde)

    zero = jnp.zeros(N)
    a_vdot = jax.grad(vdot_fn)(zero)
    c_vdot = vdot_fn(zero)

    mu = _mu_of_q(q)
    j_mu = manipulability_jacobian(q)
    jdot_mu_qd = manipulability_jacobian_dot_times_qd(q, qd)
    h = mu - eps
    hd = j_mu @ qd

    a1 = a_vdot
    u1 = -c_vdot
    a2 = -j_mu
    u2 = ka0 * h + ka1 * hd + jdot_mu_qd

    bias = bias_forces(q, qd)
    grav = gravity_forces(q)

    return m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h


def solve_control_qp(q, qd, p_des, pd_des, pdd_des,
                      kp_task=50.0, kd_task=20.0, kd_null=5.0,
                      eps=0.035, ka=(100.0, 20.0), w_reg=1e-6):
    """
    Task-space (3-DoF end-effector position) passivity-based controller with an
    exponential CBF constraint enforcing a minimum manipulability index, following
    Kurtz, Wensing & Lin (2021, arXiv:2109.13349): a QP over qdd whose cost pulls
    toward the nominal operational-space PD command, subject to Vdot<=0
    (passivity of the tracking-error storage function) and hdd>=-Ka[h;hd]
    (forward invariance of h=mu-eps>0). Both constraints are affine in qdd; the
    affine coefficients are extracted by autodiff (grad/jvp) instead of hand-built
    Christoffel-symbol algebra.
    """
    m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h = _qp_ingredients(
        jnp.asarray(q), jnp.asarray(qd), jnp.asarray(p_des), jnp.asarray(pd_des),
        jnp.asarray(pdd_des), kp_task, kd_task, kd_null, eps, ka[0], ka[1])

    p_mat = sparse.csc_matrix(np.eye(N) * (1.0 + w_reg))
    q_vec = -np.asarray(qdd_nom)

    a_full = sparse.csc_matrix(np.vstack([np.asarray(a1).reshape(1, N),
                                           np.asarray(a2).reshape(1, N)]))
    l_full = np.array([-1e20, -1e20])
    u_full = np.array([float(u1), float(u2)])

    prob = osqp.OSQP()
    prob.setup(p_mat, q_vec, a_full, l_full, u_full, verbose=False, polish=True)
    res = prob.solve()

    if res.info.status_val != osqp.constant("OSQP_SOLVED"):
        # The Vdot<=0 passivity constraint is a soft, secondary objective; the
        # singularity CBF is the safety-critical one. If the two are jointly
        # infeasible (numerically degenerate Vdot row near zero tracking error
        # combined with a tight CBF margin -- see Discovery Experiment
        # rigid_body_dynamics/singularity_avoidance_validation.py), drop the
        # passivity constraint and keep only the CBF one, which alone is always
        # feasible for this cost.
        a_cbf = sparse.csc_matrix(np.asarray(a2).reshape(1, N))
        l_cbf = np.array([-1e20])
        u_cbf = np.array([float(u2)])
        prob = osqp.OSQP()
        prob.setup(p_mat, q_vec, a_cbf, l_cbf, u_cbf, verbose=False, polish=True)
        res = prob.solve()

    qdd = np.asarray(res.x)
    tau = np.asarray(m) @ qdd + np.asarray(bias) + np.asarray(grav)
    return qdd, tau, float(mu), float(h)
