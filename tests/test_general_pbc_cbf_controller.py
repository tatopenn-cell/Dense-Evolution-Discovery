import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics"))

from urdf_dynamics import RigidBodyModel  # noqa: E402
from general_pbc_cbf_controller import solve_control_qp as solve_general, _qp_ingredients  # noqa: E402
from pbc_singularity_cbf_controller import solve_control_qp as solve_gen3_hardcoded  # noqa: E402

URDF_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics", "urdf")


def test_general_controller_matches_gen3_hardcoded_controller():
    """
    Cross-check at the exact state that triggered Experiment 61's real
    OSQP-infeasibility bug: the general (any-robot) controller must reproduce
    the Gen3-specific one to machine precision, including the infeasibility
    fallback path.
    """
    model = RigidBodyModel(os.path.join(URDF_DIR, "GEN3_URDF_V12.urdf"))
    q = jnp.array([0.024434754271692588, 0.21033097888352037, -0.013211504941075125,
                   -1.0242941965027483, -0.007324506055652218, 1.1038655977606235,
                   -0.9812817032340545])
    qd = jnp.array([0.02551702814644204, -0.09708874319385337, -0.00864499017131824,
                    0.9385065660201053, -0.010950684566497454, -0.7280601780288772,
                    -0.3852007482333173])
    p_des = jnp.array([-0.09341796453028546, -0.02485595, 1.072509868409771])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd_g, tau_g, mu_g, h_g = solve_general(model, "end_effector_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)
    qdd_h, tau_h, mu_h, h_h = solve_gen3_hardcoded(q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.max(np.abs(qdd_g - qdd_h)) < 1e-9
    assert np.max(np.abs(tau_g - tau_h)) < 1e-9
    assert abs(mu_g - mu_h) < 1e-9
    assert abs(h_g - h_h) < 1e-9


def test_general_controller_stays_finite_on_a_different_robot():
    """Same controller, a genuinely different real robot (Gen3 6-DoF): must
    stay finite and bounded near a singular target, not just on the Gen3-7DoF
    case the module was originally built for."""
    model = RigidBodyModel(os.path.join(URDF_DIR, "GEN3-6DOF.urdf"))
    q = jnp.array([0.51570841, 0.7814598, -1.17051634, 0.37482289, -0.25813235, 0.34260431])
    qd = jnp.zeros(6)
    p_des = jnp.array([0.0, 0.00135062, 1.11510006])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_general(model, "bracelet_with_vision_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    # A large finite qdd is expected here -- the target is at the singular
    # pose itself, so the controller pushes hard against the CBF boundary.
    # The real failure mode this guards against (Experiment 61's OSQP
    # infeasibility bug) produces qdd with norm in the billions, not hundreds.
    assert np.linalg.norm(qdd) < 1e4


def test_general_controller_stays_finite_on_a_third_robot_different_manufacturer():
    """Third robot, a different manufacturer (Franka Panda, 7 revolute + 2
    prismatic finger joints): completes the same 3-robot validation bar
    RigidBodyModel was held to before its own promotion."""
    model = RigidBodyModel(os.path.join(URDF_DIR, "panda.urdf"))
    q = jnp.array([-0.19360032, 0.89941805, -0.27271074, -1.15251618,
                   -2.92540148, 2.89498581, 2.73584013, 0.0, 0.0])
    qd = jnp.zeros(9)
    p_des = jnp.array([-0.2295431, -0.47162945, 1.00885136])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_general(model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_joint_limit_cbf_brakes_a_joint_at_its_real_limit():
    """
    Real per-joint limits from the URDF's <limit> tags are enforced as an
    additional CBF (Kurtz et al.'s own "joint" constraint type): a joint
    sitting right at its real bound with velocity pushing further past it
    must be braked toward zero/negative acceleration, not left to the
    unconstrained nominal command (which here demands strong further
    acceleration in the wrong direction).
    """
    model = RigidBodyModel(os.path.join(URDF_DIR, "panda.urdf"))
    link = "panda_hand"
    # panda_joint4's real range is [-3.1416, 0.0] -- start right at its
    # upper bound with positive velocity driving further past it.
    q = jnp.array([0.0, 0.5, 0.0, -0.001, 0.0, 1.5, 0.0, 0.0, 0.0])
    qd = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_des = model.link_position(q, link) + jnp.array([0.0, 0.0, 0.3])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_general(model, link, q, qd, p_des, pd_des, pdd_des, eps=0.03)

    _, _, _, qdd_nom, _, _, _, _, _, _, qdd_lb, qdd_ub = _qp_ingredients(
        model, link, q, qd, jnp.asarray(p_des), pd_des, pdd_des, 50.0, 20.0, 5.0, 0.03, 100.0, 20.0)

    q4_max = float(model.q_max[3])
    assert q[3] < q4_max
    # The unconstrained nominal command demands far stronger deceleration
    # (qdd_nom[3] = -205.8 here) than the real CBF box allows
    # ([-7.175, -4.999]); the solved qdd3 must land inside that real box
    # (continuous-time guarantee at this instant -- not a full-dt
    # forward-Euler prediction, which would show the same small
    # zero-order-hold discretization gap documented elsewhere in this
    # project for the singularity CBF), clamped rather than left at the
    # nominal command's much larger magnitude.
    assert float(qdd_nom[3]) < float(qdd_lb[3])
    assert float(qdd_lb[3]) - 1e-6 <= float(qdd[3]) <= float(qdd_ub[3]) + 1e-6
