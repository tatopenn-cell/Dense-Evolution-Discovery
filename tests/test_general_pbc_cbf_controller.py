import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics"))

from urdf_dynamics import RigidBodyModel  # noqa: E402
from general_pbc_cbf_controller import solve_control_qp as solve_general  # noqa: E402
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
