import sys
import os

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics"))

from gen3_dynamics import N  # noqa: E402
from pbc_singularity_cbf_controller import solve_control_qp, _mu_of_q  # noqa: E402


def test_controller_stays_finite_near_a_documented_infeasible_state():
    """
    Regression test for a real bug found in Discovery Experiment 61
    (scripts/rigid_body_dynamics/singularity_avoidance_validation.py): at a 500Hz
    control rate, the joint state below made the joint QP (Vdot<=0 passivity
    constraint together with the singularity CBF constraint) primal infeasible.
    OSQP was returning its infeasibility certificate (a huge, meaningless vector)
    as if it were a real solution, which blew up the closed-loop simulation to
    NaN within two steps. The fix drops the soft passivity constraint and
    re-solves with only the hard CBF constraint whenever the joint QP is
    infeasible.
    """
    q = jnp.array([0.024434754271692588, 0.21033097888352037, -0.013211504941075125,
                   -1.0242941965027483, -0.007324506055652218, 1.1038655977606235,
                   -0.9812817032340545])
    qd = jnp.array([0.02551702814644204, -0.09708874319385337, -0.00864499017131824,
                    0.9385065660201053, -0.010950684566497454, -0.7280601780288772,
                    -0.3852007482333173])
    p_des = jnp.array([-0.09341796453028546, -0.02485595, 1.072509868409771])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_control_qp(q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 100.0
    assert np.linalg.norm(tau) < 1000.0


def test_manipulability_positive_away_from_singularity():
    q_bent = jnp.array([0.0, 0.3, 0.0, -1.8, 0.0, 1.0, 0.0])
    q_straight = jnp.zeros(N)
    assert float(_mu_of_q(q_bent)) > 0.05
    assert float(_mu_of_q(q_straight)) < 1e-6
