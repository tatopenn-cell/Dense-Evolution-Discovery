import sys
import os

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_body_dynamics"))

from gen3_dynamics import mass_matrix, total_energy, N  # noqa: E402
from energy_conservation_check import simulate  # noqa: E402


def test_mass_matrix_symmetric_positive_definite():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = jnp.array(rng.uniform(-2, 2, N))
        m = np.array(mass_matrix(q))
        assert np.max(np.abs(m - m.T)) < 1e-10
        eigvals = np.linalg.eigvalsh(m)
        assert eigvals.min() > 0


def test_energy_conserved_under_free_dynamics():
    q0 = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd0 = jnp.array([0.4, -0.2, 0.5, 0.1, -0.3, 0.2, 0.6])

    drifts = []
    for dt in (1e-2, 1e-3, 1e-4):
        n_steps = int(0.5 / dt)
        energies = simulate(q0, qd0, dt, n_steps)
        e0 = float(energies[0])
        drifts.append(float(jnp.max(jnp.abs(energies - e0))) / abs(e0))

    assert drifts[0] < 1e-1
    assert drifts[1] < drifts[0] / 100
    assert drifts[2] < drifts[1] / 100
