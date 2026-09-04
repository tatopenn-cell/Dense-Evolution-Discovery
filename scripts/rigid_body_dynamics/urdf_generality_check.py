from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from urdf_dynamics import RigidBodyModel

jax.config.update("jax_enable_x64", True)


def make_rk4(model):
    @jax.jit
    def rk4_step(q, qd, dt):
        tau = jnp.zeros(model.n)

        def deriv(q, qd):
            return qd, model.forward_dynamics(q, qd, tau)

        k1q, k1v = deriv(q, qd)
        k2q, k2v = deriv(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
        k3q, k3v = deriv(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
        k4q, k4v = deriv(q + dt * k3q, qd + dt * k3v)
        q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd_next = qd + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        return q_next, qd_next

    @partial(jax.jit, static_argnames=("n_steps",))
    def simulate(q0, qd0, dt, n_steps):
        def scan_fn(carry, _):
            q, qd = carry
            q_next, qd_next = rk4_step(q, qd, dt)
            return (q_next, qd_next), model.total_energy(q_next, qd_next)

        _, energies = jax.lax.scan(scan_fn, (q0, qd0), None, length=n_steps)
        e0 = model.total_energy(q0, qd0)
        return jnp.concatenate([e0[None], energies])

    return simulate


def check_energy_conservation(model, q0, qd0, tag):
    simulate = make_rk4(model)
    print(f"--- {tag}: n_dof={model.n} ---")
    for dt in (1e-2, 1e-3, 1e-4):
        n_steps = int(round(1.0 / dt))
        energies = simulate(q0, qd0, dt, n_steps)
        e0 = float(energies[0])
        drift = float(jnp.max(jnp.abs(energies - e0)))
        print(f"  dt={dt:.0e}  steps={n_steps:6d}  E0={e0:10.4f} J  "
              f"max|dE|={drift:.3e} J  rel={drift/abs(e0):.3e}")


def check_mass_matrix_spd(model, tag, n_trials=20, seed=0):
    rng = np.random.default_rng(seed)
    worst_sym = 0.0
    worst_eig = np.inf
    for _ in range(n_trials):
        q = jnp.array(rng.uniform(-2, 2, model.n))
        m = np.array(model.mass_matrix(q))
        worst_sym = max(worst_sym, np.max(np.abs(m - m.T)))
        worst_eig = min(worst_eig, np.linalg.eigvalsh(m).min())
    print(f"--- {tag}: SPD check over {n_trials} random configs -- "
          f"max asymmetry={worst_sym:.2e}, min eigenvalue={worst_eig:.5f} ---")


if __name__ == "__main__":
    m7 = RigidBodyModel("urdf/GEN3_URDF_V12.urdf")
    q0_7 = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd0_7 = jnp.array([0.4, -0.2, 0.5, 0.1, -0.3, 0.2, 0.6])
    check_mass_matrix_spd(m7, "Gen3 7-DOF (general parser)")
    check_energy_conservation(m7, q0_7, qd0_7, "Gen3 7-DOF (general parser)")

    m6 = RigidBodyModel("urdf/GEN3-6DOF.urdf")
    q0_6 = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8])
    qd0_6 = jnp.array([0.4, -0.2, 0.5, 0.1, -0.3, 0.2])
    check_mass_matrix_spd(m6, "Gen3 6-DOF (different real robot)")
    check_energy_conservation(m6, q0_6, qd0_6, "Gen3 6-DOF (different real robot)")
