import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gen3_dynamics import forward_dynamics, total_energy, N


def rk4_step(q, qd, dt):
    tau = jnp.zeros(N)

    def deriv(q, qd):
        return qd, forward_dynamics(q, qd, tau)

    k1q, k1v = deriv(q, qd)
    k2q, k2v = deriv(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
    k3q, k3v = deriv(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
    k4q, k4v = deriv(q + dt * k3q, qd + dt * k3v)

    q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
    qd_next = qd + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return q_next, qd_next


def simulate(q0, qd0, dt, n_steps):
    def scan_fn(carry, _):
        q, qd = carry
        q_next, qd_next = rk4_step(q, qd, dt)
        return (q_next, qd_next), total_energy(q_next, qd_next)

    _, energies = jax.lax.scan(scan_fn, (q0, qd0), None, length=n_steps)
    e0 = total_energy(q0, qd0)
    return jnp.concatenate([e0[None], energies])


if __name__ == "__main__":
    q0 = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd0 = jnp.array([0.4, -0.2, 0.5, 0.1, -0.3, 0.2, 0.6])

    for dt in (1e-2, 1e-3, 1e-4):
        n_steps = int(2.0 / dt)
        energies = simulate(q0, qd0, dt, n_steps)
        e0 = float(energies[0])
        drift = float(jnp.max(jnp.abs(energies - e0)))
        rel_drift = drift / abs(e0)
        print(f"dt={dt:.0e}  steps={n_steps:6d}  E0={e0:.6f} J  "
              f"max|dE|={drift:.3e} J  rel={rel_drift:.3e}")
