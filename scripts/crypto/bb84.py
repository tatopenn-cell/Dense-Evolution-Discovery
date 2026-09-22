"""
BB84 quantum key distribution, built entirely from existing Dense-Evolution
primitives (statevector, gates, measurement, the depolarizing channel) --
the reference pattern every crypto-q protocol in this section follows:
prepare -> channel -> measure -> sift -> QBER.

Reused, not reimplemented: de.DenseSVSimulator, de.GATES, sim.measure (with
jax_key), NoiseModel.apply_to_sv. No new quantum primitive is introduced.

VALIDATION (dense-evolution 8.1.81, N=5000 rounds, 5 seeds), matching the
crypto-q RFC (Dense-Evolution-Discovery issue #189):

    Scenario              QBER observed        Expected   z-score
    Perfect channel       0.0000 +/- 0.0000    0          pass
    Depolarizing p=0.05   0.0365 +/- 0.0023    0.0333     +0.87
    Depolarizing p=0.10   0.0694 +/- 0.0015    0.0667     +0.56
    Depolarizing p=0.20   0.1386 +/- 0.0034    0.1333     +0.77
    Intercept-resend      0.2518 +/- 0.0017    0.2500     +0.20

All z-scores within +/-2 sigma. Run this file directly to reproduce them.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import dense_evolution as de
from dense_evolution.noise import NoiseModel


def prepare_state(base, bit):
    """base=0 -> Z basis (|0>, |1>), base=1 -> X basis (|+>, |->)."""
    sim = de.DenseSVSimulator(1)
    if base == 0:
        if bit == 1:
            sim.apply_gate_1q(de.GATES["x"], 0)
    else:
        sim.apply_gate_1q(de.GATES["h"], 0)
        if bit == 1:
            sim.apply_gate_1q(de.GATES["z"], 0)
    return sim


def measure_in_basis(sim, base, rng):
    """Measure in Z (base=0) or X (base=1) by applying H first for X."""
    if base == 1:
        sim.apply_gate_1q(de.GATES["h"], 0)
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    return int(sim.measure(0, jax_key=key))


def apply_depolarizing(sim, p, rng):
    """Apply NoiseModel's real depolarizing channel in place."""
    if p == 0.0:
        return
    sv = np.asarray(sim.get_statevector())
    sv_noisy = NoiseModel.apply_to_sv(sv, 1, "depolarizing", p, rng=rng)
    sim.set_initial_state(np.asarray(sv_noisy))


def bb84_round(rng, p_channel=0.0, eve=False):
    a_base = int(rng.integers(0, 2))
    a_bit = int(rng.integers(0, 2))
    sim = prepare_state(a_base, a_bit)

    if eve:
        e_base = int(rng.integers(0, 2))
        e_bit = measure_in_basis(sim, e_base, rng)
        sim = prepare_state(e_base, e_bit)

    apply_depolarizing(sim, p_channel, rng)

    b_base = int(rng.integers(0, 2))
    b_bit = measure_in_basis(sim, b_base, rng)
    return a_base, b_base, a_bit, b_bit


def bb84_run(n_rounds, p_channel=0.0, eve=False, seed=None):
    rng = np.random.default_rng(seed)
    a_bases = np.empty(n_rounds, dtype=np.int8)
    b_bases = np.empty(n_rounds, dtype=np.int8)
    a_bits = np.empty(n_rounds, dtype=np.int8)
    b_bits = np.empty(n_rounds, dtype=np.int8)

    for i in range(n_rounds):
        a_bases[i], b_bases[i], a_bits[i], b_bits[i] = bb84_round(rng, p_channel, eve)

    keep = a_bases == b_bases
    n_sifted = int(keep.sum())
    if n_sifted == 0:
        return 0.0, 0
    qber = float(np.mean(a_bits[keep] != b_bits[keep]))
    return qber, n_sifted


def summarize(runs, label, expected):
    arr = np.array(runs)
    mean = float(arr.mean())
    std_err = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    sigma = np.sqrt(expected * (1 - expected) / 2500) if expected > 0 else 0.0
    z = (mean - expected) / sigma if sigma > 1e-9 else 0.0
    print(
        f"{label:28s}  QBER = {mean:.4f} +/- {std_err:.4f}  "
        f"expected {expected:.4f}  z = {z:+.2f}"
    )


if __name__ == "__main__":
    N = 5000
    SEEDS = range(5)

    print("=" * 72)
    print(f"BB84 validation -- N={N} rounds, {len(SEEDS)} independent seeds each")
    print("=" * 72)

    print("\n[TEST 1] Perfect channel (expected QBER = 0)")
    runs = [bb84_run(N, p_channel=0.0, eve=False, seed=s)[0] for s in SEEDS]
    summarize(runs, "ideal channel", expected=0.0)

    print("\n[TEST 2] Depolarizing channel (expected QBER = 2p/3)")
    for p in [0.05, 0.10, 0.20]:
        runs = [bb84_run(N, p_channel=p, eve=False, seed=s)[0] for s in SEEDS]
        summarize(runs, f"depolarizing p={p:.2f}", expected=2 * p / 3)

    print("\n[TEST 3] Intercept-resend attack (expected QBER = 0.25)")
    runs = [bb84_run(N, p_channel=0.0, eve=True, seed=s)[0] for s in SEEDS]
    summarize(runs, "intercept-resend", expected=0.25)
