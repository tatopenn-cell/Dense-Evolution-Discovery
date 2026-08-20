"""Experiment 32: implementing and validating the classical Kullback-Leibler
divergence, and testing whether it actually differs from the scalar
quantity already used in dense_evolution.mitigation.healing.

Origin: dense_evolution/mitigation/healing.py's own module docstring
already flags an honest gap -- calculate_vettore_dinamico's core term,
log(E_B/E_A), is a log-likelihood ratio, "the same elementary quantity
Kullback-Leibler divergence is built from", but explicitly NOT a full KL
divergence: it is one un-weighted log-ratio between two scalars, not a
probability-weighted sum over a distribution (D_KL(p||q) = sum_x p(x) *
log(p(x)/q(x))). This experiment builds the real thing (Kullback, S. &
Leibler, R.A., "On Information and Sufficiency", Ann. Math. Statist.
22(1), 79-86, 1951) -- checked directly against the paper's own text
(Section 2, eq. 2.2-2.3), not assumed from the textbook formula alone.
What this module implements is what Kullback & Leibler call I(1:2), "the
mean information for discrimination between H1 and H2" -- what the
broader literature later popularized as "the KL divergence". Their OWN
word "divergence", J(1,2) = I(1:2) + I(2:1) (eq. 2.9), names the
symmetrized sum of both directions instead -- a real terminological
nuance worth being precise about, not the asymmetric quantity implemented
here. This experiment builds the real (asymmetric, I(1:2)) thing and
checks whether it is actually a different signal in practice, not just
different on paper -- the concrete question a healing-pipeline maintainer
would ask before adding it as a diagnostic.

    python scripts/kullback_leibler_divergence.py
"""
import pathlib

import numpy as np
from scipy.stats import entropy as scipy_entropy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dense_evolution import DenseSVSimulator
from dense_evolution.mitigation.healing import calculate_vettore_dinamico, calculate_phi_ab

jax.config.update("jax_enable_x64", True)

_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_EPS = 1e-12


def kl_divergence(p, q):
    """D_KL(p||q) = sum_x p(x) * log2(p(x)/q(x)), in bits.

    0 * log(0/q) = 0 by convention. Returns +inf when p has support where
    q does not (p(x) > 0, q(x) == 0 for some x) -- the correct value for a
    genuine support violation, not a numerical artifact to clamp away
    (the same lesson already learned for sandwiched_renyi_divergence's
    alpha>1 case, Experiment 29).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_is_zero = p < _EPS
    q_is_zero = q < _EPS
    if np.any((~p_is_zero) & q_is_zero):
        return np.inf
    safe_p = np.where(p_is_zero, 1.0, p)
    safe_q = np.where(q_is_zero, 1.0, q)
    terms = np.where(p_is_zero, 0.0, safe_p * np.log2(safe_p / safe_q))
    return float(np.sum(terms))


def part1_validate_against_scipy_reference():
    print("=" * 70)
    print("PART 1: validate kl_divergence against an independent reference")
    print("=" * 70)
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = rng.integers(2, 8)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        mine = kl_divergence(p, q)
        # scipy.stats.entropy(p, q) computes sum(p * log(p/q)) in NATS by
        # default -- convert to bits (log2) for a fair comparison, since
        # this module (like the rest of dense_evolution.mitigation) uses
        # log2 throughout.
        ref_nats = scipy_entropy(p, q)
        ref_bits = ref_nats / np.log(2.0)
        assert abs(mine - ref_bits) < 1e-9, f"trial {trial}: mine={mine}, scipy={ref_bits}"
    print(f"  20/20 random trials (2-8 outcomes) match scipy.stats.entropy to 1e-9 bits -- PASS")


def part2_gibbs_inequality_and_asymmetry():
    print()
    print("=" * 70)
    print("PART 2: Gibbs' inequality (D_KL >= 0) and asymmetry")
    print("=" * 70)
    rng = np.random.default_rng(1)
    min_seen = np.inf
    for _ in range(200):
        n = rng.integers(2, 10)
        p = rng.dirichlet(np.ones(n))
        q = rng.dirichlet(np.ones(n))
        d = kl_divergence(p, q)
        min_seen = min(min_seen, d)
    assert min_seen >= -1e-9, f"D_KL went negative: {min_seen}"
    print(f"  200 random (p,q) pairs, min D_KL(p||q) = {min_seen:.2e} -- never negative, PASS")

    # NOTE: q=[0.1,0.2,0.7] (p reversed) looked like a natural "asymmetric"
    # choice but isn't one -- reversing p's indices happens to make
    # D_KL(p||q) == D_KL(q||p) exactly (the sum is invariant under that
    # particular index permutation composed with swapping p and q). Caught
    # by actually running the assertion rather than assuming asymmetry
    # from the formula alone -- this q avoids that accidental symmetry.
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    d_pq = kl_divergence(p, q)
    d_qp = kl_divergence(q, p)
    assert abs(d_pq - d_qp) > 0.1, "expected a clearly asymmetric example, got near-equal"
    print(f"  D_KL(p||q) = {d_pq:.4f} bits, D_KL(q||p) = {d_qp:.4f} bits -- confirmed asymmetric, PASS")

    p_same = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p_same, p_same) < 1e-12
    print(f"  D_KL(p||p) = {kl_divergence(p_same, p_same):.2e} bits -- zero at equality, PASS")


def part3_support_violation():
    print()
    print("=" * 70)
    print("PART 3: support violation gives +inf, not a finite wrong number")
    print("=" * 70)
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.3, 0.3, 0.4])
    d = kl_divergence(p, q)
    assert np.isfinite(d)
    print(f"  p has zero mass where q doesn't (p=[.5,.5,0], q=[.3,.3,.4]): D_KL = {d:.4f} bits, finite -- PASS")

    p2 = np.array([0.5, 0.5, 0.0])
    q2 = np.array([0.5, 0.0, 0.5])
    d2 = kl_divergence(p2, q2)
    assert np.isinf(d2)
    print(f"  p has mass where q is exactly zero (p=[.5,.5,0], q=[.5,0,.5]): D_KL = {d2} -- correctly +inf, PASS")


def part4_real_use_measurement_distributions():
    print()
    print("=" * 70)
    print("PART 4: applied to real measurement-outcome distributions")
    print("=" * 70)
    # |000> (no gates applied) has a maximally peaked, non-uniform
    # measurement distribution ([1,0,...,0]) -- unlike |+++> (all-H), whose
    # distribution IS already uniform, which would make mixing in a
    # uniform "noise" term a no-op (D_KL identically 0 regardless of
    # noise) -- caught by actually running this and seeing D_KL == 0 at
    # every noise level, not assumed safe from the state choice alone.
    n_qubits = 3
    sim_ideal = DenseSVSimulator(n_qubits)
    p_ideal = np.abs(np.asarray(sim_ideal.sv)) ** 2

    divergences, noise_levels = [], [0.0, 0.05, 0.1, 0.2, 0.4]
    for noise in noise_levels:
        # A crude but real depolarizing-like perturbation of the ideal
        # distribution: mix in a uniform distribution proportional to
        # `noise`, then renormalize -- deliberately simple since the point
        # here is testing the divergence's response, not modeling a real
        # channel (dense_evolution.mitigation already has real noise
        # channels for that; see NoiseModel.apply_to_sv).
        uniform = np.ones_like(p_ideal) / len(p_ideal)
        p_noisy = (1 - noise) * p_ideal + noise * uniform
        p_noisy = p_noisy / p_noisy.sum()
        d = kl_divergence(p_ideal, p_noisy)
        divergences.append(d)
        print(f"  noise={noise:.2f}: D_KL(ideal||noisy) = {d:.4f} bits")

    assert divergences == sorted(divergences), "D_KL should increase monotonically with noise here (by construction)"
    print(f"  monotonically increasing with injected noise, as expected by construction -- PASS")
    return p_ideal, divergences, noise_levels


def part5_compare_against_healing_scalar_on_the_same_states():
    print()
    print("=" * 70)
    print("PART 5: is this actually different from healing.py's existing scalar?")
    print("=" * 70)
    # calculate_vettore_dinamico takes two scalar "energies" E_A, E_B and
    # an alignment factor Phi_AB; the closest honest analogue on the same
    # two states here is E = sum of the (real-valued) state vector itself,
    # or its probability-weighted norm -- there is no single canonical
    # choice, which is itself part of the answer: the healing scalar
    # collapses a whole distribution down to one number before comparing,
    # while D_KL compares the two distributions directly, term by term.
    n_qubits = 3
    sim_ideal = DenseSVSimulator(n_qubits)
    psi_ideal = np.asarray(sim_ideal.sv)  # |000>: non-uniform distribution, see part4's note
    p_ideal = np.abs(psi_ideal) ** 2

    kl_values, healing_values = [], []
    for noise in (0.0, 0.05, 0.1, 0.2, 0.4):
        uniform = np.ones_like(p_ideal) / len(p_ideal)
        p_noisy = (1 - noise) * p_ideal + noise * uniform
        p_noisy = p_noisy / p_noisy.sum()
        kl_values.append(kl_divergence(p_ideal, p_noisy))

        # E_A/E_B here: total probability mass retained in the original
        # support (a real, if simple, scalar summary of "how much did
        # this distribution change") -- E_A is always 1.0 by construction
        # (p_ideal sums to 1), E_B is p_noisy's overlap with p_ideal's
        # support, giving a genuinely different, coarser signal than D_KL.
        E_A = jnp.array(1.0)
        E_B = jnp.array(float(np.sum(np.minimum(p_ideal, p_noisy))))
        Phi_AB = calculate_phi_ab(jnp.array(psi_ideal.real), jnp.array(psi_ideal.real) * (1 - noise), jnp.ones_like(jnp.array(psi_ideal.real)))
        v_dinamic = float(calculate_vettore_dinamico(E_A, E_B, Phi_AB))
        healing_values.append(v_dinamic)
        print(f"  noise={noise:.2f}: D_KL = {kl_values[-1]:.4f} bits, healing v_dinamic = {v_dinamic:.4f}")

    # The point of this part: confirm the two signals are NOT simply
    # rescalings of one another (different functional shape), which is
    # the concrete, checkable version of "this is a genuinely different
    # quantity" rather than just asserting it from the formulas alone.
    kl_arr, heal_arr = np.array(kl_values), np.array(healing_values)
    ratio = kl_arr[1:] / np.maximum(np.abs(heal_arr[1:]), 1e-9)
    assert ratio.std() / max(ratio.mean(), 1e-9) > 0.05, "the two signals look like a trivial rescaling of each other"
    print(f"  ratio D_KL/v_dinamic varies across the noise sweep (not a constant rescaling) -- confirmed genuinely different signals, PASS")
    return kl_values, healing_values


def make_plot(p_ideal, divergences, noise_levels, kl_values, healing_values):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(range(len(p_ideal)), p_ideal, color="#4C72B0")
    axes[0].set_title("Ideal measurement-outcome distribution\n(3-qubit |+++> state)")
    axes[0].set_xlabel("bitstring index")
    axes[0].set_ylabel("probability")

    axes[1].plot(noise_levels, divergences, "o-", color="#DD8452")
    axes[1].set_title("D_KL(ideal || noisy) vs. injected noise")
    axes[1].set_xlabel("noise fraction")
    axes[1].set_ylabel("D_KL (bits)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(noise_levels, kl_values, "o-", label="D_KL (this experiment)", color="#DD8452")
    axes[2].plot(noise_levels, healing_values, "s-", label="healing v_dinamic", color="#55A868")
    axes[2].set_title("D_KL vs. healing.py's existing scalar")
    axes[2].set_xlabel("noise fraction")
    axes[2].set_ylabel("value")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _IMAGES_DIR.mkdir(exist_ok=True)
    out_path = _IMAGES_DIR / "kullback_leibler_divergence.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    part1_validate_against_scipy_reference()
    part2_gibbs_inequality_and_asymmetry()
    part3_support_violation()
    p_ideal, divergences, noise_levels = part4_real_use_measurement_distributions()
    kl_values, healing_values = part5_compare_against_healing_scalar_on_the_same_states()
    make_plot(p_ideal, divergences, noise_levels, kl_values, healing_values)
    print()
    print("=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
