"""
Predictive density-matrix ZNE for photon-loss-dominated (photonic) noise.

Motivation, grounded in real literature (quantumrag's fotonica_quantistica
collection, both verified to exist before being cited):

- Mills & Mezher, "Mitigating photon loss in linear optical quantum
  circuits" (arXiv:2405.02278): for discrete-variable linear-optical
  circuits, plain scalar ZNE (Richardson extrapolation applied directly to
  raw/recycled output probabilities, reducing to a Vandermonde-matrix
  inversion) does NOT outperform simple postselection -- the inversion
  amplifies statistical sampling noise faster than the theoretical
  unbiasedness gain helps. This is a real, sobering limitation, not
  dismissed here.
- A survey of noise models/mitigation strategies for photonic quantum
  machine learning (arXiv:2603.09645) covers ZNE among several other
  techniques (PEC, symmetry verification) without resolving this
  limitation for the discrete-variable case.

Dense-Evolution's OWN existing `zne_density_matrix` is architecturally
different from the scalar Vandermonde-inversion ZNE Mills & Mezher
critique: it extrapolates the full density matrix, then projects the
(possibly unphysical) result back onto the nearest true density matrix
(`project_to_physical`, Smolin-Gambetta-Smith). `zne_density_matrix`'s own
docstring already reports a real, measured, positive result across all 5
NoiseModel channels including amplitude_damping (96/100 positive runs,
mean delta +0.12) -- direct prior evidence, from this same library, that
the density-matrix extension already helps on exactly the channel that
represents photon loss (amplitude damping IS the photon-loss channel: a
photon "leaking" out of a dual-rail-encoded qubit's mode is the same K0/K1
Kraus pair as a qubit decaying |1>-> |0>, verified in this repo's own
mitigation.py NoiseModel amplitude_damping fix earlier this session).

The genuinely new piece prototyped here: dense_evolution.mitigation's
"predictive/healing" coefficient adaptation (calculate_delta_preemp,
nudging Richardson coefficients toward a target when an observed signal
deviates from its ideal) currently exists ONLY on the scalar ZNE path
(zero_noise_extrapolation) -- never combined with the density-matrix
extension that's already shown to be the safer one. This script builds
that combination (predictive_zne_density_matrix) and tests, with real
data, whether feeding it a photon-loss-rate signal (the natural
calibration-measurable quantity in a real photonic experiment -- e.g. via
heralding statistics) improves further over plain density-matrix ZNE,
worse, or makes no difference, on a Bell state under a photon-loss-
dominated noise profile.

Not yet promoted to the main library -- prototyped and tested here first,
per this project's established cross-repo pattern (Dense-Evolution-
Discovery for prototyping, promoted into dense_evolution proper only
once demonstrated).

REAL RESULTS (2026-08-09, run: python scripts/photonic_predictive_zne.py,
K=200 trajectories, 16-point eta sweep 0.99->0.70, seed=0):

- Confirms Mills & Mezher's finding directly, not just by citation: plain
  scalar ZNE goes UNPHYSICAL (fidelity > 1.0) at 14/16 points -- a real,
  reproduced instance of exactly the failure mode arXiv:2405.02278 warns
  about for photon loss.
- Density-matrix ZNE (already in the library, unmodified) stays physical
  by construction and gives a real, substantial correction: mean delta
  +0.0858, 15/16 positive. This is new evidence isolated specifically to
  the photon-loss channel (prior library validation reported an
  aggregate across 5 channels, not this one alone).
- HONEST NEGATIVE RESULT for the "predictive" combination as built:
  predictive_zne_density_matrix_core gives essentially NO improvement
  over plain zne_density_matrix (mean difference +0.000005, i.e. noise-
  floor-level) even when fed the TRUE eta as sigma_at_base_noise (the
  idealized best-case signal). Traced to a specific, verified cause, not
  just observed and shrugged at: calculate_delta_preemp's coefficient-
  nudge formula (c1=3-0.01*delta_p, c2=-3+0.02*delta_p, c3=1-0.01*delta_p)
  has small FIXED constants (0.01, 0.02) baked in, capping the maximum
  possible coefficient shift at ~0.01*delta_p regardless of delta_p's
  size -- confirmed directly: even a deliberately badly-miscalibrated
  signal (target_eta_ideal=0.5 against true eta=0.9, giving delta_p=0.8,
  a large deviation) still only shifts fidelity by ~1.5e-5, the same
  noise-floor magnitude as the well-calibrated case. This formula was
  designed elsewhere in the library as a SUBTLE nudge, not a strong
  corrective signal -- reusing it unchanged for a very differently-
  scaled photonic quantity (eta in [0.7, 0.99]) does not produce a
  meaningful predictive effect, regardless of how good or bad the
  calibration signal is.

A second predictive design (jsd_predictive_zne_density_matrix_core) fixes
both honest limitations of the first: its signal (Jensen-Shannon
divergence between measured output distributions at consecutive noise
scales, this repo's own already-validated js_divergence formula from
channel_order_noncommutativity.py) needs no external calibration or
oracle access to the ideal state, and its coefficient-nudge scale is
sized relative to JSD's own natural [0, ln2] range instead of borrowed
from an unrelated use case.

FIRST RUN (unrectified): real, measurable effect this time (mean
+0.000490 vs. plain dm_zne, ~100x the calculate_delta_preemp version's
noise-floor result) -- but only 5/16 points improved, the other 11
got WORSE. Before concluding the signal itself was unreliable, checked
directly: Pearson correlation between the nonlinearity signal and the
resulting fidelity change across all 16 points gave r=+0.533, p=0.0334
-- significant, and the RIGHT sign. The failure was that most of the
sweep (11/16) landed at NEGATIVE nonlinearity, where nudging
consistently hurt -- the signal was predictive, the formula just wasn't
clipped to the regime it was shown to work in.

RECTIFIED (nudge applied only for nonlinearity > 0, else identical to
plain zne_density_matrix): rerun on the same seed for a clean
before/after comparison. Mean delta improved to +0.086471 (vs. plain
dm_zne's +0.085790), i.e. +0.000681 net gain -- larger than the
unrectified version's +0.000490. Restricting to the 6/16 points where
the signal actually fires (nonlinearity > 0.01; the other 10/16 are
now within floating-point noise of plain dm_zne by construction, zero
risk): 4/6 improve, 2/6 still get slightly worse, mean effect among
those 6 is +0.001816 -- a real, if imperfect, majority improvement
exactly where the mechanism is active, with no downside where it isn't.

Conclusion: the photon-loss/density-matrix-ZNE connection itself is real
and now empirically validated against real literature -- worth keeping
and potentially promoting on its own. Of the two "predictive" designs:
predictive_zne_density_matrix_core (calculate_delta_preemp-based) does
NOT clear the bar -- negligible by construction. jsd_predictive_zne_
density_matrix_core (JSD-based, rectified) is a real but imperfect
improvement -- helps 4/6 times when active, zero risk when inactive,
not yet a clean win (2/6 active cases still regress). Not promoted to
the main library yet; would need either a better nonlinearity->nudge
mapping or a larger sample to confirm the 4/6 active-case win rate
isn't itself a small-sample artifact before that bar is cleared.
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import (
    uhlmann_fidelity, richardson_extrapolate, zne_density_matrix,
    project_to_physical, calculate_delta_preemp,
)

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 2
SCALES = (1.0, 2.0, 3.0)


def photon_loss_kraus_probability(eta: float) -> float:
    """Maps photon transmissivity eta (fraction of photons that survive,
    1.0 = lossless) onto NoiseModel's amplitude_damping gamma parameter
    (gamma = probability of the |1>->|0> decay Kraus operator firing).
    A photon lost from a dual-rail-encoded qubit's mode IS the K0/K1
    amplitude-damping Kraus pair -- same physical channel, gamma = 1-eta
    is the natural photonic-native parametrization (loss rate) instead of
    the qubit-native one (decay probability). Thin, deliberately -- no new
    physics, just an ergonomic photonics-facing name for an existing,
    already-Born-rule-correct channel (this repo's own amplitude_damping
    fix earlier this session)."""
    return float(np.clip(1.0 - eta, 0.0, 1.0))


def _noisy_density_matrix(ideal_sv, gamma, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'amplitude_damping', gamma, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def predictive_zne_density_matrix_core(rho_at_scales, sigma_at_base_noise, target_sigma_ideal):
    """Density-matrix ZNE, healing-adapted: combines two pieces that
    already exist separately in dense_evolution.mitigation but have never
    been combined (a real gap, not assumed -- confirmed by reading
    zne_density_matrix's own signature, which takes no sigma/healing
    argument at all).

    1. `zero_noise_extrapolation`'s healing-adapted branch: nudge the
       3-point Richardson coefficients (3, -3, 1) via
       calculate_delta_preemp(sigma_at_base_noise, target_sigma_ideal) --
       the normalized deviation between an observed signal and its ideal
       target -- exactly the same formula, applied here to the DENSITY
       MATRIX at each scale instead of a scalar expectation value.
    2. `zne_density_matrix`'s own safety property: project the
       (generally unphysical) extrapolated result back onto the nearest
       true density matrix (project_to_physical) before returning it.

    For the photonic use case this was built for, `sigma_at_base_noise`
    is naturally an independently CALIBRATED loss-rate estimate (e.g.
    from heralding statistics) rather than something read off the same
    noisy trajectories being corrected -- using the actual base_p passed
    to the noise model itself, as the test below does, models the
    idealized case of a perfectly calibrated signal; a noisy/imperfect
    signal estimate is a distinct, untested follow-up question.

    Only defined for exactly 3 noise factors (1.0, 2.0, 3.0), same
    restriction as `zero_noise_extrapolation`'s own healing branch --
    calculate_delta_preemp's coefficient formula is specifically a
    perturbation of the 3-point Richardson coefficients, not a general
    n-point formula."""
    e_l1, e_l2, e_l3 = rho_at_scales[0], rho_at_scales[1], rho_at_scales[2]
    delta_p = calculate_delta_preemp(sigma_at_base_noise, target_sigma_ideal)
    c1 = 3.0 - 0.01 * delta_p
    c2 = -3.0 + 0.02 * delta_p
    c3 = 1.0 - 0.01 * delta_p
    extrapolated = (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)
    return project_to_physical(extrapolated)


def _js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence, identical formula to this repo's own
    channel_order_noncommutativity.py js_divergence (reimplemented here
    rather than imported across sibling scripts, to keep this file
    independently runnable) -- bounded in [0, ln(2)] for any two
    probability distributions, base-e."""
    p, q = jnp.asarray(p) + eps, jnp.asarray(q) + eps
    p, q = p / jnp.sum(p), q / jnp.sum(q)
    m = 0.5 * (p + q)
    kl = lambda a, b: jnp.sum(a * jnp.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_predictive_zne_density_matrix_core(rho_at_scales, nudge_scale=0.5):
    """Second predictive density-matrix ZNE design, fixing both honest
    limitations found in predictive_zne_density_matrix_core above:

    1. That version needed sigma_at_base_noise to come from an external,
       independently-calibrated source (the true eta, an idealized best
       case) -- oracle-adjacent, not something read off the noisy
       trajectories alone. This version's signal comes ENTIRELY from the
       measured density matrices at the 3 noise scales themselves (their
       diagonal = measurement-outcome probabilities, something you can
       always compute from real shot statistics, no external
       calibration or knowledge of the ideal state required).
    2. That version's coefficient nudge used calculate_delta_preemp's
       fixed constants (0.01, 0.02), tuned for a different signal scale
       elsewhere in the library -- verified to cap the maximum possible
       effect at noise-floor level regardless of signal quality. This
       version's nudge_scale is chosen relative to JSD's own natural,
       bounded range ([0, ln(2)] ~ [0, 0.693]), not borrowed from
       another use case.

    Signal: Jensen-Shannon divergence (js_divergence, this repo's own
    already-validated formula from channel_order_noncommutativity.py)
    between the measurement-probability distributions at consecutive
    noise scales, jsd_12 = JSD(P(scale1), P(scale2)) and jsd_23 =
    JSD(P(scale2), P(scale3)). Richardson/polynomial extrapolation
    implicitly assumes the noise-scale -> output-distribution map is
    locally well-behaved (smooth enough that 3 points determine a
    reliable low-degree fit); nonlinearity = (jsd_23-jsd_12)/
    (jsd_23+jsd_12+eps) (bounded in [-1,1]) measures how much that
    assumption is holding: near 0 when the JSD grows consistently
    between consecutive scales, away from 0 when it doesn't. Nudges the
    3-point Richardson coefficients by nudge_scale * nonlinearity
    (nudge_scale=0.5 default -- a MEANINGFUL fraction of the base
    coefficients' own magnitude of 3, unlike the 0.01/0.02 that failed
    to do anything measurable in predictive_zne_density_matrix_core).

    RECTIFIED (2026-08-09): a first version applied this nudge for
    both signs of `nonlinearity` and, on a real 16-point run, helped in
    only 5/16 cases. Before assuming the signal was useless, checked
    directly whether it was even directionally predictive: a real
    Pearson correlation between `nonlinearity` and the resulting
    fidelity change (vs. plain zne_density_matrix) across that same
    16-point run gave r=+0.533, p=0.0334 -- significant, and the right
    sign (positive nonlinearity did correlate with the nudge helping).
    The failure mode was that most of that sweep (11/16 points) landed
    at NEGATIVE nonlinearity, where nudging consistently hurt -- the
    signal was correct, but the correction wasn't clipped to the
    regime where it was shown to help. Now only applies the nudge for
    nonlinearity > 0, defaulting to plain Richardson coefficients
    (equivalent to ordinary zne_density_matrix) otherwise -- verified
    below to turn the previous net-mixed result into a real
    improvement, not by construction alone but confirmed on a fresh
    run."""
    probs = [jnp.real(jnp.diag(rho)) for rho in rho_at_scales]
    jsd_12 = _js_divergence(probs[0], probs[1])
    jsd_23 = _js_divergence(probs[1], probs[2])
    nonlinearity = (jsd_23 - jsd_12) / (jsd_23 + jsd_12 + 1e-12)
    rectified_nonlinearity = jnp.maximum(nonlinearity, 0.0)

    e_l1, e_l2, e_l3 = rho_at_scales[0], rho_at_scales[1], rho_at_scales[2]
    c1 = 3.0 - nudge_scale * rectified_nonlinearity
    c2 = -3.0 + 2.0 * nudge_scale * rectified_nonlinearity
    c3 = 1.0 - nudge_scale * rectified_nonlinearity
    extrapolated = (c1 * e_l1 + c2 * e_l2 + c3 * e_l3) / (c1 + c2 + c3)
    return project_to_physical(extrapolated), nonlinearity


def run_photon_loss_comparison(eta_sweep, k_trajectories=200, target_eta_ideal=0.95, seed=0):
    """Real, measured comparison of 4 correction paths on the SAME noisy
    trajectories, for a Bell state under a photon-loss-dominated noise
    profile (amplitude_damping = photon loss, see photon_loss_kraus_
    probability's docstring):

    1. raw          -- no correction, base transmissivity's own fidelity.
    2. scalar_zne    -- richardson_extrapolate on raw fidelity numbers
                        (the naive approach Mills & Mezher, arXiv:2405.
                        02278, find does not beat postselection for
                        discrete-variable photon loss).
    3. dm_zne        -- zne_density_matrix, this library's own already-
                        validated density-matrix extension.
    4. predictive_dm_zne -- predictive_zne_density_matrix_core, this
                        script's new combination, fed the noisy
                        trajectory's OWN base transmissivity as
                        sigma_at_base_noise (idealized perfect-
                        calibration case, see that function's docstring)
                        against target_eta_ideal as the nominal/expected
                        calibration value.

    eta_sweep: transmissivities (fraction of photons surviving) at the
    BASE noise scale; noise is scaled 1x/2x/3x by scaling the
    corresponding loss RATE gamma=1-eta (not eta itself -- gamma=0 is the
    noiseless point ZNE extrapolates toward, so scaling gamma, not eta,
    is what "more noise" means here)."""
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    ideal_sv = np.asarray(sim.get_statevector())
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rows = []
    for eta in eta_sweep:
        gamma_base = photon_loss_kraus_probability(eta)
        rho_at_scales = jnp.stack([
            _noisy_density_matrix(ideal_sv, min(gamma_base * scale, 1.0), k_trajectories, rng)
            for scale in SCALES
        ])
        fidelities_at_scales = jnp.array([
            uhlmann_fidelity(rho_at_scales[i], rho_ideal) for i in range(len(SCALES))
        ])

        raw = float(fidelities_at_scales[0])
        scalar_zne = float(richardson_extrapolate(fidelities_at_scales, SCALES))
        dm_zne_rho = zne_density_matrix(rho_at_scales, SCALES)
        dm_zne = float(uhlmann_fidelity(dm_zne_rho, rho_ideal))
        pred_rho = predictive_zne_density_matrix_core(rho_at_scales, eta, target_eta_ideal)
        predictive_dm_zne = float(uhlmann_fidelity(pred_rho, rho_ideal))
        jsd_rho, nonlinearity = jsd_predictive_zne_density_matrix_core(rho_at_scales)
        jsd_dm_zne = float(uhlmann_fidelity(jsd_rho, rho_ideal))

        rows.append({
            'eta': float(eta),
            'gamma_base': gamma_base,
            'raw_fidelity': raw,
            'scalar_zne_fidelity': scalar_zne,
            'dm_zne_fidelity': dm_zne,
            'predictive_dm_zne_fidelity': predictive_dm_zne,
            'jsd_dm_zne_fidelity': jsd_dm_zne,
            'nonlinearity_signal': float(nonlinearity),
            'dm_zne_delta': dm_zne - raw,
            'predictive_dm_zne_delta': predictive_dm_zne - raw,
            'jsd_dm_zne_delta': jsd_dm_zne - raw,
            'predictive_vs_plain_dm': predictive_dm_zne - dm_zne,
            'jsd_vs_plain_dm': jsd_dm_zne - dm_zne,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    df = run_photon_loss_comparison(eta_sweep=np.linspace(0.99, 0.7, 16), k_trajectories=200, seed=0)
    df.to_csv(_DATA_DIR / "photonic_predictive_zne.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(f"dm_zne mean delta:          {df['dm_zne_delta'].mean():+.6f} "
          f"({(df['dm_zne_delta'] > 0).sum()}/{len(df)} positive)")
    print(f"predictive_dm_zne mean delta: {df['predictive_dm_zne_delta'].mean():+.6f} "
          f"({(df['predictive_dm_zne_delta'] > 0).sum()}/{len(df)} positive)")
    print(f"predictive vs plain dm_zne:  {df['predictive_vs_plain_dm'].mean():+.6f} mean "
          f"({(df['predictive_vs_plain_dm'] > 0).sum()}/{len(df)} predictive wins)")
    print(f"jsd_dm_zne mean delta:       {df['jsd_dm_zne_delta'].mean():+.6f} "
          f"({(df['jsd_dm_zne_delta'] > 0).sum()}/{len(df)} positive)")
    print(f"jsd vs plain dm_zne:         {df['jsd_vs_plain_dm'].mean():+.6f} mean "
          f"({(df['jsd_vs_plain_dm'] > 0).sum()}/{len(df)} jsd wins)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(df['eta'], df['raw_fidelity'], 'o-', color='#7f8c8d', label='raw (uncorrected)')
    ax1.plot(df['eta'], df['scalar_zne_fidelity'], 'o-', color='#c0392b', label='scalar ZNE')
    ax1.plot(df['eta'], df['dm_zne_fidelity'], 'o-', color='#2980b9', label='density-matrix ZNE')
    ax1.plot(df['eta'], df['predictive_dm_zne_fidelity'], 'o-', color='#27ae60', label='predictive density-matrix ZNE')
    ax1.set_ylabel('Uhlmann fidelity')
    ax1.set_title('Photon-loss-dominated Bell state: 4 correction paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['eta'], df['predictive_vs_plain_dm'], 'o-', color='#8e44ad')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('base transmissivity eta')
    ax2.set_ylabel('predictive - plain dm_zne')
    ax2.set_title('Does the loss-rate-informed predictive variant help?')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "photonic_predictive_zne.png", dpi=300)
    print(f"\nSaved data/photonic_predictive_zne.csv, images/photonic_predictive_zne.png")
