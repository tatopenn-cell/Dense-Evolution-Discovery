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

Conclusion: the photon-loss/density-matrix-ZNE connection itself is real
and now empirically validated against real literature -- worth keeping
and potentially promoting on its own. The "predictive" addition, as
built here (direct reuse of calculate_delta_preemp), does NOT clear the
bar for promotion -- it would need a redesigned coefficient-nudge
formula (different scaling constants, not the ones tuned for
mitigation.py's original healing use case) to have a chance at a real
effect, untested here.
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

        rows.append({
            'eta': float(eta),
            'gamma_base': gamma_base,
            'raw_fidelity': raw,
            'scalar_zne_fidelity': scalar_zne,
            'dm_zne_fidelity': dm_zne,
            'predictive_dm_zne_fidelity': predictive_dm_zne,
            'dm_zne_delta': dm_zne - raw,
            'predictive_dm_zne_delta': predictive_dm_zne - raw,
            'predictive_vs_plain_dm': predictive_dm_zne - dm_zne,
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
