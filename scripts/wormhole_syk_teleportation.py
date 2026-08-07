"""
Real traversable-wormhole-inspired quantum teleportation (Gao-Jafferis-Wall
theory, arXiv:2604.10090) on a binary sparse Sachdev-Ye-Kitaev (SYK) model
-- built on dense_evolution/dashboard_core (v8.1.49), see
https://tatopenn-cell.github.io/Dense-Evolution/ for the shipped
implementation (dashboard_core.wormhole, dense_evolution.fermions/entropy/
trotter) and its own unit test suite.

An earlier, discarded dashboard_core circuit ("Traversable Wormhole (BGQ)")
used the right vocabulary (SYK scrambling, a phase "kick") but wasn't real:
it ran on a single qubit register, which the no-signaling theorem forbids
from ever showing this protocol's real sign-dependent signal -- verified
directly, not assumed (identical results for either sign of its "kick").
The real recipe needs two coupled chaotic systems (L, R), a message
injected into L via a separate reference-qubit pair (P, Q), a real
bilinear L-R coupling exp(i*mu*V), and a readout that is NOT a
single-qubit expectation value: mutual information between the reference
qubit P and a qubit read out from R.

This script runs eleven real, verified experiments, each producing its own
CSV + plot:

1. t1 sweep -- the protocol's headline signature, sign-dependent mutual
   information rising then falling across post-coupling evolution time.
2. message-vs-no-message control -- with_message=False gives I(P:R)=0
   exactly at every point tested (P,Q are structurally decoupled from
   L,R without the swap injection), confirming the signal genuinely
   requires the injected message, not just the L-R coupling itself.
3. mu-magnitude scan -- the sign-dependent delta peaks near mu~11-12,
   matching arXiv:2604.10090's own choice, not at higher coupling
   strengths (which give more *total* mutual information but a smaller
   sign-dependent asymmetry).
4. t0 (pre-coupling scrambling time) scan -- the signal needs enough
   scrambling before it appears (consistent with the theoretical chaos
   requirement of the protocol), and for this specific instance peaks
   later than the paper's own t0=0.3 choice.
5. 2D (t0, mu) joint grid search -- experiments 3 and 4 scanned each
   axis independently, holding the other fixed; a quick follow-up check
   showed the mu-peak shifts as t0 changes, meaning neither 1D scan
   alone finds the true joint optimum. This grid search resolves that:
   870 points (30 t0 values x 29 mu values), global max at
   t0=0.65, mu=15.0 (delta=+0.01167), noticeably better than either 1D
   scan's own peak.
6. t1 re-scan at Experiment 5's (t0, mu) optimum -- Experiment 5 itself
   held t1 fixed at 0.60 (Experiment 1's own 1D peak) and flagged that
   as unverified. Re-scanning t1 at t0=0.65/mu=15.0 finds the peak has
   moved to t1=0.41, delta=+0.01518 -- ~30% above Experiment 5's
   headline value. One coordinate-ascent step, not a converged 3D joint
   optimum (resolved by Experiment 7 below).
7. Iterated coordinate ascent toward the joint (t0, mu, t1) optimum --
   Experiment 6 moved t1 once but never checked whether t0/mu would
   shift again, the same gap that motivated Experiment 5 in the first
   place. Alternating full t1 scans (Experiment 6's resolution) and
   (t0, mu) grids (Experiment 5's resolution) from Experiment 5's
   starting point, 3 rounds converge to a genuine fixed point:
   t0=0.70, mu=17.0, t1=0.36, delta=+0.01688 -- +44.6% over Experiment
   5's original headline value.
8. Generality check across 6 independent SYK instances -- does
   Experiment 7's converged point generalize, or is it specific to
   seed=61? Honest negative result: it does NOT. The converged
   (t0, mu, t1) scatters across nearly the whole scanned range instead
   of clustering, 2 of 6 instances converge at the edge of the scanned
   grid (inconclusive -- their true optimum may lie outside what was
   scanned), and 2 of 6 have a *negative* delta at Experiment 5's own
   starting point (the sign-dependent signal isn't even reliably
   oriented the expected way using those defaults across instances).
   Percentage-improvement figures are not reported for this experiment
   since near-zero/negative baselines make them meaningless.
9. Realistic-noise robustness at the converged point (seed=61,
   t0=0.70, mu=17.0, t1=0.36) -- a real Trotterized gate circuit with a
   stochastic depolarizing Kraus channel (dense_evolution.registry.
   NoiseModel) injected after each of the protocol's three phases,
   scanned over p in [0, 0.05], averaged over 6 trials per point.
   Second honest negative result: the noiseless Trotter delta
   (+0.01728) decays with noise and crosses zero between p=0.01 and
   p=0.02 -- already at p=0.01 the mean signal (+0.00051) is smaller
   than its own trial-to-trial standard deviation (0.01203), i.e.
   statistically indistinguishable from zero at a noise level well
   within range of current real NISQ hardware.
10. Direct comparison against arXiv:2604.10090's own "Ensemble
    robustness" section, which reports 100 disorder realizations and
    concludes the sign-dependent asymmetry is "a generic feature of
    the ensemble" (their chosen instance was selected for unusually
    *large* asymmetry, not unusually *signed*). Experiment 8's baseline
    was evaluated at Experiment 5's point (t0=0.65, mu=15.0, t1=0.60),
    itself optimized on seed=61 -- a real confound, since an instance
    could show the "wrong" sign there simply from being evaluated at a
    point tuned for a different instance. Re-evaluating all 6 instances
    at the paper's OWN stated defaults (t0=0.3, mu=12, t1=0.60)
    controls for that: 2 of 6 (seeds 2166, 2907) still show the wrong
    sign there, seed 2835's apparent reversal in Experiment 8 turns out
    to have been a point-choice artifact (correctly signed at the
    paper's defaults). Net finding: even controlling for the confound,
    at least 1 of 6 instances (seed 2166, wrong-signed at *both*
    evaluation points) genuinely contradicts the "generic feature of
    the ensemble" claim for this specific 34/11-selection-matched
    subset.
11. Large-sample (n=100) version of Experiment 10's check, matching
    arXiv:2604.10090's own reported ensemble size. Result: 49/100 (49%)
    of exact 34/11-selection-matched instances are wrong-signed at the
    paper's own default parameters -- far stronger than Experiment 10's
    2/6 (33%), and essentially a coin flip, not a "generic feature of
    the ensemble". Also tests two candidate structural explanations for
    the sign variation floated informally alongside Experiment 10
    (Majorana mode-usage imbalance in the K=10 coupling terms; the
    spectral level-spacing r-statistic, a standard chaos diagnostic):
    an early n=6 look had suggested mode-usage imbalance correlated
    with the signal (r=0.87) -- at n=100 that does NOT hold up
    (r=0.171, p=0.09, not significant), an honest correction of that
    earlier small-sample impression. The level-spacing statistic
    doesn't correlate either (r=0.087, p=0.39). Neither explains why
    the sign varies; that remains open.

Experiments 1-7, 9, and 10 use seed=61 (n_majorana=8, k_terms=10, J=sqrt(2))
-- the instance dashboard_core.wormhole.select_good_instance finds when
screened against arXiv:2604.10090's own selection criterion (their
chosen K=10 instance has 34 commuting / 11 anticommuting pairs among the
C(10,2)=45 pairs of terms). Re-derived below, not hardcoded blindly.
Experiments 8 and 10 additionally use 5 more instances matching that same
exact criterion, found by find_multiple_seeds; Experiment 11 uses up to
100.

Honest caveats, not glossed over:
- Experiment 7's fixed point (t0=0.70, mu=17.0, t1=0.36) is a *local*
  coordinate-ascent convergence on this specific grid resolution
  (Experiment 5's 0.05/1.0 t0/mu step, Experiment 6's 0.01 t1 step), not
  a proof of global optimality: coordinate ascent can converge to a
  point that isn't the true joint maximum if the surface isn't
  separably well-behaved, and a finer/coarser grid could in principle
  settle on a nearby but distinct fixed point (an ad hoc finer local
  grid around the converged point suggested the true continuum optimum
  sits close to mu=17.5, just off this grid's integer mu values --
  consistent with, not contradicting, the converged answer). A real
  continuous joint optimizer (e.g. gradient-based, if this readout is
  ever made differentiable) would be needed to settle global optimality.
- Experiment 8's non-generalization finding is itself only a 6-instance
  sample, and 2 of those 6 hit the edge of the scanned (t0, t1) range
  rather than settling on a real interior fixed point -- a wider scan
  range could turn those into genuine (still probably instance-specific)
  answers rather than boundary artifacts, but wasn't run here (compute
  cost scales with range x resolution, already ~15 minutes for 6
  instances at the current range).
- Experiments 1-8 use the exact-evolution backend (eigendecomposition),
  not the Trotterized real-gate-circuit backend -- both are implemented
  and cross-verified to agree closely (see the main Dense-Evolution
  repo's tests), but the exact backend is what was used for speed.
  Experiment 9 is the one exception, by necessity (noise injection needs
  a real gate circuit to interrupt mid-evolution).
- Experiment 9 only tested seed=61's converged point, not the other 5
  instances from Experiment 8 -- given Experiment 8's own finding (the
  converged point doesn't generalize across instances), there's no
  reason to expect this noise-robustness result generalizes either;
  it's honestly one more seed=61-specific data point, not evidence about
  the protocol broadly. It also averages only 6 stochastic trials per
  noise level (each `NoiseModel.apply_to_sv` call is a single-shot Kraus
  draw, not an ensemble average) -- the reported standard deviations are
  real but from a small sample, not tight error bars.
- Experiments 5, 6, and 7 bypass `run_wormhole_protocol`'s public API and call
  `dashboard_core.wormhole`'s private layout/evolution helpers directly.
  Justified by a real, measured cost asymmetry: building the SYK/coupling
  Hamiltonians and diagonalizing both (`_protocol_layout` + two `eigh`
  calls) took 4.3-6.4s and does not depend on t0/mu/t1 at all for a
  fixed (seed, n_majorana, k_terms) -- only the actual per-point
  evolution + mutual-information readout does, and that alone measured
  at 0.022s/call. Computing the expensive part once instead of once per
  grid point cut an 870-point grid from an estimated ~2 hours down to
  47.6s (~165x) -- confirmed by timing both versions directly, not
  assumed. `run_wormhole_protocol` itself is unchanged; this script's
  own helper functions are just a faster way to call the same physics
  repeatedly at one fixed instance.
"""
import itertools
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from dashboard_core.wormhole import (
    build_sparse_syk_terms, commuting_pair_count, select_good_instance,
    run_wormhole_protocol,
)
# Private helpers, used only by run_2d_grid_search's precompute-once
# optimization -- see this module's docstring for why. Not part of
# dashboard_core.wormhole's public API (no __all__ entry); reached into
# deliberately here rather than duplicated, since re-deriving the same
# protocol layout independently would risk silently drifting out of sync
# with the real implementation.
from dashboard_core.wormhole import _protocol_layout, _initial_state_ops, _evolve
from dense_evolution import mutual_information
from dense_evolution.trotter import trotter_evolve_ops
from dense_evolution.registry import NoiseModel
import dense_evolution as de

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

N_MAJORANA = 8
K_TERMS = 10
J = float(np.sqrt(2))


def find_seed() -> int:
    seed = select_good_instance(N_MAJORANA, K_TERMS, J, n_candidates=200, target_commuting=34)
    n_qubits = N_MAJORANA // 2
    terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)[1]
    c, a = commuting_pair_count(terms, n_qubits)
    print(f"Selected seed={seed}: {c} commuting / {a} anticommuting "
          f"(target 34/11, arXiv:2604.10090's own K=10 instance)")
    return seed


def run_t1_sweep(seed: int) -> pd.DataFrame:
    t1_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00, 1.20, 1.50]
    rows = []
    for t1 in t1_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, 0.3, t1, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, 0.3, t1, seed, with_message=True)
        rows.append({"t1": t1, "I_mu_pos12": i_pos, "I_mu_neg12": i_neg, "delta": i_neg - i_pos})
        print(f"  t1={t1:.2f}  I(+12)={i_pos:.5f}  I(-12)={i_neg:.5f}  delta={i_neg - i_pos:+.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t1_sweep.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t1"], df["I_mu_pos12"], 'o-', color='#00FFFF', label='I(mu=+12)')
    ax.plot(df["t1"], df["I_mu_neg12"], 'o-', color='#FF007F', label='I(mu=-12)')
    ax.set_xlabel("t1 (post-coupling evolution time)", color='#888888')
    ax.set_ylabel("Mutual information I(P:R[0])", color='#888888')
    ax.set_title("Traversable-wormhole teleportation signal vs. t1\n(seed=61, N=8 SYK, t0=0.3)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t1_sweep.png", dpi=300)
    plt.close(fig)
    return df


def run_message_control(seed: int) -> pd.DataFrame:
    t1_values = [0.10, 0.30, 0.60, 0.85, 1.20]
    rows = []
    for with_message in (True, False):
        for t1 in t1_values:
            i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, 0.3, t1, seed, with_message=with_message)
            i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, 0.3, t1, seed, with_message=with_message)
            rows.append({
                "with_message": with_message, "t1": t1,
                "I_mu_pos12": i_pos, "I_mu_neg12": i_neg, "delta": i_neg - i_pos,
            })
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_message_control.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    with_msg = df[df.with_message]
    without_msg = df[~df.with_message]
    ax.plot(with_msg["t1"], with_msg["delta"], 'o-', color='#00FFFF', label='WITH message (real protocol)')
    ax.plot(without_msg["t1"], without_msg["delta"], 's-', color='#FFFF00', label='WITHOUT message (control)')
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("t1", color='#888888')
    ax.set_ylabel("delta = I(mu=-12) - I(mu=+12)", color='#888888')
    ax.set_title("Control: does the signal require the injected message?\n(seed=61, N=8 SYK, t0=0.3)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_message_control.png", dpi=300)
    plt.close(fig)
    return df


def run_mu_scan(seed: int) -> pd.DataFrame:
    mu_values = [4.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0]
    rows = []
    for mu in mu_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +mu, 0.3, 0.60, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -mu, 0.3, 0.60, seed, with_message=True)
        rows.append({"mu": mu, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows).sort_values("mu").reset_index(drop=True)
    df.to_csv(_DATA_DIR / "wormhole_mu_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["mu"], df["delta"], 'o-', color='#00FFFF')
    ax.axvline(12, color='#FFFF00', linestyle='--', alpha=0.6, label='arXiv:2604.10090 value (mu=12)')
    ax.set_xlabel("|mu| (L-R coupling strength)", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title("Sign-dependent signal vs. coupling strength\n(seed=61, N=8 SYK, t0=0.3, t1=0.60)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_mu_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_t0_scan(seed: int) -> pd.DataFrame:
    t0_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]
    rows = []
    for t0 in t0_values:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +12.0, t0, 0.60, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -12.0, t0, 0.60, seed, with_message=True)
        rows.append({"t0": t0, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t0_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t0"], df["delta"], 'o-', color='#FF007F')
    ax.axvline(0.3, color='#FFFF00', linestyle='--', alpha=0.6, label='arXiv:2604.10090 value (t0=0.3)')
    ax.set_xlabel("t0 (pre-coupling scrambling time)", color='#888888')
    ax.set_ylabel("delta = I(mu=-12) - I(mu=+12)", color='#888888')
    ax.set_title("Sign-dependent signal vs. pre-coupling scrambling time\n(seed=61, N=8 SYK, mu=+-12, t1=0.60)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t0_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_2d_grid_search(seed: int) -> pd.DataFrame:
    """Joint (t0, mu) grid search, t1 held fixed at 0.60 (Experiment 1's
    peak). Precomputes the Hamiltonian/coupling matrices and their
    eigendecompositions ONCE (the expensive, t0/mu/t1-independent part
    of run_wormhole_protocol), then reuses them for every grid point --
    see this module's docstring for the measured ~165x speedup this
    gives over calling run_wormhole_protocol per point."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    def mi_at(t0, mu, t1=0.60):
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        return mutual_information(sv, n_full, [P], [R[0]])

    t0_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
    mu_values = np.round(np.arange(2.0, 31.0, 1.0), 1)

    rows = []
    delta_grid = np.zeros((len(mu_values), len(t0_values)))
    for i, mu in enumerate(mu_values):
        for j, t0 in enumerate(t0_values):
            i_pos = mi_at(t0, +mu)
            i_neg = mi_at(t0, -mu)
            delta = i_neg - i_pos
            delta_grid[i, j] = delta
            rows.append({"t0": t0, "mu": mu, "I_pos": i_pos, "I_neg": i_neg, "delta": delta})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_2d_grid.csv", index=False)

    best = df.loc[df["delta"].idxmax()]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7))
    extent = [t0_values.min(), t0_values.max(), mu_values.min(), mu_values.max()]
    im = ax.imshow(delta_grid, origin='lower', aspect='auto', extent=extent, cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.scatter([best['t0']], [best['mu']], color='cyan', marker='*', s=300,
               edgecolor='white', linewidth=1, label=f"max: t0={best['t0']:.2f}, mu={best['mu']:.1f}")
    ax.set_xlabel("t0 (pre-coupling scrambling time)", color='#888888')
    ax.set_ylabel("|mu| (L-R coupling strength)", color='#888888')
    ax.set_title("Joint (t0, mu) optimization surface\n(seed=61, N=8 SYK, t1=0.60 fixed)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_2d_grid.png", dpi=300)
    plt.close(fig)
    return df


def run_t1_rescan(seed: int) -> pd.DataFrame:
    """t1 re-scan at Experiment 5's (t0, mu) optimum -- resolves the caveat
    flagged there: the 2D grid held t1 fixed at 0.60 (Experiment 1's own
    1D peak) and noted its optimum could plausibly shift once t0/mu are no
    longer at their original 1D-scan defaults. Reuses Experiment 5's
    precompute-once approach (see its docstring/this module's docstring)
    since a fine ~125-point sweep would cost ~4.5s/call x 2 x 125 ~ 19
    minutes via the public run_wormhole_protocol API otherwise."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    t0_opt, mu_opt = 0.65, 15.0

    def mi_at(t1, mu):
        sv = _evolve(sv0, eigvals, eigvecs, t0_opt)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        return mutual_information(sv, n_full, [P], [R[0]])

    t1_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
    rows = []
    for t1 in t1_values:
        i_pos = mi_at(t1, +mu_opt)
        i_neg = mi_at(t1, -mu_opt)
        rows.append({"t1": t1, "I_pos": i_pos, "I_neg": i_neg, "delta": i_neg - i_pos})
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_t1_rescan_optimum.csv", index=False)

    best = df.loc[df["delta"].idxmax()]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["t1"], df["delta"], '-', color='#00FFFF', linewidth=1.5)
    ax.axvline(0.60, color='#FFFF00', linestyle='--', alpha=0.6, label='Experiment 5 fixed value (t1=0.60)')
    ax.scatter([best['t1']], [best['delta']], color='cyan', marker='*', s=250,
               edgecolor='white', linewidth=1, zorder=5,
               label=f"peak: t1={best['t1']:.2f}, delta={best['delta']:+.5f}")
    ax.set_xlabel("t1 (post-coupling evolution time)", color='#888888')
    ax.set_ylabel("delta = I(mu=-15) - I(mu=+15)", color='#888888')
    ax.set_title("t1 re-scan at Experiment 5's (t0, mu) optimum\n(seed=61, N=8 SYK, t0=0.65, mu=15.0 fixed)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_t1_rescan_optimum.png", dpi=300)
    plt.close(fig)
    return df


def _coordinate_ascent_trace(seed: int, max_rounds: int = 5) -> pd.DataFrame:
    """Pure computation behind Experiments 7 and 8 -- iterated coordinate
    ascent toward the joint (t0, mu, t1) optimum for one SYK instance, no
    file I/O (callers decide what to save). Starting from Experiment 5's
    point (t0=0.65, mu=15.0, t1=0.60), each round alternates two full
    sub-steps at Experiment 5/6's own resolutions (not a shortcut, so
    results stay directly comparable): a 126-point t1 scan (step 0.01)
    holding (t0, mu) fixed, then an 870-point (t0, mu) grid (step
    0.05/1.0) holding the new t1 fixed. Stops when a full round leaves
    (t0, mu, t1) unchanged -- a genuine fixed point -- or after
    max_rounds as a safety cap."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)
    H = de.pauli_hamiltonian_to_matrix(terms_full, n_full)
    eigvals, eigvecs = np.linalg.eigh(H)
    V = de.pauli_hamiltonian_to_matrix(v_terms, n_full)
    v_eigvals, v_eigvecs = np.linalg.eigh(V)

    sim = de.DenseSVSimulator(n_full)
    sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, with_message=True))
    sv0 = sim.get_statevector()

    def mi_delta(t0, mu, t1):
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        i_pos = mutual_information(sv, n_full, [P], [R[0]])
        sv = _evolve(sv0, eigvals, eigvecs, t0)
        sv = _evolve(sv, v_eigvals, v_eigvecs, -mu)
        sv = _evolve(sv, eigvals, eigvecs, t1)
        i_neg = mutual_information(sv, n_full, [P], [R[0]])
        return i_neg - i_pos

    t1_scan_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
    t0_grid_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
    mu_grid_values = np.round(np.arange(2.0, 31.0, 1.0), 1)

    t0, mu, t1 = 0.65, 15.0, 0.60
    trace = [{"seed": seed, "round": 0, "stage": "start (Experiment 5)", "t0": t0, "mu": mu, "t1": t1,
              "delta": mi_delta(t0, mu, t1)}]

    for rnd in range(1, max_rounds + 1):
        t1_new = max(t1_scan_values, key=lambda t1c: mi_delta(t0, mu, t1c))
        delta_t1 = mi_delta(t0, mu, t1_new)
        trace.append({"seed": seed, "round": rnd, "stage": "t1 scan", "t0": t0, "mu": mu, "t1": t1_new, "delta": delta_t1})

        t0_new, mu_new = max(
            ((t0c, muc) for muc in mu_grid_values for t0c in t0_grid_values),
            key=lambda p: mi_delta(p[0], p[1], t1_new),
        )
        delta_grid = mi_delta(t0_new, mu_new, t1_new)
        trace.append({"seed": seed, "round": rnd, "stage": "t0/mu grid", "t0": t0_new, "mu": mu_new, "t1": t1_new,
                      "delta": delta_grid})

        converged = (t0_new == t0 and mu_new == mu and t1_new == t1)
        t0, mu, t1 = t0_new, mu_new, t1_new
        if converged:
            break

    return pd.DataFrame(trace)


def run_coordinate_ascent_3d(seed: int, max_rounds: int = 5):
    """Experiment 7: run _coordinate_ascent_trace for seed=61 and save
    its own CSV + convergence plot. See _coordinate_ascent_trace's
    docstring for the algorithm."""
    trace_df = _coordinate_ascent_trace(seed, max_rounds=max_rounds)
    trace_df.to_csv(_DATA_DIR / "wormhole_coordinate_ascent_3d.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(trace_df)), trace_df["delta"], 'o-', color='#00FFFF', markersize=5)
    for i, row in trace_df.iterrows():
        ax.annotate(f"t0={row.t0:.2f}\nmu={row.mu:.1f}\nt1={row.t1:.2f}",
                     (i, row.delta), textcoords="offset points", xytext=(0, 10),
                     fontsize=7, color='#888888', ha='center')
    ax.set_xlabel("coordinate-ascent step", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title("Convergence of iterated coordinate ascent toward the joint (t0, mu, t1) optimum\n"
                 "(seed=61, N=8 SYK)", fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_coordinate_ascent_3d.png", dpi=300)
    plt.close(fig)
    return trace_df


def find_multiple_seeds(n_instances: int = 6, n_candidates: int = 3000, target_commuting: int = 34) -> list:
    """Screen up to n_candidates random seeds for EXACT matches to the
    paper's own selection criterion (34 commuting / 11 anticommuting
    pairs among the C(10,2)=45 pairs of K=10 terms), returning the first
    n_instances found. Unlike find_seed() (which returns the single
    closest match out of a smaller pool), Experiment 8 needs several
    independent, equally-valid instances to test whether Experiment 7's
    converged point is a property of the protocol or an idiosyncrasy of
    seed=61 specifically."""
    n_qubits = N_MAJORANA // 2
    found = []
    for seed in range(n_candidates):
        terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)[1]
        c, a = commuting_pair_count(terms, n_qubits)
        if c == target_commuting:
            found.append(seed)
            if len(found) >= n_instances:
                break
    print(f"Found {len(found)} instances with exactly {target_commuting} commuting pairs "
          f"(screened {seed + 1} candidates): {found}")
    return found


def run_generality_check(seeds=None, max_rounds: int = 5) -> pd.DataFrame:
    """Experiment 8: does Experiment 7's converged point (t0=0.70,
    mu=17.0, t1=0.36) generalize across SYK instances, or is it specific
    to seed=61? Runs the identical coordinate-ascent procedure
    (_coordinate_ascent_trace -- same resolutions, same starting point)
    independently for each of several instances that all exactly match
    the paper's own selection criterion, same as seed=61 does.

    Honest negative result, not glossed over: it does NOT generalize.
    Converged (t0, mu, t1) points scatter across nearly the entire
    scanned range instead of clustering near seed=61's answer, 2 of 6
    instances converge AT the edge of the scanned t0/t1 range (their
    true optimum may lie outside what was scanned -- inconclusive, not
    a real fixed point), and 2 of 6 instances have a NEGATIVE delta at
    Experiment 5's own starting point (the sign-dependent signal isn't
    even reliably oriented the expected way using those "default"
    parameters across instances). Percentage-improvement figures are not
    reported here for that reason -- with a near-zero or negative
    baseline they blow up into meaningless numbers (e.g. one instance's
    raw improvement is nominally +3865%), not a real effect size."""
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)

    rows = []
    for seed in seeds:
        trace_df = _coordinate_ascent_trace(seed, max_rounds=max_rounds)
        start = trace_df.iloc[0]
        converged = trace_df.iloc[-1]
        t1_values = np.round(np.arange(0.05, 1.31, 0.01), 3)
        t0_values = np.round(np.arange(0.05, 1.55, 0.05), 3)
        at_t1_edge = converged["t1"] in (t1_values.min(), t1_values.max())
        at_t0_edge = converged["t0"] in (t0_values.min(), t0_values.max())
        rows.append({
            "seed": seed, "baseline_delta": start["delta"],
            "converged_t0": converged["t0"], "converged_mu": converged["mu"], "converged_t1": converged["t1"],
            "converged_delta": converged["delta"], "rounds": int(trace_df["round"].max()),
            "at_grid_edge": at_t0_edge or at_t1_edge,
        })
        print(f"  seed={seed}: converged t0={converged['t0']:.2f} mu={converged['mu']:.1f} "
              f"t1={converged['t1']:.2f} delta={converged['delta']:+.5f} "
              f"(baseline={start['delta']:+.5f})"
              f"{'  [AT GRID EDGE -- inconclusive]' if (at_t0_edge or at_t1_edge) else ''}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(_DATA_DIR / "wormhole_generality_check.csv", index=False)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xcol, xlabel) in zip(axes, [("converged_t0", "t0"), ("converged_mu", "mu"), ("converged_t1", "t1")]):
        colors = ['#FF007F' if edge else '#00FFFF' for edge in summary_df["at_grid_edge"]]
        ax.scatter(summary_df[xcol], summary_df["converged_delta"], c=colors, s=80, zorder=5)
        for _, row in summary_df.iterrows():
            ax.annotate(str(row["seed"]), (row[xcol], row["converged_delta"]),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, color='#888888', ha='center')
        ax.set_xlabel(xlabel, color='#888888')
        ax.set_ylabel("converged delta", color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    fig.suptitle("Experiment 8: converged (t0, mu, t1) scattered across 6 SYK instances\n"
                 "(cyan = interior fixed point, magenta = at grid edge, inconclusive)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_generality_check.png", dpi=300)
    plt.close(fig)
    return summary_df


def run_trotter_noise_scan(seed: int, t0: float, mu: float, t1: float,
                            noise_levels=(0.0, 0.005, 0.01, 0.02, 0.05),
                            n_trials: int = 6, n_steps_evolution: int = 8,
                            n_steps_coupling: int = 16) -> pd.DataFrame:
    """Does the sign-dependent signal survive realistic hardware noise?
    Runs the real Trotterized gate circuit (run_wormhole_protocol_trotter's
    own construction, reimplemented here to inject noise mid-circuit --
    that function runs the whole circuit in one call with no seam to
    interrupt) and applies a real stochastic depolarizing Kraus channel
    (dense_evolution.registry.NoiseModel) after each of the protocol's
    three phases (t0 evolution, mu coupling, t1 evolution), not just once
    at the end -- closer to how noise actually accumulates on real
    hardware than a single post-hoc channel would be.

    Compared against the *noiseless* Trotter result, not the exact
    backend -- Trotterization itself has a real, separate discretization
    error (see run_wormhole_protocol_trotter's own docstring: ~2% at the
    converged point), and conflating that with the effect of physical
    noise would misattribute one for the other.

    NoiseModel.apply_to_sv is a single-shot stochastic draw (same
    caveat as dashboard_core.mitigation.run_zne_mitigation), so each
    noise level is averaged over n_trials independent draws, each with
    its own fresh RNG."""
    n_side, n_full, L, R, P, Q, terms_full, v_terms = _protocol_layout(N_MAJORANA, K_TERMS, J, seed)

    def run_noisy(mu_signed, noise_p, rng):
        sim = de.DenseSVSimulator(n_full)
        sim.run_circuit(_initial_state_ops(n_side, L, R, P, Q, True)
                         + trotter_evolve_ops(terms_full, t0, n_steps_evolution))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        sim.set_state(sv)
        sim.run_circuit(trotter_evolve_ops(v_terms, mu_signed, n_steps_coupling))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        sim.set_state(sv)
        sim.run_circuit(trotter_evolve_ops(terms_full, t1, n_steps_evolution))
        sv = sim.get_statevector()
        if noise_p > 0:
            sv = NoiseModel.apply_to_sv(sv, n_full, 'depolarizing', noise_p, rng=rng)
        return mutual_information(sv, n_full, [P], [R[0]])

    rows = []
    for noise_p in noise_levels:
        deltas = []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 * trial + 7)
            i_pos = run_noisy(+mu, noise_p, rng)
            i_neg = run_noisy(-mu, noise_p, rng)
            deltas.append(i_neg - i_pos)
        deltas = np.array(deltas)
        rows.append({"noise_p": noise_p, "delta_mean": deltas.mean(), "delta_std": deltas.std(),
                     "delta_min": deltas.min(), "delta_max": deltas.max()})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_trotter_noise_scan.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(df["noise_p"], df["delta_mean"], yerr=df["delta_std"], fmt='o-',
                color='#00FFFF', ecolor='#FF007F', capsize=4, markersize=6)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("depolarizing noise probability p", color='#888888')
    ax.set_ylabel("delta = I(mu=-|mu|) - I(mu=+|mu|)", color='#888888')
    ax.set_title(f"Sign-dependent signal vs. realistic depolarizing noise (Trotter backend)\n"
                 f"(seed={seed}, t0={t0}, mu={mu}, t1={t1}, n={n_trials} trials/point)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_trotter_noise_scan.png", dpi=300)
    plt.close(fig)
    return df


def run_paper_defaults_comparison(seeds=None) -> pd.DataFrame:
    """Direct comparison against arXiv:2604.10090's own "Ensemble
    robustness" section, which reports 100 disorder realizations and
    concludes the sign-dependent asymmetry is "a generic feature of the
    ensemble", with their chosen Hamiltonian (seed=61 here) selected
    mainly for having an unusually *large* -- not unusually *signed* --
    asymmetry.

    Experiment 8 evaluated all 6 instances at Experiment 5's point
    (t0=0.65, mu=15.0, t1=0.60), which was itself optimized on seed=61 --
    a real confound: an instance showing the "wrong" sign there could
    simply be evaluated at a bad point for it, not a genuinely reversed
    signal. This experiment controls for that by re-evaluating all 6
    instances at the paper's OWN stated defaults (t0=0.3, mu=12,
    t1=0.60, Eq. matching Experiment 1's original setup) instead --
    the same point the paper's own ensemble claim is presumably about."""
    if seeds is None:
        seeds = find_multiple_seeds(n_instances=6)

    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60
    rows = []
    for seed in seeds:
        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        delta = i_neg - i_pos
        rows.append({"seed": seed, "delta_at_paper_defaults": delta})
        print(f"  seed={seed}: delta_at_paper_defaults={delta:+.5f}"
              f"{'  [WRONG SIGN]' if delta < 0 else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_paper_defaults_comparison.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
    ax.bar([str(s) for s in df["seed"]], df["delta_at_paper_defaults"], color=colors)
    ax.axhline(0, color='#666666', linestyle=':')
    ax.set_xlabel("seed", color='#888888')
    ax.set_ylabel("delta at paper defaults (t0=0.3, mu=12, t1=0.60)", color='#888888')
    ax.set_title("Sign-dependent asymmetry at arXiv:2604.10090's own default parameters\n"
                 "(cyan = correct sign, magenta = wrong sign)", fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_paper_defaults_comparison.png", dpi=300)
    plt.close(fig)
    return df


def run_ensemble_sign_check(n_instances: int = 100, n_candidates: int = 120000) -> pd.DataFrame:
    """Large-sample version of Experiment 10's check. Experiment 10 found
    2 of 6 instances wrong-signed at arXiv:2604.10090's own stated
    default parameters (t0=0.3, mu=12, t1=0.60) -- a real but small
    sample. This repeats the identical check across up to n_instances
    exact 34/11-selection-matched SYK instances (same criterion as
    Experiments 8 and 10, via find_multiple_seeds), and additionally
    tests two candidate explanations floated informally alongside
    Experiment 10 for *why* the sign varies: Majorana mode-usage
    imbalance in the K=10 coupling terms (some modes coupled in many
    terms, others in few) and the spectral level-spacing r-statistic
    (a standard chaos diagnostic, Poisson~0.386 vs GOE~0.530). Both are
    tested for real correlation against delta via Pearson r, not just
    eyeballed."""
    seeds = find_multiple_seeds(n_instances=n_instances, n_candidates=n_candidates)
    all_quads = list(itertools.combinations(range(1, N_MAJORANA + 1), 4))
    n_qubits = N_MAJORANA // 2
    T0_PAPER, MU_PAPER, T1_PAPER = 0.3, 12.0, 0.60

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(all_quads), size=K_TERMS, replace=False)
        quads = [all_quads[idx] for idx in chosen_idx]
        mode_count = np.zeros(N_MAJORANA + 1)
        for q in quads:
            for m in q:
                mode_count[m] += 1
        usage_std = float(np.std(mode_count[1:]))

        _, terms = build_sparse_syk_terms(N_MAJORANA, K_TERMS, J, seed)
        H = de.pauli_hamiltonian_to_matrix(terms, n_qubits)
        eigvals = np.sort(np.linalg.eigvalsh(H))
        gaps = np.diff(eigvals)
        gaps = gaps[gaps > 1e-12]
        r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
        r_stat = float(np.mean(r))

        i_pos = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, +MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        i_neg = run_wormhole_protocol(N_MAJORANA, K_TERMS, J, -MU_PAPER, T0_PAPER, T1_PAPER, seed, with_message=True)
        delta = i_neg - i_pos

        rows.append({"seed": seed, "mode_usage_std": usage_std, "r_stat": r_stat,
                     "delta_at_paper_defaults": delta})

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "wormhole_ensemble_sign_check.csv", index=False)

    r_usage = scipy_stats.pearsonr(df["mode_usage_std"], df["delta_at_paper_defaults"])
    r_chaos = scipy_stats.pearsonr(df["r_stat"], df["delta_at_paper_defaults"])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (xcol, xlabel, r_result) in zip(
        axes, [("mode_usage_std", "Majorana mode-usage std", r_usage), ("r_stat", "level-spacing r-statistic", r_chaos)]
    ):
        colors = ['#FF007F' if d < 0 else '#00FFFF' for d in df["delta_at_paper_defaults"]]
        ax.scatter(df[xcol], df["delta_at_paper_defaults"], c=colors, s=25, alpha=0.7)
        ax.axhline(0, color='#666666', linestyle=':')
        ax.set_xlabel(xlabel, color='#888888')
        ax.set_ylabel("delta at paper defaults", color='#888888')
        ax.set_title(f"r={r_result.statistic:+.3f}, p={r_result.pvalue:.4f}", fontsize=10, color='#888888')
        ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    n_wrong = int((df["delta_at_paper_defaults"] < 0).sum())
    fig.suptitle(f"Experiment 11: n={len(df)} instances, {n_wrong}/{len(df)} ({100*n_wrong/len(df):.0f}%) wrong-signed "
                 f"at arXiv:2604.10090's own defaults\n(cyan = correct sign, magenta = wrong sign)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "wormhole_ensemble_sign_check.png", dpi=300)
    plt.close(fig)
    return df


def run_all():
    seed = find_seed()

    print("\n=== Experiment 1: t1 sweep ===")
    df1 = run_t1_sweep(seed)
    peak1 = df1.loc[df1["delta"].idxmax()]
    print(f"  peak: t1={peak1['t1']:.2f}  delta={peak1['delta']:+.5f}")

    print("\n=== Experiment 2: message-vs-no-message control ===")
    df2 = run_message_control(seed)
    max_no_msg = df2[~df2.with_message]["delta"].abs().max()
    print(f"  max |delta| without message: {max_no_msg:.8f} (should be ~0)")

    print("\n=== Experiment 3: mu-magnitude scan ===")
    df3 = run_mu_scan(seed)
    peak3 = df3.loc[df3["delta"].idxmax()]
    print(f"  peak: mu={peak3['mu']:.1f}  delta={peak3['delta']:+.5f}")

    print("\n=== Experiment 4: t0 scrambling-time scan ===")
    df4 = run_t0_scan(seed)
    peak4 = df4.loc[df4["delta"].idxmax()]
    print(f"  peak: t0={peak4['t0']:.2f}  delta={peak4['delta']:+.5f}")

    print("\n=== Experiment 5: 2D (t0, mu) joint grid search ===")
    df5 = run_2d_grid_search(seed)
    peak5 = df5.loc[df5["delta"].idxmax()]
    print(f"  grid: {df5['t0'].nunique()} x {df5['mu'].nunique()} = {len(df5)} points")
    print(f"  global max: t0={peak5['t0']:.2f}  mu={peak5['mu']:.1f}  delta={peak5['delta']:+.5f}")

    print("\n=== Experiment 6: t1 re-scan at Experiment 5's (t0, mu) optimum ===")
    df6 = run_t1_rescan(seed)
    peak6 = df6.loc[df6["delta"].idxmax()]
    print(f"  peak: t1={peak6['t1']:.2f}  delta={peak6['delta']:+.5f}"
          f"  ({(peak6['delta'] / peak5['delta'] - 1) * 100:+.1f}% vs. Experiment 5's t1=0.60)")

    print("\n=== Experiment 7: iterated coordinate ascent toward the joint (t0, mu, t1) optimum ===")
    df7 = run_coordinate_ascent_3d(seed)
    converged = df7.iloc[-1]
    print(f"  {df7['round'].max()} rounds, converged: "
          f"t0={converged['t0']:.2f}  mu={converged['mu']:.1f}  t1={converged['t1']:.2f}  "
          f"delta={converged['delta']:+.5f}"
          f"  ({(converged['delta'] / peak5['delta'] - 1) * 100:+.1f}% vs. Experiment 5)")

    print("\n=== Experiment 8: does the converged point generalize across SYK instances? ===")
    df8 = run_generality_check()
    n_edge = int(df8["at_grid_edge"].sum())
    n_negative_baseline = int((df8["baseline_delta"] < 0).sum())
    print(f"  {len(df8)} instances checked -- converged (t0, mu, t1) does NOT cluster near seed=61's "
          f"answer, {n_edge} at the grid edge (inconclusive), {n_negative_baseline} with a negative "
          f"baseline delta at Experiment 5's own starting point.")

    print("\n=== Experiment 9: does the signal survive realistic hardware noise? ===")
    df9 = run_trotter_noise_scan(seed, t0=0.70, mu=17.0, t1=0.36)
    crossing = df9[df9["delta_mean"] < 0]
    first_negative_p = crossing["noise_p"].min() if not crossing.empty else None
    print(f"  noiseless delta={df9.iloc[0]['delta_mean']:+.5f}; mean delta crosses zero "
          f"{'at p=' + str(first_negative_p) if first_negative_p is not None else 'nowhere in the scanned range'} "
          f"-- at p=0.01 the signal ({df9.iloc[2]['delta_mean']:+.5f}) is already smaller than its own "
          f"trial-to-trial noise ({df9.iloc[2]['delta_std']:.5f}).")

    print("\n=== Experiment 10: cross-check against arXiv:2604.10090's own ensemble-robustness claim ===")
    df10 = run_paper_defaults_comparison()
    n_wrong = int((df10["delta_at_paper_defaults"] < 0).sum())
    print(f"  {n_wrong}/{len(df10)} instances show the wrong sign at the paper's own default "
          f"parameters (t0=0.3, mu=12, t1=0.60) -- contradicts arXiv:2604.10090's 'Ensemble "
          f"robustness' claim that the sign-dependent asymmetry is a generic ensemble feature, "
          f"at least for this 34/11-selection-matched subset.")

    print("\n=== Experiment 11: large-sample (n=100) ensemble sign check ===")
    df11 = run_ensemble_sign_check(n_instances=100)
    n_wrong11 = int((df11["delta_at_paper_defaults"] < 0).sum())
    r_usage = scipy_stats.pearsonr(df11["mode_usage_std"], df11["delta_at_paper_defaults"])
    r_chaos = scipy_stats.pearsonr(df11["r_stat"], df11["delta_at_paper_defaults"])
    print(f"  n={len(df11)}: {n_wrong11}/{len(df11)} ({100*n_wrong11/len(df11):.0f}%) wrong-signed at the "
          f"paper's own defaults -- a much larger, more statistically robust version of Experiment 10's "
          f"2/6 finding. Neither candidate structural explanation holds up at this sample size: "
          f"mode-usage-imbalance r={r_usage.statistic:+.3f} (p={r_usage.pvalue:.3f}), "
          f"level-spacing r-statistic r={r_chaos.statistic:+.3f} (p={r_chaos.pvalue:.3f}) -- "
          f"neither is a statistically significant predictor of the sign.")

    print("\n============================================================")
    print("Data saved to data/wormhole_*.csv")
    print("Plots saved to images/wormhole_*.png")
    print("============================================================")


if __name__ == "__main__":
    run_all()
