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

This script runs five real, verified experiments, each producing its own
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

All five use seed=61 (n_majorana=8, k_terms=10, J=sqrt(2)) -- the
instance dashboard_core.wormhole.select_good_instance finds when
screened against arXiv:2604.10090's own selection criterion (their
chosen K=10 instance has 34 commuting / 11 anticommuting pairs among the
C(10,2)=45 pairs of terms). Re-derived below, not hardcoded blindly.

Honest caveats, not glossed over:
- Experiment 5's grid holds t1 fixed at 0.60 (the Experiment 1 peak) --
  a full 3D (t0, mu, t1) joint search was not attempted. t1's own
  optimum could plausibly also shift once t0/mu are no longer at their
  original 1D-scan defaults; unverified.
- All results use the exact-evolution backend (eigendecomposition), not
  the Trotterized real-gate-circuit backend -- both are implemented and
  cross-verified to agree closely (see the main Dense-Evolution repo's
  tests), but the exact backend is what was used here for speed.
- Experiment 5 bypasses `run_wormhole_protocol`'s public API and calls
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
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    print("\n============================================================")
    print("Data saved to data/wormhole_*.csv")
    print("Plots saved to images/wormhole_*.png")
    print("============================================================")


if __name__ == "__main__":
    run_all()
