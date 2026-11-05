"""
Dense-Evolution issue #358: Cr-O dimer, 3-21g, 32 electrons, R in [1.6, 2.4] A.
UHF and CUHF (ROHF) for n_unpaired in {2, 4, 6}, warm-started along the curve,
with convergence, k > 0, CUHF <S^2> = S(S+1) and neighbour-jump checks.
Needs dense-evolution from main (run_uhf / run_cuhf, PR #359).

Usage: python cr_o_open_shell_curve.py
"""
import numpy as np

from dense_evolution.native_hf.assembly import build_core_hamiltonian, build_overlap_matrix
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.libcint_bridge import build_repulsion_tensor_libcint
from dense_evolution.native_hf.scf import run_cuhf, run_uhf

# =============================================================================
# Cr-O test (issue #358)
# =============================================================================

BOHR_PER_ANG    = 1.8897259886
ATOMIC_NUMBERS  = [24, 8]
NUCLEAR_CHARGES = [24.0, 8.0]
N_ELECTRONS     = 32
R_VALUES        = np.linspace(1.6, 2.4, 15)
SPINS           = [2, 4, 6]
BASIS           = "3-21g"


def _build(R):
    geom = np.array([[0.0, 0.0, 0.0],
                     [R * BOHR_PER_ANG, 0.0, 0.0]])
    shells = build_molecule_shells(ATOMIC_NUMBERS, geom, BASIS)
    S = build_overlap_matrix(shells)
    H = build_core_hamiltonian(shells, NUCLEAR_CHARGES, geom)
    V = build_repulsion_tensor_libcint(ATOMIC_NUMBERS, geom, BASIS)
    return S, H, V, geom


def run_curve(method, n_unpaired):
    E, conv, iters, s2 = [], [], [], []
    Ca_prev = Cb_prev = None
    fn = run_uhf if method == "UHF" else run_cuhf
    for R in R_VALUES:
        S, H, V, geom = _build(R)
        res = fn(S, H, V, N_ELECTRONS, NUCLEAR_CHARGES, geom,
                 n_unpaired=n_unpaired,
                 C_alpha_init=Ca_prev, C_beta_init=Cb_prev)
        E.append(res.total_energy)
        conv.append(res.converged)
        iters.append(res.n_iterations)
        s2.append(res.spin_squared)
        Ca_prev = res.orbital_coefficients_alpha
        Cb_prev = res.orbital_coefficients_beta
    return (np.array(E), np.array(conv), np.array(iters), np.array(s2))


def main():
    print("=" * 100)
    print(" Cr-O dimer, 3-21g, 32 electrons, R in [1.6, 2.4] A, 15 points")
    print("=" * 100)
    print()

    curves = {}
    for method in ["UHF", "CUHF"]:
        for n_un in SPINS:
            print(f"  running {method}, n_unpaired = {n_un} ...")
            curves[(method, n_un)] = run_curve(method, n_un)
    print()

    # Energy table
    print(" Total energy (Hartree):")
    print()
    header = f"  {'R (A)':>6}"
    for method in ["UHF", "CUHF"]:
        for n_un in SPINS:
            header += f"  {method}-{n_un}u".rjust(15)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, R in enumerate(R_VALUES):
        row = f"  {R:>6.3f}"
        for method in ["UHF", "CUHF"]:
            for n_un in SPINS:
                row += f"  {curves[(method, n_un)][0][i]:>15.6f}"
        print(row)

    # Convergence
    print()
    print(" Convergence (T/F + iterations):")
    print()
    header = f"  {'R (A)':>6}"
    for method in ["UHF", "CUHF"]:
        for n_un in SPINS:
            header += f"  {method}-{n_un}u".rjust(15)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, R in enumerate(R_VALUES):
        row = f"  {R:>6.3f}"
        for method in ["UHF", "CUHF"]:
            for n_un in SPINS:
                _E, conv, iters, _s2 = curves[(method, n_un)]
                tag = "T" if conv[i] else "F"
                row += f"  {tag} {iters[i]:>3d}".rjust(15)
        print(row)

    # <S^2> for CUHF
    print()
    print(" CUHF <S^2> vs target S(S+1):")
    print()
    header = f"  {'R (A)':>6}"
    for n_un in SPINS:
        header += f"  {'target':>8}  {str(n_un)+'u':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, R in enumerate(R_VALUES):
        row = f"  {R:>6.3f}"
        for n_un in SPINS:
            S_ = n_un / 2
            target = S_ * (S_ + 1)
            val = curves[("CUHF", n_un)][3][i]
            row += f"  {target:>8.3f}  {val:>15.10f}"
        print(row)

    # Best state
    print()
    print("=" * 100)
    print(" Best spin state selection")
    print("=" * 100)
    print()
    summary = []
    for method in ["UHF", "CUHF"]:
        for n_un in SPINS:
            E, _conv, _iters, _s2 = curves[(method, n_un)]
            E_min = float(E.min())
            R_min = float(R_VALUES[E.argmin()])
            summary.append((method, n_un, E_min, R_min))

    print(f"  {'method':>6} {'n_unpaired':>10} {'E_min (Ha)':>18} {'R(E_min) (A)':>14}")
    print("  " + "-" * 52)
    for m, n, e, r in summary:
        print(f"  {m:>6} {n:>10} {e:>18.6f} {r:>14.4f}")

    best = min(summary, key=lambda x: x[2])
    best_method, best_nun, _, _ = best
    E_best, conv_best, iters_best, s2_best = curves[(best_method, best_nun)]

    print()
    print(f"  -> best: {best_method}, n_unpaired = {best_nun}, "
          f"E_min = {best[2]:.6f} Ha at R = {best[3]:.4f} A")

    # Detailed table for the best curve
    print()
    print("=" * 100)
    print(f" Detailed curve - {best_method}, n_unpaired = {best_nun}")
    print("=" * 100)
    print()
    print(f"  {'R (A)':>6} {'E (Ha)':>18} {'dE to prev':>16} "
          f"{'conv':>6} {'iter':>6} {'<S^2>':>16}")
    print("  " + "-" * 78)
    for i, R in enumerate(R_VALUES):
        dE = "  ---" if i == 0 else f"  {E_best[i] - E_best[i-1]:+.6e}"
        print(f"  {R:>6.3f} {E_best[i]:>18.8f} {dE:>16} "
              f"{str(bool(conv_best[i])):>6} {iters_best[i]:>6d} "
              f"{s2_best[i]:>16.10f}")

    jumps = np.abs(np.diff(E_best))
    print()
    print(f"  max |E(R_i) - E(R_(i+1))| = {jumps.max():.6f} Ha")

    # Parabola fit
    i_min = int(E_best.argmin())
    k, R_e, E_e = None, None, None
    if 2 <= i_min <= len(R_VALUES) - 3:
        idx = slice(i_min - 2, i_min + 3)
        R_fit = R_VALUES[idx]
        E_fit = E_best[idx]
        a2, a1, a0 = np.polyfit(R_fit, E_fit, 2)
        k   = 2.0 * a2
        R_e = -a1 / (2.0 * a2)
        E_e = a0 - a1 ** 2 / (4.0 * a2)
        print()
        print("  Parabola fit on the 5 points around the minimum:")
        print(f"    R_e      = {R_e:.6f} A")
        print(f"    E(R_e)   = {E_e:.8f} Ha")
        print(f"    k = 2*a2 = {k:.6f} Ha/A^2")
    else:
        print()
        print(f"  Minimum at boundary (index {i_min}); no parabola fit.")

    # Assertions
    print()
    print("=" * 100)
    print(" Assertions")
    print("=" * 100)
    print()

    failures = []

    bad_conv = [key for key, v in curves.items() if not v[1].all()]
    if bad_conv:
        failures.append(f"non-converged: {bad_conv}")
    print(f"  [1] All points converged: {'OK' if not bad_conv else 'FAIL'}")

    if k is not None:
        if k <= 0:
            failures.append(f"k = {k} <= 0")
        print(f"  [2] k > 0: {k:.6f}  {'OK' if k > 0 else 'FAIL'}")
    else:
        print("  [2] Skipped (min at boundary).")

    worst_dev = 0.0
    for n_un in SPINS:
        S_ = n_un / 2
        target = S_ * (S_ + 1)
        dev = float(np.abs(curves[("CUHF", n_un)][3] - target).max())
        worst_dev = max(worst_dev, dev)
        if dev > 1e-8:
            failures.append(f"CUHF {n_un}u: |<S^2> - S(S+1)| = {dev:.2e}")
    print(f"  [3] CUHF <S^2> = S(S+1) to 1e-8: worst = {worst_dev:.2e}  "
          f"{'OK' if worst_dev <= 1e-8 else 'FAIL'}")

    if jumps.max() < 0.05:
        print(f"  [4] Max neighbour jump < 0.05 Ha: {jumps.max():.6f}  OK")
    else:
        failures.append(f"max jump = {jumps.max():.6f}")
        print(f"  [4] Max neighbour jump < 0.05 Ha: {jumps.max():.6f}  FAIL")

    uhf_s2 = curves[("UHF", best_nun)][3]
    cuhf_s2 = curves[("CUHF", best_nun)][3]
    S_ = best_nun / 2
    target = S_ * (S_ + 1)
    uhf_dev = float(np.abs(uhf_s2 - target).max())
    cuhf_dev = float(np.abs(cuhf_s2 - target).max())
    print(f"  [5] Spin contamination (n_unpaired = {best_nun}, S(S+1) = {target}):")
    print(f"      UHF  <S^2> range [{uhf_s2.min():.6f}, {uhf_s2.max():.6f}], "
          f"max dev {uhf_dev:.2e}")
    print(f"      CUHF <S^2> range [{cuhf_s2.min():.6f}, {cuhf_s2.max():.6f}], "
          f"max dev {cuhf_dev:.2e}")

    print()
    if failures:
        print(f"  RESULT: {len(failures)} assertion(s) failed:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  RESULT: all assertions passed.")


if __name__ == "__main__":
    main()
