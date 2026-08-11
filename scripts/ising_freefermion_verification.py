"""
Independent free-fermion (Jordan-Wigner + Bogoliubov-de Gennes) exact
cross-check of the TFIM critical point found by many-body Lanczos
diagonalization in ising_exact_verification.py (g* = 0.860).

This is a genuinely different algorithm: instead of diagonalizing the full
2^N-dim many-body Hilbert space, it diagonalizes a single-particle 2N x 2N
BdG matrix (N=12 -> 24x24), the same method behind Pfeuty's 1970 exact
solution of the open TFIM chain.

Convention used elsewhere in this repo: H = -sum_{i} Z_i Z_{i+1} - g sum_i X_i.
The textbook free-fermion solution (Pfeuty, Lieb-Schultz-Mattis, Sachdev)
is normally stated for H' = -J sum_i X_i X_{i+1} - h sum_i Z_i. The two are
related by a single-qubit Hadamard on every site (X <-> Z, an exact unitary
relabeling), so <Z_i Z_{i+1}>_H = <X_i X_{i+1}>_{H'} with J=1, h=g. All the
free-fermion machinery below is built for H' and the textbook X-X/Z
convention, then reinterpreted via that identity to compare with the
Z-Z/X many-body result.

Jordan-Wigner mapping (standard, verified against exact many-body
diagonalization at small N in `_selftest_against_small_ED` before trusting
it at N=12):

    H' = -J sum_{j=1}^{N-1} (c_j^dag - c_j)(c_{j+1}^dag + c_{j+1})
         - h sum_{j=1}^{N} (2 c_j^dag c_j - 1)

    sigma^x_j sigma^x_{j+1} = (c_j^dag - c_j)(c_{j+1}^dag + c_{j+1})

BdG matrix M = [[A, B], [-B, -A]] (A_jj = -2h, A_{j,j+1}=A_{j+1,j}=-J;
B_{j,j+1}=-J, B_{j+1,j}=+J), diagonalized densely; ground-state fermionic
correlators <c_i^dag c_j> and <c_i c_j> read off the negative-eigenvalue
eigenvectors, then combined into <XX> via Wick's theorem.
"""

import pathlib
import time

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Free-fermion (BdG) machinery for H' = -J sum X_i X_{i+1} - h sum Z_i
# ---------------------------------------------------------------------------

def build_AB(n, J, h):
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for j in range(n):
        A[j, j] = -2.0 * h
    for j in range(n - 1):
        A[j, j + 1] = -J
        A[j + 1, j] = -J
        B[j, j + 1] = -J
        B[j + 1, j] = J
    return A, B


def bdg_solve(A, B):
    """Diagonalize M = [[A,B],[-B,-A]] and return sorted eigenvalues/vectors."""
    n = A.shape[0]
    M = np.block([[A, B], [-B, -A]])
    assert np.allclose(M, M.T), "BdG matrix should be real symmetric"
    evals, evecs = np.linalg.eigh(M)
    return evals, evecs, n


def ground_state_correlators(A, B):
    """<c_i^dag c_j> and <c_i c_j> in the BdG ground state (Wick-ready).

    Derivation: writing Psi=(c,c^dag), M=[[A,B],[-B,-A]], H'=(1/2)Psi^dag M Psi
    + const. M's eigenvectors pair up under the particle-hole flip tau=[[0,I],[I,0]]
    (tau M tau = -M): a positive-eigenvalue eigenvector w_k=(u_k,v_k) (M w_k=eps_k w_k,
    eps_k>0) has a partner (v_k,u_k) at -eps_k. Defining eta_k = u_k.c + v_k.c^dag
    (the k-th Bogoliubov quasiparticle, annihilated by the ground state) and
    inverting gives c_i = sum_k u_k[i] eta_k + v_k[i] eta_k^dag, from which
    Wick contraction against eta_k|0>=0 yields:
        <c_i^dag c_j> = sum_k v_k[i] v_k[j]  = (V V^T)_{ij}
        <c_i c_j>     = sum_k u_k[i] v_k[j]  = (U V^T)_{ij}
    with U,V built from the POSITIVE-eigenvalue eigenvectors only.
    Verified against brute-force many-body ED at small N below.
    """
    evals, evecs, n = bdg_solve(A, B)
    pos = evals > 0
    W = evecs[:, pos]  # shape (2n, n_occ) -- generically n_occ == n
    U = W[:n, :]
    V = W[n:, :]
    cdag_c = V @ V.T   # <c_i^dag c_j> = sum_pos V_i V_j
    c_c = U @ V.T      # <c_i c_j>     = sum_pos U_i V_j
    return cdag_c, c_c


def xx_correlator(n, J, h):
    """<sigma^x_j sigma^x_{j+1}> for all bonds via Wick's theorem.

    sigma^x_j sigma^x_{j+1} = (c_j^dag - c_j)(c_{j+1}^dag + c_{j+1})
        = c_j^dag c_{j+1}^dag + c_j^dag c_{j+1} + c_{j+1}^dag c_j - c_j c_{j+1}
    With <c_i^dag c_j> = cdag_c[i,j] and <c_i c_j> = c_c[i,j] (see
    ground_state_correlators), <c_j^dag c_{j+1}^dag> = c_c[j+1, j] (derived
    the same way as c_c, verified numerically against small-N ED below).
    """
    A, B = build_AB(n, J, h)
    cdag_c, c_c = ground_state_correlators(A, B)
    xx = np.empty(n - 1)
    for j in range(n - 1):
        term_pp = c_c[j + 1, j]        # <c_j^dag c_{j+1}^dag>
        term_ph1 = cdag_c[j, j + 1]    # <c_j^dag c_{j+1}>
        term_ph2 = cdag_c[j + 1, j]    # <c_{j+1}^dag c_j>
        term_mm = c_c[j, j + 1]        # <c_j c_{j+1}>
        xx[j] = float(term_pp + term_ph1 + term_ph2 - term_mm)
    return xx


def excitation_spectrum(n, J, h):
    evals, _, _ = bdg_solve(*build_AB(n, J, h))
    return np.sort(evals[evals >= 0])


# ---------------------------------------------------------------------------
# Small-N brute-force many-body ED for H' (sanity/self-test only)
# ---------------------------------------------------------------------------

def _small_ED_xx(n, J, h):
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I2 = np.eye(2)

    def op_on(op, q):
        m = 1.0
        for i in range(n):
            m = np.kron(m, op if i == q else I2)
        return m

    xs = [op_on(X, q) for q in range(n)]
    zs = [op_on(Z, q) for q in range(n)]
    H = np.zeros((2 ** n, 2 ** n))
    for j in range(n - 1):
        H += -J * xs[j] @ xs[j + 1]
    for j in range(n):
        H += -h * zs[j]
    vals, vecs = np.linalg.eigh(H)
    psi = vecs[:, 0]
    xx = np.array([float(psi.conj() @ (xs[j] @ xs[j + 1]) @ psi) for j in range(n - 1)])
    return xx, vals[0]


def _selftest_against_small_ED():
    print("Self-test: free-fermion pipeline vs brute-force many-body ED (small N)")
    rng = np.random.default_rng(0)
    max_err = 0.0
    for n in (2, 3, 4, 5):
        for _ in range(3):
            J = 1.0
            h = float(rng.uniform(0.0, 2.0))
            xx_ff = xx_correlator(n, J, h)
            xx_ed, _ = _small_ED_xx(n, J, h)
            err = float(np.max(np.abs(xx_ff - xx_ed)))
            max_err = max(max_err, err)
            print(f"  N={n} h={h:.4f}  max|free-fermion - ED| = {err:.3e}")
    print(f"Overall max abs error across self-test: {max_err:.3e}")
    assert max_err < 1e-8, "Free-fermion pipeline disagrees with brute-force ED -- bug in JW/BdG code"
    print("Self-test PASSED: free-fermion <XX> matches many-body ED to machine precision.\n")


# ---------------------------------------------------------------------------
# Main sweep at N=12, comparing against the many-body g*=0.860 result
# ---------------------------------------------------------------------------

def main():
    _selftest_against_small_ED()

    N = 12  # matches scan_ising.py / ising_exact_verification.py exactly
    n_points = 501
    g_grid = np.linspace(0.0, 2.5, n_points)

    zz_mean = np.empty(n_points)   # <Z_i Z_{i+1}>_H = <X_i X_{i+1}>_H', averaged over bonds
    eps1 = np.empty(n_points)      # lowest single-particle energy
    eps2 = np.empty(n_points)      # second-lowest single-particle energy

    t0 = time.perf_counter()
    for idx, g in enumerate(g_grid):
        xx = xx_correlator(N, 1.0, g)
        zz_mean[idx] = float(np.mean(xx))
        spec = excitation_spectrum(N, 1.0, g)
        eps1[idx] = float(spec[0])
        eps2[idx] = float(spec[1])
        if (idx + 1) % 100 == 0 or idx == 0 or idx == n_points - 1:
            print(f"g={g:.4f}  <ZZ>(free-fermion)={zz_mean[idx]:+.6f}  eps1={eps1[idx]:.6f}  eps2={eps2[idx]:.6f}")
    elapsed = time.perf_counter() - t0
    print(f"N=12 sweep done in {elapsed:.2f} s ({n_points} points)")

    susc = -np.gradient(zz_mean, g_grid)
    g_star_susc = float(g_grid[np.argmax(susc)])

    # Fermion-gap-based indicator. eps1 (the naive lowest single-particle
    # energy) is dominated by the open chain's exponentially-localized
    # Majorana edge zero mode: it stays ~0 throughout the whole ordered
    # phase (g<~1), not just at criticality, so its raw minimum (near g=0)
    # is a finite-size boundary artifact, not a critical-point indicator.
    # eps2 (second-lowest single-particle energy) is the lowest *bulk*
    # excitation once that trivial edge mode is excluded, and its minimum
    # is the textbook free-fermion pseudo-gap indicator of the QPT.
    mask = g_grid > 0.3  # exclude the g~0 region where eps1/eps2 are both
    # numerically degenerate near zero (near-decoupled edge Majoranas)
    g_star_gap1_min = float(g_grid[np.argmin(eps1)])
    g_star_gap2_min = float(g_grid[mask][np.argmin(eps2[mask])])

    df = pd.DataFrame({
        "Campo_g": g_grid,
        "ZZ_freefermion": zz_mean,
        "Susceptibility_freefermion": susc,
        "Eps1_lowest_singleparticle": eps1,
        "Eps2_bulk_gap_proxy": eps2,
    })
    csv_path = _DATA_DIR / "ising_freefermion_verification.csv"
    df.to_csv(csv_path, index=False)

    # Spot-check against the many-body ED curve at a handful of g values.
    many_body_csv = _DATA_DIR / "ising_exact_verification.csv"
    print("=" * 70)
    if many_body_csv.exists():
        mb = pd.read_csv(many_body_csv)
        spot_g = [0.2, 0.5, 0.86, 1.0, 1.309, 1.6, 2.0]
        print("Pointwise <ZZ> spot-check vs many-body Lanczos (ising_exact_verification.csv):")
        max_spot_err = 0.0
        for gv in spot_g:
            i_ff = int(np.argmin(np.abs(g_grid - gv)))
            i_mb = int(np.argmin(np.abs(mb["Campo_g"].values - gv)))
            ff_val = zz_mean[i_ff]
            mb_val = float(mb["Exact_ZZ"].values[i_mb])
            diff = abs(ff_val - mb_val)
            max_spot_err = max(max_spot_err, diff)
            print(f"  g={gv:.3f}: free-fermion <ZZ>={ff_val:+.6f}  many-body <ZZ>={mb_val:+.6f}  diff={diff:.2e}")
        print(f"Max pointwise spot-check diff: {max_spot_err:.2e}")
    else:
        print("many-body ising_exact_verification.csv not found -- run ising_exact_verification.py first for the spot-check.")
    print("=" * 70)

    print(f"Free-fermion susceptibility-peak critical point: g* = {g_star_susc:.4f}")
    print(f"Many-body Lanczos critical point (prior result):  g* = 0.8600")
    print(f"Deviation: {abs(g_star_susc - 0.860):.4f}")
    print(f"Naive lowest-mode (eps1) gap minimum: g* = {g_star_gap1_min:.4f} -- artifact of the open chain's")
    print("  exponentially-localized Majorana edge zero mode (stays ~0 throughout the whole ordered")
    print("  phase g<~1), NOT a real critical-point indicator; shown for completeness only.")
    print(f"Bulk-gap proxy (eps2, second-lowest mode) minimum: g* = {g_star_gap2_min:.4f} -- the physically")
    print("  meaningful secondary indicator, once the trivial edge mode is excluded.")
    print(f"scan_ising.py's original claim g=1.309: deviation from free-fermion g* = {abs(1.309 - g_star_susc):.4f}")
    print("=" * 70)
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
