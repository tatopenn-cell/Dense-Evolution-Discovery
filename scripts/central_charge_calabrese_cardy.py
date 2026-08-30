"""
Central charge extraction via entanglement entropy scaling (Calabrese &
Cardy, J. Stat. Mech. 2004, P06002; see also Tong's lecture notes SS4.4.3,
"c is for Cardy") -- does the exact critical TFIM ground state's
entanglement entropy really follow the open-chain CFT prediction

    S(L) = (c/6) * ln[ (2N/pi) * sin(pi*L/N) ] + const

and does fitting it recover the known Ising CFT central charge c=1/2?

TWO INDEPENDENT VALIDATIONS before trusting any physics conclusion:

1. Computation check: the entanglement entropy computed via
   dense_evolution.partial_trace/von_neumann_entropy on the many-body
   Lanczos ground state is cross-checked against a completely independent
   method -- free-fermion (Jordan-Wigner + Bogoliubov-de Gennes) exact
   diagonalization, using Peschel's formula (J. Phys. A 36, L205, 2003)
   for the entanglement entropy of a Gaussian fermionic state via its
   Majorana covariance matrix. The Majorana-correlator algebra used here
   was self-tested against brute-force many-body ED at N=6 (diff ~1e-15)
   before trusting it at N=12 -- same discipline as
   ising_freefermion_verification.py's own XX-correlator self-test.

2. Critical-point check: a real methodological pitfall was found and
   fixed during this experiment. ising_exact_verification.py's g*=0.8600
   is the finite-size SUSCEPTIBILITY-PEAK location (the right point for
   that script's <ZZ> ansatz-tracking purpose) -- NOT the same as the
   textbook self-dual CFT critical point g=1.0 (H=-sum ZZ - g*sum X is
   Kramers-Wannier self-dual at g=1 in the thermodynamic limit). At finite
   N these two "critical" points do not coincide. Fitting the CFT formula
   at g*=0.86 gives extracted c=0.98 (~2x the theoretical 0.5, a real but
   misleading artifact of using the wrong "critical" point); at the true
   self-dual g=1.0, extracted c=0.565 -- much closer to 0.5, the residual
   ~13% gap plausibly explained by finite-size corrections at this modest
   N=12 (the CFT formula is an asymptotic large-N/L result).

Produces `data/central_charge_calabrese_cardy.csv`.

    python scripts/central_charge_calabrese_cardy.py
"""
import csv
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

import dense_evolution as de

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_CSV = _REPO_ROOT / "data" / "central_charge_calabrese_cardy.csv"

G_SELFDUAL_CFT = 1.0     # textbook infinite-system critical point (Kramers-Wannier self-dual)
G_SUSCEPTIBILITY_PEAK = 0.8600  # ising_exact_verification.py's finite-size g* (different notion of "critical")
G_OFFCRITICAL = 1.8      # negative control, deep in the paramagnetic phase
THEORY_C_ISING = 0.5

N_SELFTEST = 6
L_SELFTEST = [1, 2, 3]


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ising_exact = _import_script("ising_exact_verification")
freefermion = _import_script("ising_freefermion_verification")
N = ising_exact.N


# ---------------------------------------------------------------------------
# Free-fermion (Majorana/Peschel) entanglement entropy, self-tested below
# ---------------------------------------------------------------------------

def majorana_gamma_block(cdag_c, c_c):
    """The 'even-odd' block of the Majorana correlation matrix is the only
    nonzero piece: <a_2j a_2k> and <a_(2j+1) a_(2k+1)> vanish identically
    here because cdag_c is symmetric and c_c antisymmetric (both verified
    numerically, matching how ground_state_correlators builds them:
    cdag_c = V@V.T is manifestly symmetric, c_c = U@V.T with zero diagonal
    from fermion Pauli exclusion). Gamma[2j, 2k+1] = <a_2j a_(2k+1)> / i
    (the purely-imaginary cross-correlator with its i factored out, real)."""
    n = cdag_c.shape[0]
    Gamma = np.zeros((2 * n, 2 * n))
    for j in range(n):
        for k in range(n):
            djk = 1.0 if j == k else 0.0
            eo = (djk - cdag_c[k, j]) - c_c[j, k] + c_c[k, j] - cdag_c[j, k]
            Gamma[2 * j, 2 * k + 1] = eo
            Gamma[2 * k + 1, 2 * j] = -eo
    return Gamma


def entanglement_entropy_freefermion(A, B, L):
    cdag_c, c_c = freefermion.ground_state_correlators(A, B)
    Gamma = majorana_gamma_block(cdag_c, c_c)
    idx = list(range(2 * L))
    Gsub = Gamma[np.ix_(idx, idx)]
    evals = np.linalg.eigvalsh(1j * Gsub)
    nu = np.sort(evals[evals > 1e-9])
    S = 0.0
    for v in nu:
        v = min(v, 1 - 1e-12)
        p, q = (1 + v) / 2, (1 - v) / 2
        if q > 1e-14:
            S += -(p * np.log(p) + q * np.log(q))
    return float(S)


def _brute_force_entropy_selftest(n, J, h, L):
    """Independent brute-force many-body ED, used ONLY to self-test the
    Majorana formula above at small n before trusting it at N=12."""
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I2 = np.eye(2)

    def op_on(op, q):
        m = 1.0
        for i in range(n):
            m = np.kron(m, op if i == q else I2)
        return m

    Hm = np.zeros((2 ** n, 2 ** n))
    for i in range(n - 1):
        Hm += -J * op_on(X, i) @ op_on(X, i + 1)
    for i in range(n):
        Hm += -h * op_on(Z, i)
    vals, vecs = np.linalg.eigh(Hm)
    psi = vecs[:, 0]
    keep, trace_q = list(range(L)), [q for q in range(n) if q >= L]
    psi_t = np.transpose(psi.reshape([2] * n), keep + trace_q).reshape(2 ** L, 2 ** (n - L))
    rho = psi_t @ psi_t.conj().T
    eigs = np.clip(np.linalg.eigvalsh(rho).real, 1e-14, None)
    return float(-np.sum(eigs * np.log(eigs)))


def selftest_freefermion_formula():
    A, B = freefermion.build_AB(N_SELFTEST, 1.0, G_SUSCEPTIBILITY_PEAK)
    print(f"Self-test: free-fermion Peschel formula vs brute-force ED (N={N_SELFTEST})")
    max_diff = 0.0
    for L in L_SELFTEST:
        s_ff = entanglement_entropy_freefermion(A, B, L)
        s_bf = _brute_force_entropy_selftest(N_SELFTEST, 1.0, G_SUSCEPTIBILITY_PEAK, L)
        diff = abs(s_ff - s_bf)
        max_diff = max(max_diff, diff)
        print(f"  L={L}  free-fermion={s_ff:.6f}  brute-force={s_bf:.6f}  diff={diff:.2e}")
    assert max_diff < 1e-8, f"self-test failed: max diff {max_diff:.2e} too large, do not trust N=12 results"
    print(f"  PASSED (max diff {max_diff:.2e})\n")


# ---------------------------------------------------------------------------
# N=12 many-body ground state + entanglement entropy scaling fit
# ---------------------------------------------------------------------------

def ground_state(g, zz_sum, x_sum):
    H = -zz_sum - g * x_sum
    vals, vecs = eigsh(H, k=1, which="SA")
    return vecs[:, 0], float(vals[0])


def entanglement_entropy_curve(psi, n_qubits, l_min=2, l_max=None):
    if l_max is None:
        l_max = n_qubits - 2
    Ls = list(range(l_min, l_max + 1))
    S = [float(de.von_neumann_entropy(de.partial_trace(psi, n_qubits, list(range(L))))) for L in Ls]
    return np.array(Ls), np.array(S)


def cft_x(L, n_qubits):
    return np.log((2.0 * n_qubits / np.pi) * np.sin(np.pi * L / n_qubits))


def fit_central_charge(Ls, S, n_qubits):
    x = cft_x(Ls, n_qubits)
    slope, intercept = np.polyfit(x, S, 1)
    c = 6.0 * slope
    pred = slope * x + intercept
    ss_res, ss_tot = float(np.sum((S - pred) ** 2)), float(np.sum((S - S.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c, r2, x, pred


def main():
    selftest_freefermion_formula()

    zz_sum, x_sum = ising_exact._build_operators()
    print(f"N={N} qubits, open boundary TFIM (H = -sum ZZ - g*sum X)\n")

    rows = []
    configs = [
        ("self_dual_cft_point", G_SELFDUAL_CFT),
        ("susceptibility_peak", G_SUSCEPTIBILITY_PEAK),
        ("off_critical", G_OFFCRITICAL),
    ]
    for label, g in configs:
        psi, energy = ground_state(g, zz_sum, x_sum)
        Ls, S = entanglement_entropy_curve(psi, N)

        # cross-check against the independent free-fermion method at this g
        A, B = freefermion.build_AB(N, 1.0, g)
        S_ff = np.array([entanglement_entropy_freefermion(A, B, int(L)) for L in Ls])
        max_cross_diff = float(np.max(np.abs(S - S_ff)))

        c, r2, x, pred = fit_central_charge(Ls, S, N)
        print(f"[{label}] g={g}  E0={energy:.6f}  many-body-vs-freefermion max diff={max_cross_diff:.2e}")
        for L, s_val in zip(Ls, S):
            rows.append(dict(label=label, g=g, L=int(L), S=float(s_val)))
        note = (f"vs theory c={THEORY_C_ISING}  |delta|={abs(c - THEORY_C_ISING):.4f}"
                if label != "off_critical" else "(negative control -- CFT log-scaling should NOT hold)")
        print(f"  --> extracted c = {c:.4f}  (R^2 = {r2:.6f})  {note}\n")
        rows.append(dict(label=f"{label}_fit_summary", g=g, L=-1, S=c))

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()