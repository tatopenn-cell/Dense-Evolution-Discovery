"""
Independent verification: does scan_ising.py's fixed (non-optimized) ansatz
actually track the real 1D Transverse Field Ising Model ground-state physics?

scan_ising.py's ansatz uses a HARDCODED RZ angle (1.2) never derived from any
energy minimization -- it is not a "true variational ansatz" in the VQE
sense. This script computes the exact TFIM ground state independently via
scipy sparse Lanczos diagonalization (no dense_evolution/PennyLane circuit
involved, to keep this a genuinely independent cross-check) and compares
against the ansatz's own <ZZ> curve, reusing scan_ising.py's real functions
rather than re-deriving the circuit by hand.

H = -sum_{i=0}^{10} Z_i Z_{i+1} - g * sum_{i=0}^{11} X_i, open boundary,
N=12 qubits, matching the ansatz's own 11-bond structure exactly.
"""

import importlib.util
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_IMAGES_DIR = _REPO_ROOT / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan_ising = _import_script("scan_ising")
N = scan_ising.N_Q  # 12, matches the ansatz exactly

# ---------------------------------------------------------------------------
# Exact TFIM Hamiltonian, sparse, open boundary conditions (11 bonds).
# ---------------------------------------------------------------------------

_I2 = sp.identity(2, format="csr")
_X = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
_Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def _op_on_qubit(op, q, n=N):
    result = None
    for i in range(n):
        m = op if i == q else _I2
        result = m if result is None else sp.kron(result, m, format="csr")
    return result


def _build_operators():
    z_list = [_op_on_qubit(_Z, q) for q in range(N)]
    x_list = [_op_on_qubit(_X, q) for q in range(N)]
    zz_sum = sum(z_list[i] @ z_list[i + 1] for i in range(N - 1))
    x_sum = sum(x_list)
    return zz_sum, x_sum


def ground_state_zz(g, zz_sum, x_sum):
    H = -zz_sum - g * x_sum
    vals, vecs = eigsh(H, k=1, which="SA")
    psi = vecs[:, 0]
    zz_expect = (psi.conj() @ (zz_sum @ psi)).real / (N - 1)
    return float(zz_expect), float(vals[0])


def main():
    zz_sum, x_sum = _build_operators()

    # 501 points over [0, 2.5] (step ~0.005): the exact ground state is a
    # smooth, noise-free function of g (unlike sampled/shot data), so this
    # resolution is far finer than needed to locate a susceptibility peak
    # precisely, while keeping the ~500-point sparse Lanczos sweep well
    # under a minute. No need to match scan_ising.py's 3500 points, whose
    # resolution exists to smooth shot/sampling artifacts that don't apply
    # here.
    n_points = 501
    g_grid = np.linspace(0.0, 2.5, n_points)

    exact_zz = np.empty(n_points)
    ansatz_zz = np.empty(n_points)

    t0 = time.perf_counter()
    for idx, g in enumerate(g_grid):
        exact_zz[idx], _ = ground_state_zz(g, zz_sum, x_sum)
        prob = scan_ising.esegui_circuito_ising_reale(g)
        ansatz_zz[idx] = scan_ising.calcola_vera_correlazione_zz(prob)
        if (idx + 1) % 100 == 0 or idx == 0 or idx == n_points - 1:
            print(f"g={g:.4f}  exact<ZZ>={exact_zz[idx]:+.6f}  ansatz<ZZ>={ansatz_zz[idx]:+.6f}")
    elapsed = time.perf_counter() - t0
    print(f"Sweep done in {elapsed:.2f} s ({n_points} points)")

    exact_susc = -np.gradient(exact_zz, g_grid)
    ansatz_susc = -np.gradient(ansatz_zz, g_grid)

    g_critico_exact = g_grid[np.argmax(exact_susc)]
    g_critico_ansatz = g_grid[np.argmax(ansatz_susc)]

    abs_diff = np.abs(exact_zz - ansatz_zz)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    corr = float(np.corrcoef(exact_zz, ansatz_zz)[0, 1])

    df = pd.DataFrame({
        "Campo_g": g_grid,
        "Exact_ZZ": exact_zz,
        "Ansatz_ZZ": ansatz_zz,
        "Exact_Susceptibility": exact_susc,
        "Ansatz_Susceptibility": ansatz_susc,
    })
    csv_path = _DATA_DIR / "ising_exact_verification.csv"
    df.to_csv(csv_path, index=False)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(g_grid, exact_zz, color="#00FF7F", linewidth=2.5, label="Exact ground-state <ZZ> (Lanczos)")
    ax1.plot(g_grid, ansatz_zz, color="#FF007F", linewidth=1.8, linestyle="--", label="scan_ising.py fixed-ansatz <ZZ>")
    ax1.set_ylabel("Spin-Spin Correlation <ZZ>", color="#888888")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax1.legend(loc="upper right")
    ax1.set_title("TFIM (N=12, open BC): exact ground state vs. scan_ising.py's fixed ansatz",
                  fontsize=11, fontweight="bold", pad=15)

    ax2.plot(g_grid, exact_susc, color="#00FFFF", linewidth=2, label="Exact susceptibility (-d<ZZ>/dg)")
    ax2.plot(g_grid, ansatz_susc, color="#FFA500", linewidth=1.6, linestyle="--", label="Ansatz susceptibility")
    ax2.axvline(g_critico_exact, color="#00FF7F", linestyle=":", alpha=0.9, label=f"Exact critical g = {g_critico_exact:.3f}")
    ax2.axvline(g_critico_ansatz, color="#FF007F", linestyle=":", alpha=0.9, label=f"Ansatz critical g = {g_critico_ansatz:.3f}")
    ax2.set_xlabel("Transverse Field Strength (g)", color="#888888")
    ax2.set_ylabel("Susceptibility", color="#888888")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    png_path = _IMAGES_DIR / "ising_exact_verification.png"
    plt.savefig(png_path, dpi=300)

    print("=" * 60)
    print(f"Exact ground-state critical point:   g = {g_critico_exact:.4f}")
    print(f"Ansatz (scan_ising.py) critical point: g = {g_critico_ansatz:.4f}")
    print(f"scan_ising.py's claimed g = 1.309: deviation from exact = {abs(1.309 - g_critico_exact):.4f}")
    print(f"Pointwise <ZZ> difference -- max abs: {max_abs_diff:.4f}, mean abs: {mean_abs_diff:.4f}")
    print(f"Pearson correlation of exact vs ansatz <ZZ> curves: {corr:.4f}")
    print("=" * 60)
    print(f"CSV saved to {csv_path}")
    print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()
