"""
Tests for scripts/central_charge_calabrese_cardy.py -- imports the real
script and re-runs the free-fermion self-test plus a reduced-scope check
that the many-body and free-fermion entropy methods agree, at small N for
CI speed (the full N=12 sweep runs fine but this keeps CI fast).
"""
import importlib.util
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _import_script("central_charge_calabrese_cardy")


def test_freefermion_selftest_passes():
    """The script's own self-test: Majorana/Peschel formula must match
    brute-force many-body ED at N=6 to high precision."""
    mod.selftest_freefermion_formula()


def test_manybody_matches_freefermion_at_n8():
    """Independent cross-check at a size not covered by the module's own
    self-test: many-body partial_trace/von_neumann_entropy vs free-fermion
    Peschel formula must agree closely at N=8, g=1.0."""
    import dense_evolution as de
    from scipy.sparse.linalg import eigsh

    n = 8
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I2 = np.eye(2)

    def op_on(op, q):
        m = 1.0
        for i in range(n):
            m = np.kron(m, op if i == q else I2)
        return m

    zz_sum = sum(op_on(Z, i) @ op_on(Z, i + 1) for i in range(n - 1))
    x_sum = sum(op_on(X, i) for i in range(n))
    g = 1.0
    Hm = -zz_sum - g * x_sum
    vals, vecs = np.linalg.eigh(Hm)
    psi = vecs[:, 0]

    A, B = mod.freefermion.build_AB(n, 1.0, g)
    for L in [2, 3, 4]:
        rho = de.partial_trace(psi, n, list(range(L)))
        s_mb = de.von_neumann_entropy(rho)
        s_ff = mod.entanglement_entropy_freefermion(A, B, L)
        assert abs(s_mb - s_ff) < 1e-8, f"L={L}: many-body={s_mb} vs free-fermion={s_ff}"


def test_cft_fit_at_self_dual_point_is_close_to_theory():
    """Regression test for the headline finding: fitting at g=1.0 (the
    true self-dual CFT point, not the finite-size susceptibility peak)
    must give a central charge reasonably close to the theoretical 0.5."""
    import dense_evolution as de
    from scipy.sparse.linalg import eigsh

    zz_sum, x_sum = mod.ising_exact._build_operators()
    psi, _ = mod.ground_state(mod.G_SELFDUAL_CFT, zz_sum, x_sum)
    Ls, S = mod.entanglement_entropy_curve(psi, mod.N)
    c, r2, _, _ = mod.fit_central_charge(Ls, S, mod.N)
    assert r2 > 0.99, f"fit quality too low: R^2={r2}"
    assert abs(c - mod.THEORY_C_ISING) < 0.15, (
        f"extracted c={c:.4f} too far from theory {mod.THEORY_C_ISING} -- "
        "if this regresses, check g*/self-dual-point identification"
    )