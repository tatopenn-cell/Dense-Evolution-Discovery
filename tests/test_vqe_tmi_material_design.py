"""
Tests for scripts/vqe_tmi_material_design.py -- imports the real script
(guarded behind `if __name__ == "__main__":`, same pattern as
test_integration_smoke.py) and calls its real functions directly, at
reduced U-grid/restarts/epochs for CI speed.
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


vqe_tmi = _import_script("vqe_tmi_material_design")


def test_hamiltonian_is_hermitian():
    """build_tmi_hamiltonian must be Hermitian by construction for any U
    -- not assumed, checked directly against its own conjugate transpose."""
    for u in (0.0, 0.55, 3.27, 6.0):
        H = vqe_tmi.build_tmi_hamiltonian(u)
        assert np.allclose(H, H.conj().T, atol=1e-12), f"H(U={u}) is not Hermitian"


def test_hamiltonian_is_diagonal_at_u_zero_off_site_term():
    """At U=0 the on-site Mott repulsion term must vanish identically --
    every diagonal entry is exactly zero (only the hopping terms survive)."""
    H = vqe_tmi.build_tmi_hamiltonian(0.0)
    assert np.allclose(np.diag(H), 0.0, atol=1e-12)


def test_exact_ground_energy_is_a_true_lower_bound_for_the_hamiltonian():
    """Independent sanity check on exact_ground_energy itself: it must
    equal the smallest eigenvalue of the SAME Hamiltonian matrix
    build_tmi_hamiltonian returns, computed here with a fresh eigvalsh
    call rather than reusing exact_ground_energy's own internals."""
    for u in (0.0, 1.64, 4.91):
        H = vqe_tmi.build_tmi_hamiltonian(u)
        reference = float(np.linalg.eigvalsh(H)[0])
        assert abs(vqe_tmi.exact_ground_energy(u) - reference) < 1e-10


def test_vqe_respects_the_variational_principle():
    """Hard correctness gate: for every U tested, the VQE-optimized
    energy must never fall below the exact ground-state energy from
    direct diagonalization -- a violation would mean a bug in the
    Hamiltonian, the energy function, or the diagonalization, not a
    'lucky' optimizer."""
    result = vqe_tmi.run_experiment(
        u_range=np.array([0.0, 1.0, 3.0, 6.0]), n_starts=4, n_epochs=150, seed=1,
    )
    gap = result["E_vqe_optimized"] - result["E_exact_ground"]
    assert np.all(gap > -1e-6), (
        f"variational principle violated at U={result['U'][np.argmin(gap)]}: "
        f"gap={gap.min():.6f}"
    )


def test_vqe_at_u_zero_reaches_the_exact_ground_state():
    """At U=0 this ansatz is expressive enough to hit the true ground
    state essentially exactly -- the easiest, most unambiguous case, and
    a real regression check on the optimizer itself (not just its bound)."""
    result = vqe_tmi.run_experiment(
        u_range=np.array([0.0]), n_starts=4, n_epochs=150, seed=1,
    )
    gap = result["E_vqe_optimized"][0] - result["E_exact_ground"][0]
    assert abs(gap) < 1e-3, f"expected near-exact convergence at U=0, got gap={gap:.6f}"


def test_optimization_loop_actually_moves_the_energy():
    """The optimized energy must be far below the unoptimized (single
    random theta) baseline."""
    result = vqe_tmi.run_experiment(
        u_range=np.array([1.0, 3.0]), n_starts=4, n_epochs=150, seed=1,
    )
    improvement = result["E_random_baseline"] - result["E_vqe_optimized"]
    assert np.all(improvement > 1.0), (
        f"optimization should substantially lower the energy vs. an unoptimized "
        f"random theta, got improvements {improvement}"
    )


def test_real_gaas_point_respects_the_variational_principle():
    """Same hard correctness gate as the U sweep, applied to the two
    physically-grounded GaAs points (DFT t1, bare and dielectrically-
    screened U) instead of the arbitrary-unit sweep."""
    result = vqe_tmi.run_real_gaas_point(n_starts=4, n_epochs=150, seed=1)
    gap = result["E_vqe_optimized"] - result["E_exact_ground"]
    assert np.all(gap > -1e-6), (
        f"variational principle violated at the real GaAs point(s): gap={gap.min():.6f}"
    )


def test_gaas_screened_u_is_smaller_than_bare_u_by_the_dielectric_constant():
    """The screened U is a direct division of the bare (unscreened) DFT
    integral by GaAs's real static dielectric constant -- not a separate
    fitted number -- so this ratio must hold exactly, not approximately."""
    assert abs(vqe_tmi.U_GAAS_SCREENED_EV * vqe_tmi.GAAS_EPSILON_R - vqe_tmi.U_GAAS_BARE_EV) < 1e-9


def test_gaas_screened_point_is_in_the_weakly_correlated_regime():
    """The whole physical point of screening: dividing by GaAs's real
    dielectric constant should land U well below t1 (U/t1 << 1), matching
    GaAs being a conventional band semiconductor rather than a Mott
    insulator -- unlike the bare, unscreened value, which does not."""
    assert vqe_tmi.U_GAAS_SCREENED_EV / vqe_tmi.T1_GAAS_DFT_EV < 1.0
    assert vqe_tmi.U_GAAS_BARE_EV / vqe_tmi.T1_GAAS_DFT_EV > 1.0
