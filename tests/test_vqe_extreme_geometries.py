"""
vqe_extreme_geometries.py -- irregular/extreme molecular-geometry benchmark tests
-------------------------------------------------------------------------------
6 tests, target < 100s total. The two Adam optimizations (per-bond adaptive and
best-achievable shared theta) over all 6 scenarios are run ONCE at module import
(n_epochs=60) and reused by every test below, instead of repeating the ~85s
computation per test.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eg = _import_script("vqe_extreme_geometries")
real = _import_script("vqe_silicon_molecular")

_SCENARI = eg.build_extreme_geometries()
_NOMI = list(_SCENARI.keys())
_IDX = {n: i for i, n in enumerate(_NOMI)}
_R_MATRIX = np.stack([_SCENARI[n] for n in _NOMI])

_THETA_RIGID = np.full((len(_NOMI), eg.N_BONDS), eg.RIGID_THETA)
_E_RIGID = eg.energy_from_theta(_R_MATRIX, _THETA_RIGID)

_THETA_STAR, _E_STAR = eg.optimize_theta_per_geometry(_R_MATRIX, n_epochs=60, verbose=False)
_KINETIC_PERBOND, _ = eg.batched_per_bond_kinetic_and_jacobian(_THETA_STAR)

_THETA_SHARED_STAR, _KINETIC_SHARED = eg.optimize_shared_theta_per_geometry(_R_MATRIX, n_epochs=60)
_DEFICIT = eg.deficit_fraction(_R_MATRIX, _KINETIC_PERBOND, _KINETIC_SHARED)


def test_energy_matches_reference_scalar_formula_for_uniform_geometry():
    """energy_from_theta must reduce EXACTLY to the original single-R,
    single-theta calcola_energia_molecolare when every bond shares the same
    distance and angle -- the generalized per-bond-distance model must not
    silently change the physics of the case every other script in this repo
    already tests."""
    for R_test, theta_test in [(2.35, 0.38), (1.6, 0.62), (3.9, 1.1)]:
        R_row = np.full((1, eg.N_BONDS), R_test)
        theta_row = np.full((1, eg.N_BONDS), theta_test)
        e_generalized = eg.energy_from_theta(R_row, theta_row)[0]
        e_reference = real.calcola_energia_molecolare(R_test, theta_test)
        assert e_generalized == pytest.approx(e_reference, abs=1e-9), (
            f"R={R_test}, theta={theta_test}: generalized {e_generalized} != reference {e_reference}"
        )


def test_per_bond_jacobian_matches_finite_difference():
    """The exact per-bond Jacobian (including OFF-DIAGONAL entries d(k_q)/
    d(theta_r) for q != r -- the new part this script adds beyond the
    already-tested summed gradient in vqe_silicon_molecular_optimized_per_bond.py)
    must agree with an independent finite-difference reference computed on a
    single fresh circuit."""
    theta_vec = np.array([0.15, 0.85, 0.4, 1.3, 0.6])
    kinetic, jac = eg.batched_per_bond_kinetic_and_jacobian(theta_vec[None, :])

    def _kinetic_per_bond_single(tv):
        s = eg.de.DenseSVSimulator(n_qubits=eg.N_Q, use_float32=False)
        ops = [['x', 0]]
        for q in range(eg.N_BONDS):
            ops += [['cx', q + 1, q], ['ry', q + 1, float(tv[q])], ['cx', q, q + 1],
                    ['ry', q + 1, float(-tv[q])], ['cx', q + 1, q]]
        s.set_initial_state()
        s.run_circuit_jit_beast_mode(ops)
        return eg._kinetic_per_bond_from_sv(np.asarray(s.get_statevector()))

    kinetic_ref = _kinetic_per_bond_single(theta_vec)
    assert np.max(np.abs(kinetic[0] - kinetic_ref)) < 1e-9

    h = 1e-6
    for r in range(eg.N_BONDS):
        tp = theta_vec.copy(); tp[r] += h
        tm = theta_vec.copy(); tm[r] -= h
        fd = (_kinetic_per_bond_single(tp) - _kinetic_per_bond_single(tm)) / (2 * h)
        assert np.max(np.abs(jac[0, :, r] - fd)) < 1e-5, (
            f"bond param {r}: exact jacobian column {jac[0, :, r]} != finite-diff {fd}"
        )


def test_adaptive_per_bond_never_worse_than_rigid_fixed_theta():
    """Per-bond adaptive optimization (10 free parameters) must reach an
    energy at or below the fixed shared theta=0.38 baseline at every
    scenario -- more free parameters can only help, never hurt (Adam even
    starts from that same fixed value)."""
    assert np.all(_E_STAR <= _E_RIGID + 1e-3), (
        f"E_star={_E_STAR}, E_rigid={_E_RIGID}"
    )


def test_per_bond_never_worse_than_best_achievable_shared_theta():
    """The best possible SINGLE shared theta (re-optimized per geometry) is
    a special case of the per-bond parameter family (all bonds set equal) --
    so per-bond adaptive optimization must reach a weighted kinetic energy
    at least as large, i.e. deficit_fraction >= 0 up to optimizer tolerance."""
    assert np.all(_DEFICIT > -0.01), f"deficit_fraction should be >= 0: {dict(zip(_NOMI, _DEFICIT))}"


def test_uniform_geometries_share_identical_deficit_fraction():
    """For any UNIFORM-R geometry, the per-bond-vs-shared deficit is a pure
    ansatz-expressivity cost that cancels the overall t(R) energy scale (the
    argmax of a positively-scaled objective doesn't move with the scale) --
    it must come out IDENTICAL for equilibrium, compressed, and dissociated
    geometries, regardless of how different their raw energies are."""
    uniform_names = ["uniforme_equilibrio", "uniforme_compressa", "uniforme_dissociata"]
    values = [_DEFICIT[_IDX[n]] for n in uniform_names]
    assert max(values) - min(values) < 5e-3, (
        f"uniform-geometry deficit_fraction should be scale-invariant: {dict(zip(uniform_names, values))}"
    )


def test_double_ended_mutation_is_the_hardest_case_for_the_rigid_approximation():
    """Empirically (see the top-of-file comment in vqe_extreme_geometries.py)
    a single shared angle is NOT simply 'worse under any distortion': a
    single localized mutation or an alternating distortion pattern sit BELOW
    the uniform baseline deficit, while two mutated bonds at opposite ends
    of the chain (mutazioni_congiunte_estremi) is the standout case where
    the rigid ansatz is most penalized. This locks in that finding as a
    regression test rather than letting it silently drift."""
    idx_worst = int(np.argmax(_DEFICIT))
    assert _NOMI[idx_worst] == "mutazioni_congiunte_estremi", (
        f"expected mutazioni_congiunte_estremi to have the largest deficit_fraction, "
        f"got {_NOMI[idx_worst]}: {dict(zip(_NOMI, _DEFICIT))}"
    )
    baseline = _DEFICIT[_IDX["uniforme_equilibrio"]]
    worst = _DEFICIT[_IDX["mutazioni_congiunte_estremi"]]
    assert worst > baseline + 0.1, (
        f"double-ended mutation deficit ({worst:.4f}) should clearly exceed "
        f"the uniform baseline ({baseline:.4f}) by a wide margin"
    )


if __name__ == "__main__":
    print("============================================================")
    print("🔬 EXTREME-GEOMETRY BENCHMARK: standalone diagnostic run")
    print("============================================================")
    for i, nome in enumerate(_NOMI):
        print(f"   {nome:28s} | E_rigida={_E_RIGID[i]:+.4f} eV | E_adattiva={_E_STAR[i]:+.4f} eV "
              f"| deficit_frazionario={_DEFICIT[i]:.4f}")
