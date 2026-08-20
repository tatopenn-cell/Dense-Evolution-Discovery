"""
vqe_extreme_geometries_deep.py -- 12-parameter (7-qubit, 6-bond) benchmark tests
-------------------------------------------------------------------------------
7 tests. The expensive per-bond/shared-theta optimizations (6 scenarios) and
the conformational search are each run ONCE at module import and reused by
every test below, instead of repeating the computation per test.
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


deep = _import_script("vqe_extreme_geometries_deep")

_SCENARI = deep.build_extreme_geometries()
_NOMI = list(_SCENARI.keys())
_IDX = {n: i for i, n in enumerate(_NOMI)}
_R_MATRIX = np.stack([_SCENARI[n] for n in _NOMI])

_THETA_RIGID = np.full((len(_NOMI), deep.N_BONDS), deep.RIGID_THETA)
_E_RIGID = deep.energy_from_theta(_R_MATRIX, _THETA_RIGID)

_THETA_STAR, _E_STAR = deep.optimize_theta_per_geometry(_R_MATRIX, n_epochs=40, verbose=False)
_KINETIC_PERBOND, _ = deep.batched_per_bond_kinetic_and_jacobian(_THETA_STAR)

_THETA_SHARED_STAR, _KINETIC_SHARED = deep.optimize_shared_theta_per_geometry(_R_MATRIX, n_epochs=40)
_DEFICIT = deep.deficit_fraction(_R_MATRIX, _KINETIC_PERBOND, _KINETIC_SHARED)


def test_energy_matches_independent_reference_for_uniform_geometry():
    """energy_from_theta's formula (local per-bond hopping + repulsion at
    the mean bond length) must match an INDEPENDENT calculation on a fresh
    7-qubit/6-bond circuit for uniform R/theta.

    Unlike the 10-parameter script (N_Q=6, 5 bonds -- matching
    vqe_silicon_molecular.calcola_energia_molecolare's hardcoded 5-bond
    chain exactly, so it collapses onto that reference formula), this
    6-bond/7-qubit chain is a DIFFERENT physical system: comparing it
    directly to the 5-bond reference formula is not a meaningful
    equivalence (an earlier version of this test did exactly that and
    failed, correctly -- summing kinetic contributions over 6 bonds is
    not the same quantity as summing over 5). So this builds its own
    independent reference on the same 6-bond model instead."""
    for R_test, theta_test in [(2.35, 0.38), (1.6, 0.62), (3.9, 1.1)]:
        t_R = deep.T0_MOL * np.exp(-deep.BETA * (R_test - deep.R0_MOL))
        v_rep = deep.V0_MOL * np.exp(-deep.GAMMA * (R_test - deep.R0_MOL))

        s = deep.de.DenseSVSimulator(n_qubits=deep.N_Q, use_float32=False)
        ops = [['x', 0]]
        for q in range(deep.N_BONDS):
            ops += [['cx', q + 1, q], ['ry', q + 1, float(theta_test)], ['cx', q, q + 1],
                    ['ry', q + 1, float(-theta_test)], ['cx', q + 1, q]]
        s.set_initial_state()
        s.run_circuit_jit_beast_mode(ops)
        kinetic_total = np.sum(deep._kinetic_per_bond_from_sv(np.asarray(s.get_statevector())))
        e_reference = -(t_R / 2.0) * kinetic_total + v_rep

        R_row = np.full((1, deep.N_BONDS), R_test)
        theta_row = np.full((1, deep.N_BONDS), theta_test)
        e_generalized = deep.energy_from_theta(R_row, theta_row)[0]
        assert e_generalized == pytest.approx(e_reference, abs=1e-9), (
            f"R={R_test}, theta={theta_test}: generalized {e_generalized} != independent reference {e_reference}"
        )


def test_per_bond_jacobian_matches_finite_difference():
    """The exact per-bond Jacobian (including off-diagonal entries) must
    agree with an independent finite-difference reference on a fresh
    single circuit, for the 6-bond (12-parameter) chain."""
    theta_vec = np.array([0.15, 0.85, 0.4, 1.3, 0.6, 1.0])
    kinetic, jac = deep.batched_per_bond_kinetic_and_jacobian(theta_vec[None, :])

    def _kinetic_per_bond_single(tv):
        s = deep.de.DenseSVSimulator(n_qubits=deep.N_Q, use_float32=False)
        ops = [['x', 0]]
        for q in range(deep.N_BONDS):
            ops += [['cx', q + 1, q], ['ry', q + 1, float(tv[q])], ['cx', q, q + 1],
                    ['ry', q + 1, float(-tv[q])], ['cx', q + 1, q]]
        s.set_initial_state()
        s.run_circuit_jit_beast_mode(ops)
        return deep._kinetic_per_bond_from_sv(np.asarray(s.get_statevector()))

    kinetic_ref = _kinetic_per_bond_single(theta_vec)
    assert np.max(np.abs(kinetic[0] - kinetic_ref)) < 1e-9

    h = 1e-6
    for r in range(deep.N_BONDS):
        tp = theta_vec.copy(); tp[r] += h
        tm = theta_vec.copy(); tm[r] -= h
        fd = (_kinetic_per_bond_single(tp) - _kinetic_per_bond_single(tm)) / (2 * h)
        assert np.max(np.abs(jac[0, :, r] - fd)) < 1e-5, (
            f"bond param {r}: exact jacobian column {jac[0, :, r]} != finite-diff {fd}"
        )


def test_adaptive_per_bond_never_worse_than_rigid_fixed_theta():
    """Per-bond adaptive optimization (12 free parameters) must reach an
    energy at or below the fixed shared theta=0.38 baseline at every
    scenario."""
    assert np.all(_E_STAR <= _E_RIGID + 1e-3), f"E_star={_E_STAR}, E_rigid={_E_RIGID}"


def test_per_bond_never_worse_than_best_achievable_shared_theta():
    """The best possible single shared theta is a special case of the
    per-bond parameter family, so per-bond adaptive optimization must reach
    a weighted kinetic energy at least as large (deficit_fraction >= 0 up
    to optimizer tolerance)."""
    assert np.all(_DEFICIT > -0.01), f"deficit_fraction should be >= 0: {dict(zip(_NOMI, _DEFICIT))}"


def test_uniform_geometries_share_identical_deficit_fraction():
    """For any UNIFORM-R geometry, deficit_fraction is a pure ansatz-
    expressivity cost that cancels the overall t(R) energy scale -- it
    must come out IDENTICAL for equilibrium, compressed, and dissociated
    geometries, regardless of chain length."""
    uniform_names = ["uniforme_equilibrio", "uniforme_compressa", "uniforme_dissociata"]
    values = [_DEFICIT[_IDX[n]] for n in uniform_names]
    assert max(values) - min(values) < 5e-3, (
        f"uniform-geometry deficit_fraction should be scale-invariant: {dict(zip(uniform_names, values))}"
    )


def test_double_ended_mutation_is_the_hardest_case_at_this_depth():
    """At 12 parameters, mutazione_localizzata and distorsione_alternata
    measured ABOVE the uniform baseline (unlike at 10 parameters, where
    they sat below it -- see the top-of-file note in
    vqe_extreme_geometries_deep.py). Only mutazioni_congiunte_estremi is a
    robust standout across both depths: this locks in that it remains the
    single worst scenario at 12 parameters too, without assuming the other
    two scenarios' below/above-baseline direction carries over from the
    10-parameter script."""
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


def test_conformational_search_converges_toward_an_interior_stationary_point():
    """optimize_geometry_and_theta_jointly must find a genuine interior
    minimum, not an artifact of the R_bounds clip -- this is a regression
    guard against the bug found during development: with the wrong
    (mean-based) repulsion model, every tested starting point drove one
    bond straight to the clip boundary with a suspiciously large negative
    energy. With the correct per-bond repulsion, R* must stay strictly
    inside the bounds and the gradient magnitude (both on R and on theta)
    must shrink substantially from its initial value -- not full
    convergence (that needs hundreds more epochs than a CI budget allows),
    but clearly moving toward a stationary point, not a boundary."""
    R_init = np.stack([
        np.full(deep.N_BONDS, deep.R0_MOL),
        deep.R0_MOL + np.array([0.3, -0.2, 0.4, -0.3, 0.2, -0.1]),
    ])
    R_star, theta_star, E_star, grad_R, theta_grad = deep.optimize_geometry_and_theta_jointly(
        R_init, n_epochs=100, lr_R=0.04, lr_theta=0.15, verbose=False)

    lo, hi = 0.6, 8.0
    margin = 0.05
    assert np.all(R_star > lo + margin) and np.all(R_star < hi - margin), (
        f"R* should stay strictly inside (R_bounds), got {R_star} -- "
        f"a value at the boundary means the search is hitting an artificial "
        f"clip instead of a genuine minimum (the bug this test guards against)"
    )
    assert np.all(np.isfinite(E_star))

    # gradient at epoch 1 (before any update) as the "initial" reference
    kinetic0, jac0 = deep.batched_per_bond_kinetic_and_jacobian(
        np.full((R_init.shape[0], deep.N_BONDS), deep.RIGID_THETA))
    t_local0 = deep._local_hopping(R_init)
    grad_R0 = (deep.BETA / 2.0) * t_local0 * kinetic0 - deep.GAMMA * deep._local_repulsion_per_bond(R_init)
    initial_grad_mag = np.mean(np.abs(grad_R0))
    final_grad_mag = np.mean(np.abs(grad_R))
    assert final_grad_mag < 0.5 * initial_grad_mag, (
        f"R gradient magnitude should shrink substantially: "
        f"initial={initial_grad_mag:.4f}, final={final_grad_mag:.4f}"
    )
