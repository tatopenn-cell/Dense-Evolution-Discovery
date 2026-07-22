"""
vqe_silicon_molecular_optimized.py -- exact-gradient Adam optimization tests
-------------------------------------------------------------------------------
2 tests, target < 20s total.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = _import_script("vqe_silicon_molecular_optimized")
real = _import_script("vqe_silicon_molecular")


def test_batched_exact_gradient_matches_finite_difference_and_real_script():
    """The batched kinetic/gradient function must agree with (a) an
    independent finite-difference gradient and (b) the REAL
    calcola_energia_molecolare from vqe_silicon_molecular.py, not a
    re-implementation."""
    R_test = 2.35
    t_R = opt.T0_MOL * np.exp(-opt.BETA * (R_test - opt.R0_MOL))
    V_rep = opt.V0_MOL * np.exp(-opt.GAMMA * (R_test - opt.R0_MOL))

    for theta0 in [0.2, 0.62, 1.4]:
        kinetic, grad = opt.batched_kinetic_and_exact_gradient(np.array([theta0]))

        e_at_theta = real.calcola_energia_molecolare(R_test, theta0)
        kinetic_from_real = (e_at_theta - V_rep) / (-(t_R / 2.0))
        assert kinetic[0] == pytest.approx(kinetic_from_real, abs=1e-9), (
            f"theta={theta0}: batched kinetic {kinetic[0]} != real-script kinetic {kinetic_from_real}"
        )

        h = 1e-6
        e_plus = real.calcola_energia_molecolare(R_test, theta0 + h)
        e_minus = real.calcola_energia_molecolare(R_test, theta0 - h)
        fd_grad_kinetic = ((e_plus - e_minus) / (2 * h)) / (-(t_R / 2.0))
        assert grad[0] == pytest.approx(fd_grad_kinetic, abs=1e-5), (
            f"theta={theta0}: exact PSR grad {grad[0]} != finite-diff grad {fd_grad_kinetic}"
        )


def test_optimize_pec_converges_and_never_gets_worse():
    """A small-scale Adam run (5 R points, 30 epochs) must: (1) reduce the
    mean gradient magnitude substantially from its initial value, and
    (2) never end up at a HIGHER energy than the fixed theta=0.38 starting
    point at any R -- Adam starts at theta=0.38 and should only improve."""
    R_space = np.linspace(1.4, 4.0, 5)

    theta0 = np.full(len(R_space), 0.38)
    kinetic0, grad0 = opt.batched_kinetic_and_exact_gradient(theta0)
    t_R = opt.T0_MOL * np.exp(-opt.BETA * (R_space - opt.R0_MOL))
    V_rep = opt.V0_MOL * np.exp(-opt.GAMMA * (R_space - opt.R0_MOL))
    E_fixed = -(t_R / 2.0) * kinetic0 + V_rep
    initial_grad_mag = np.mean(np.abs(-(t_R / 2.0) * grad0))

    theta_star, E_star, grad_final = opt.optimize_pec(R_space, n_epochs=30, verbose=False)
    final_grad_mag = np.mean(np.abs(-(t_R / 2.0) * grad_final))

    assert final_grad_mag < 0.3 * initial_grad_mag, (
        f"gradient magnitude should shrink substantially: "
        f"initial={initial_grad_mag:.4f}, final={final_grad_mag:.4f}"
    )
    assert np.all(E_star <= E_fixed + 1e-6), (
        f"optimized energy must never exceed the fixed-theta=0.38 starting energy: "
        f"E_star={E_star}, E_fixed={E_fixed}"
    )
