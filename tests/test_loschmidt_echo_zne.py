"""
Tests for scripts/loschmidt_echo_zne.py -- imports the real script (guarded
behind `if __name__ == "__main__":`, same pattern as
test_integration_smoke.py) and calls its real functions directly, at
reduced trajectory count for CI speed.
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


loschmidt = _import_script("loschmidt_echo_zne")


def test_noiseless_echo_is_exact():
    """p=0 forward+backward must return EXACTLY the initial state -- if
    this doesn't hold, invert_layer() is wrong and nothing downstream
    (noisy fidelities, ZNE correction) can be trusted."""
    fidelity = loschmidt.noiseless_echo_fidelity(seed=0)
    assert abs(fidelity - 1.0) < 1e-9, (
        f"noiseless echo fidelity should be 1.0, got {fidelity:.12f}"
    )


def test_invert_layer_is_actual_inverse():
    """Independent check of invert_layer() alone (not routed through
    run_echo): apply one random forward layer then its inverse directly on
    a simulator and confirm the statevector returns to the initial state,
    for several different random layers."""
    import dense_evolution as de

    rng = np.random.default_rng(123)
    for _ in range(5):
        layers = loschmidt.build_forward_layers(rng)
        layer = layers[0]

        sim = loschmidt.initial_state()
        ideal_sv = np.asarray(sim.get_statevector())

        sim.run_circuit(layer, transpile=False)
        sim.run_circuit(loschmidt.invert_layer(layer), transpile=False)
        final_sv = np.asarray(sim.get_statevector())

        fidelity = float(np.abs(np.vdot(ideal_sv, final_sv)) ** 2)
        assert abs(fidelity - 1.0) < 1e-9, (
            f"layer + invert_layer(layer) should be identity, fidelity={fidelity:.12f}"
        )


def test_noisy_echo_fidelity_is_a_valid_probability_below_one():
    """A single noisy trajectory's return fidelity must be a valid
    probability (in [0, 1]) and strictly below the noiseless value of 1.0
    -- noise must actually degrade the echo."""
    rng = np.random.default_rng(0)
    forward_layers = loschmidt.build_forward_layers(rng)
    ideal_sv = np.asarray(loschmidt.initial_state().get_statevector())

    final_sv = loschmidt.run_echo(forward_layers, p=0.05, seed=1)
    fidelity = float(np.abs(np.vdot(ideal_sv, final_sv)) ** 2)

    assert 0.0 <= fidelity <= 1.0
    assert fidelity < 1.0 - 1e-6, "noise should measurably degrade the echo"


def test_zne_correction_improves_return_fidelity():
    """Calls the REAL run_experiment() from loschmidt_echo_zne.py (reduced
    K_TRAJECTORIES for CI speed) and checks the actual physics claim: ZNE
    Richardson extrapolation over the noisy return density matrix must
    recover more of the lost fidelity than the raw (unmitigated) noisy
    echo, at two independent seeds."""
    for seed in (42, 7):
        raw_fidelity, corrected_fidelity, _, _ = loschmidt.run_experiment(
            seed=seed, scales=(1.0, 1.5, 2.0), k_trajectories=40, base_p=0.015,
        )
        assert 0.0 <= raw_fidelity <= 1.0
        assert 0.0 <= corrected_fidelity <= 1.0
        assert corrected_fidelity > raw_fidelity, (
            f"seed={seed}: ZNE-corrected fidelity ({corrected_fidelity:.4f}) should "
            f"exceed the raw noisy fidelity ({raw_fidelity:.4f})"
        )
