"""
Smoke test for scripts/quantum_ruzsa_magic_entropy.py -- imports the real
script and calls its real functions directly.
"""
import importlib.util
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ruzsa = _import_script("quantum_ruzsa_magic_entropy")


def test_key_unitary_matches_lemma_9_identity():
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                idx_in = (x1 << 2) | (x2 << 1) | x3
                sv_in = jnp.zeros(8, dtype=jnp.complex128).at[idx_in].set(1.0)
                sv_out = ruzsa.KEY_UNITARY_K3 @ sv_in
                y1, y2, y3 = x1 ^ x2 ^ x3, x2 ^ x1, x3 ^ x1
                idx_expected = (y1 << 2) | (y2 << 1) | y3
                out_idx = int(jnp.argmax(jnp.abs(sv_out)))
                assert out_idx == idx_expected
                assert abs(complex(sv_out[out_idx])) > 0.999


def test_key_unitary_is_unitary():
    v = ruzsa.KEY_UNITARY_K3
    identity = jnp.eye(8, dtype=jnp.complex128)
    assert np.allclose(np.array(v @ jnp.conj(v).T), np.array(identity), atol=1e-10)


def test_all_single_qubit_stabilizer_states_have_zero_magic_entropy():
    for name, rho in ruzsa._stabilizer_states().items():
        m = ruzsa.magic_entropy(rho)
        assert m < 1e-8, f"{name} expected ~0 magic entropy, got {m}"


def test_t_and_h_states_have_nonzero_magic_entropy():
    for rho in (ruzsa.t_state_rho(), ruzsa.h_state_rho()):
        m = ruzsa.magic_entropy(rho)
        assert m > 1e-4


def test_self_convolve_output_is_a_valid_density_matrix():
    rho = ruzsa.t_state_rho()
    reduced = ruzsa.self_convolve_3(rho)
    assert abs(complex(jnp.trace(reduced)) - 1.0) < 1e-9
    ev = np.linalg.eigvalsh(np.array(reduced))
    assert np.all(ev > -1e-9)


def test_fully_depolarized_state_has_maximal_magic_entropy():
    maximally_mixed = jnp.eye(2, dtype=jnp.complex128) / 2.0
    m = ruzsa.magic_entropy(maximally_mixed)
    assert abs(m - 1.0) < 1e-6


def test_fully_amplitude_damped_state_returns_to_zero_magic_entropy():
    rho_damped = ruzsa.amplitude_damping_1q(ruzsa.t_state_rho(), 1.0)
    m = ruzsa.magic_entropy(rho_damped)
    assert m < 1e-6
