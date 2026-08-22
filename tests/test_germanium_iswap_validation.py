"""
Tests for scripts/germanium_iswap_validation.py -- imports the real script
(runs the full experiment on import, same convention as this repo's other
tests) and checks its real module-level results.
"""
import importlib.util
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_script(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


germ = _import_script("germanium_iswap_validation")


def test_exact_evolution_reaches_full_swap():
    assert germ.prob_10_exact > 0.9999
    assert germ.prob_01_exact < 1e-4


def test_trotter_circuit_matches_native_iswap():
    # XX/YY always commute (Bell basis), so even 4 slices should already be
    # extremely close to the native iSWAP gate -- no real Trotter error here.
    for n_slices, gate_count, fidelity in germ.trotter_convergence:
        assert fidelity > 0.999
    assert germ.trotter_convergence[-1][2] > 0.999999


def test_spam_channel_matches_uhlmann_fidelity_exactly():
    # The hand-derived exact sequential-composition formula must match
    # dense_evolution.uhlmann_fidelity to machine precision.
    for result in germ.spam_roundtrip_results.values():
        assert abs(result["uhlmann"] - result["exact"]) < 1e-9


def test_fiswap_inferred_matches_paper():
    # FQPT=0.60 / <diag(PSPAM)>=0.69 must reproduce the paper's own FiSWAP~87%.
    assert abs(germ.FISWAP_INFERRED - 0.87) < 0.01


def test_randomized_benchmarking_recovers_injected_noise_parameter():
    assert abs(germ.rb_result["r_fit"] - germ.rb_result["r_expected"]) < 0.01


def test_per_state_spam_profile_matches_reported_mean():
    weighted_mean = (germ.mean_low * germ.N_LOW +
                      germ.p_high_states * (germ.N_TOTAL - germ.N_LOW)) / germ.N_TOTAL
    assert abs(weighted_mean - germ.DIAG_PSPAM_MEASURED) < 1e-9


def test_general_regime_trotter_error_is_real_and_converges():
    # Unlike the pure iSWAP point, this Hamiltonian genuinely does not
    # commute with itself at different times, so infidelity must shrink
    # monotonically (order 1) as slice count grows.
    infidelities_order1 = [inf for _, inf in germ.general_trotter_results[1]]
    assert infidelities_order1[0] > infidelities_order1[-1]
    assert all(a > b for a, b in zip(infidelities_order1, infidelities_order1[1:]))


def test_order2_trotter_beats_order1_at_matched_slice_count():
    for (n1, inf1), (n2, inf2) in zip(germ.general_trotter_results[1], germ.general_trotter_results[2]):
        assert n1 == n2
        assert inf2 < inf1
