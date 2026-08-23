"""
Cosmic-ray burst as an erasure: does telling the decoder WHICH qubits were
hit by the burst (real-time detection -- exactly what the paper's own
"operate the chip like a particle detector" method provides, arXiv:2104.05219)
let a Steane [[7,1,3]] code survive an event that would otherwise be
catastrophic?

Chains dense_evolution's newest utilities (continuous_dissipative_evolve /
amplitude_damping_channel / cosmic_ray_burst_profile, promoted from
Dense-Evolution-Discovery Experiment 34) with its promoted QEC decoders
(compute_syndrome / erasure_aware_decode / blind_minimum_weight_decode --
generalized from this same repo's earlier Steane erasure work,
scripts/steane_code_block6_erasure_conversion.py).

Setup: Steane's real erasure bound (Grassl, Beth, Pellizzari 1997) is
exactly d-1=2 simultaneous erasures. The real cosmic-ray hot spot
(arXiv:2104.05219, Fig. 3c) affects a localized patch of physically
adjacent qubits -- approximated here as exactly 2 "hot spot" qubits,
matching the code's real correction capacity by construction, to isolate
the effect cleanly (the same isolation steane_code_block6 used, now
driven by a realistic burst-derived probability instead of an arbitrary
one).

SIMPLIFICATION, stated plainly: models the burst as inducing bit-flip (X)
errors on the affected qubits, not a full I/X/Y/Z maximally-mixed erasure
like block6's photon-loss model -- consistent with amplitude damping's
own bit-flip-like treatment in this repo's simplified Pauli-error
simulations. Non-hot-spot qubits also carry a small independent X-error
rate, so this isn't an artificially clean single-event shot.
"""
import numpy as np
import jax.numpy as jnp

from dense_evolution import (cosmic_ray_burst_profile, compute_syndrome,
                              erasure_aware_decode, blind_minimum_weight_decode)

N_QUBITS = 7
STEANE_X = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']
STEANE_Z = ['IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
STABILIZERS = STEANE_X + STEANE_Z

HOT_SPOT = (0, 1)   # exactly d-1=2, Steane's real erasure-correcting capacity
BASELINE_P = 0.01   # small independent background X-error rate, all 7 qubits

# Real burst peak: at t=1ms both of cosmic_ray_burst_profile's rise stages
# are long saturated while its 25ms decay has barely acted -- the paper's
# real peak ratio (3.75x baseline) applies almost in full here.
PEAK_TIME_US = 1000.0
P_HOTSPOT_PEAK = float(cosmic_ray_burst_profile(jnp.array([PEAK_TIME_US]), baseline_gamma=BASELINE_P)[0])


def sample_shot(rng):
    # Herald a hot-spot qubit ONLY on the trials where the burst actually
    # strikes it this time -- NOT unconditionally on every trial. An
    # earlier version of this script heralded (0, 1) on every single shot
    # regardless of whether they were actually disturbed, which forced
    # erasure_aware_decode to restrict its search there even when the real
    # error was elsewhere (a baseline qubit) or nonexistent -- an
    # unrealistic erasure model that made the informed decoder WORSE than
    # blind. Real erasure heralding is a per-trial event, exactly like
    # steane_code_block6_erasure_conversion.py's own `heralded = [q for q
    # in range(N_DATA) if rng.random() < p]`.
    true_error = ['I'] * N_QUBITS
    heralded = []
    for q in HOT_SPOT:
        if rng.random() < P_HOTSPOT_PEAK:
            true_error[q] = 'X'
            heralded.append(q)
    for q in range(N_QUBITS):
        if q in HOT_SPOT:
            continue
        if rng.random() < BASELINE_P:
            true_error[q] = 'X'
    return true_error, heralded


def residual_is_logical_failure(true_error, correction):
    # Steane, logical X_L=XXXXXXX / logical Z_L=ZZZZZZZ convention: a
    # residual (true_error XOR correction) with odd total X-parity
    # anticommutes with logical Z_L -- a real logical bit-flip survived.
    # Even parity means a harmless stabilizer element (or identity). Same
    # check as steane_code_block6_erasure_conversion.py's
    # apply_correction_and_check.
    parity = 0
    for te, corr in zip(true_error, correction):
        parity ^= int((te in ('X', 'Y')) != (corr in ('X', 'Y')))
    return parity == 1


def run(n_trials=20000, seed=2026):
    rng = np.random.default_rng(seed)
    n_fail_blind = 0
    n_fail_erasure = 0
    n_heralded_shots = 0
    n_heralded_fail_blind = 0
    n_heralded_fail_erasure = 0
    for _ in range(n_trials):
        true_error, heralded = sample_shot(rng)
        syndrome = compute_syndrome(''.join(true_error), STABILIZERS)

        corr_blind = blind_minimum_weight_decode(syndrome, N_QUBITS, STABILIZERS)

        # Erasure-aware STRATEGY, not just the raw decoder call: use herald
        # info when there is any and it resolves the syndrome uniquely;
        # otherwise fall back to blind decoding -- the same policy
        # steane_code_block6_erasure_conversion.py's erasure_aware_decode
        # follows (never worse than blind, only better when herald info
        # actually helps).
        corr_erasure = None
        if heralded:
            n_heralded_shots += 1
            corr_erasure = erasure_aware_decode(syndrome, heralded, N_QUBITS, STABILIZERS)
        if corr_erasure is None:
            corr_erasure = corr_blind

        fail_blind = corr_blind is None or residual_is_logical_failure(true_error, list(corr_blind))
        fail_erasure = corr_erasure is None or residual_is_logical_failure(true_error, list(corr_erasure))
        n_fail_blind += int(fail_blind)
        n_fail_erasure += int(fail_erasure)
        if heralded:
            n_heralded_fail_blind += int(fail_blind)
            n_heralded_fail_erasure += int(fail_erasure)

    return dict(
        rate_blind=n_fail_blind / n_trials,
        rate_erasure=n_fail_erasure / n_trials,
        n_heralded_shots=n_heralded_shots,
        heralded_rate_blind=(n_heralded_fail_blind / n_heralded_shots) if n_heralded_shots else float('nan'),
        heralded_rate_erasure=(n_heralded_fail_erasure / n_heralded_shots) if n_heralded_shots else float('nan'),
    )


N_TRIALS = 20000
RESULTS = run(N_TRIALS)


if __name__ == "__main__":
    print(f"Baseline per-qubit X-error rate: {BASELINE_P}")
    print(f"Hot-spot X-error rate at burst peak (t=1ms): {P_HOTSPOT_PEAK:.4f} "
          f"({P_HOTSPOT_PEAK / BASELINE_P:.2f}x baseline)")
    print(f"Shots with a real hot-spot herald: {RESULTS['n_heralded_shots']}/{N_TRIALS}")
    print(f"Logical error rate over all {N_TRIALS} trials:")
    print(f"  blind decoder (no herald info):         {RESULTS['rate_blind']:.4f}")
    print(f"  erasure-aware strategy (uses heralds):  {RESULTS['rate_erasure']:.4f}")
    print(f"Logical error rate, conditioned on shots with a real herald "
          f"({RESULTS['n_heralded_shots']} shots) -- where the effect actually lives:")
    print(f"  blind decoder:          {RESULTS['heralded_rate_blind']:.4f}")
    print(f"  erasure-aware strategy: {RESULTS['heralded_rate_erasure']:.4f}")
