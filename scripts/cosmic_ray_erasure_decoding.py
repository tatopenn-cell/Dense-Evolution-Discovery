"""
Cosmic-ray burst as an erasure: if the decoder is told WHICH qubits were hit
by the burst, can a Steane [[7,1,3]] code survive an event that would
otherwise be catastrophic? Perfect knowledge of the hit qubits is an
assumption of this script, not something the paper provides: arXiv:2104.05219
turns the processor into a "time-resolved detector" that estimates how many
qubits are in error over time, from matched-filtered 60 s datasets.

Chains dense_evolution's newest utilities (continuous_dissipative_evolve /
amplitude_damping_channel / cosmic_ray_burst_profile, promoted from
Dense-Evolution-Discovery Experiment 34) with its promoted QEC decoders
(compute_syndrome / erasure_aware_decode / blind_minimum_weight_decode --
generalized from this same repo's earlier Steane erasure work,
scripts/steane_code_block6_erasure_conversion.py).

Setup: Steane's erasure bound (Grassl, Beth, Pellizzari 1997) is d-1=2
simultaneous erasures. The paper reports that the burst first hits a small
patch of qubits (simultaneous errors jump from ~4 to ~10 within ~10 us), then
spreads: at ~1.5 ms the hot spot has grown and the whole qubit patch shows
elevated error rates, before an exponential recovery with a ~25 ms time
constant. The module-level run() fixes the hot spot at 2 qubits, equal to the
code's correction capacity by construction (the isolation steane_code_block6
used). sweep() removes that choice: it varies the hot-spot size from 1 to 7
qubits and the time since impact, and computes the failure probabilities
exactly by enumerating all 2**7 X-error patterns instead of sampling them.

The 3.75x peak ratio is 15/4: the paper's chip-wide count of simultaneous
errors at ~1 ms over its baseline of ~4. Applying that chip-wide ratio to the
hit qubits' own X-error probability is a modelling choice, and it understates
a localized hot spot, whose per-qubit rate is higher than the chip average.

SIMPLIFICATION, stated plainly: models the burst as inducing bit-flip (X)
errors on the affected qubits, not a full I/X/Y/Z maximally-mixed erasure
like block6's photon-loss model -- consistent with amplitude damping's
own bit-flip-like treatment in this repo's simplified Pauli-error
simulations. Non-hot-spot qubits also carry a small independent X-error
rate, so this isn't an artificially clean single-event shot.
"""
import functools

import numpy as np
import jax.numpy as jnp

from dense_evolution import (cosmic_ray_burst_profile, compute_syndrome,
                              erasure_aware_decode, blind_minimum_weight_decode)

N_QUBITS = 7
STEANE_X = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']
STEANE_Z = ['IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ']
STABILIZERS = STEANE_X + STEANE_Z

HOT_SPOT = (0, 1)   # d-1=2, Steane's erasure-correcting capacity
BASELINE_P = 0.01   # small independent background X-error rate, all 7 qubits

# Burst peak: at t=1ms cosmic_ray_burst_profile's second rise stage (tau2=300us)
# is ~96% saturated while its 25ms decay has barely acted, so the 3.75x peak
# ratio applies almost in full. The paper itself places the peak near 1.5 ms.
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


SWEEP_SIZES = tuple(range(1, N_QUBITS + 1))
SWEEP_TIMES_US = (0.0, 10.0, 100.0, 1000.0, 1500.0, 10000.0, 25000.0, 100000.0)


@functools.lru_cache(maxsize=None)
def _blind_fails(mask):
    err = ['X' if mask >> q & 1 else 'I' for q in range(N_QUBITS)]
    syn = compute_syndrome(''.join(err), STABILIZERS)
    corr = blind_minimum_weight_decode(syn, N_QUBITS, STABILIZERS)
    return corr is None or residual_is_logical_failure(err, list(corr))


@functools.lru_cache(maxsize=None)
def _erasure_fails(mask, k):
    err = ['X' if mask >> q & 1 else 'I' for q in range(N_QUBITS)]
    heralded = [q for q in range(k) if mask >> q & 1]
    syn = compute_syndrome(''.join(err), STABILIZERS)
    corr = None
    if heralded:
        corr = erasure_aware_decode(syn, heralded, N_QUBITS, STABILIZERS)
    if corr is None:
        corr = blind_minimum_weight_decode(syn, N_QUBITS, STABILIZERS)
    return corr is None or residual_is_logical_failure(err, list(corr))


def exact_rates(k, p_hot):
    # Exact, not sampled: the X-error pattern is a product distribution over
    # the 7 qubits (hot-spot qubits 0..k-1 at p_hot, the rest at BASELINE_P),
    # so summing over all 2**7 patterns gives the failure probabilities
    # with no sampling noise. Same strategy as run(): herald = hot-spot
    # qubits that actually flipped, fall back to blind when it does not help.
    tot = blind = eras = herald_tot = herald_blind = herald_eras = 0.0
    for mask in range(1 << N_QUBITS):
        pr = 1.0
        for q in range(N_QUBITS):
            p = p_hot if q < k else BASELINE_P
            pr *= p if mask >> q & 1 else 1.0 - p
        fb, fe = _blind_fails(mask), _erasure_fails(mask, k)
        tot += pr
        blind += pr * fb
        eras += pr * fe
        if mask & ((1 << k) - 1):
            herald_tot += pr
            herald_blind += pr * fb
            herald_eras += pr * fe
    return dict(rate_blind=blind, rate_erasure=eras, p_herald=herald_tot,
                heralded_rate_blind=herald_blind / herald_tot,
                heralded_rate_erasure=herald_eras / herald_tot)


def sweep(sizes=SWEEP_SIZES, times_us=SWEEP_TIMES_US):
    # Point 1 (hot-spot size) and point 3 (time since impact) together: the
    # hot-spot qubits' X-error probability follows cosmic_ray_burst_profile
    # over time, and the number of hot-spot qubits k is varied 1..7.
    p_t = [float(cosmic_ray_burst_profile(jnp.array([t]), baseline_gamma=BASELINE_P)[0])
           for t in times_us]
    return {(k, t): exact_rates(k, p) for k in sizes for t, p in zip(times_us, p_t)}


@functools.lru_cache(maxsize=None)
def _erasure_fails_h(mask, hmask):
    err = ['X' if mask >> q & 1 else 'I' for q in range(N_QUBITS)]
    heralded = [q for q in range(N_QUBITS) if hmask >> q & 1]
    syn = compute_syndrome(''.join(err), STABILIZERS)
    corr = None
    if heralded:
        corr = erasure_aware_decode(syn, heralded, N_QUBITS, STABILIZERS)
    if corr is None:
        corr = blind_minimum_weight_decode(syn, N_QUBITS, STABILIZERS)
    return corr is None or residual_is_logical_failure(err, list(corr))


def exact_rates_imperfect(k, p_hot, fn, fp):
    # Point 4: imperfect detection. A hit hot-spot qubit is flagged with
    # probability 1-fn (fn = false-negative rate); a hot-spot qubit that was
    # NOT hit is flagged anyway with probability fp (false positive). The
    # decoder treats every flag as a true erasure. Still exact: enumerates
    # every (error pattern, flag set) pair. Cost grows as 5**k, so keep k small.
    blind = eras = 0.0
    for mask in range(1 << N_QUBITS):
        pr = 1.0
        for q in range(N_QUBITS):
            p = p_hot if q < k else BASELINE_P
            pr *= p if mask >> q & 1 else 1.0 - p
        blind += pr * _blind_fails(mask)
        for hmask in range(1 << k):
            ph = 1.0
            for q in range(k):
                hit = mask >> q & 1
                flag = hmask >> q & 1
                ph *= (1.0 - fn if flag else fn) if hit else (fp if flag else 1.0 - fp)
            if ph:
                eras += pr * ph * _erasure_fails_h(mask, hmask)
    return dict(rate_blind=blind, rate_erasure=eras)


def rate_with_latency(k, t_us, latency_us, fn=0.0, fp=0.0):
    # Point 4, delay: flags only exist once detection has finished, latency_us
    # after impact. Decoding at t_us < latency_us has no herald: blind.
    p = float(cosmic_ray_burst_profile(jnp.array([t_us]), baseline_gamma=BASELINE_P)[0])
    r = exact_rates_imperfect(k, p, fn, fp)
    return r['rate_blind'] if t_us < latency_us else r['rate_erasure']


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

    ex = exact_rates(len(HOT_SPOT), P_HOTSPOT_PEAK)
    print(f"Exact (no sampling), same setup k={len(HOT_SPOT)}, t=1ms: "
          f"blind {ex['rate_blind']:.4f}, erasure {ex['rate_erasure']:.4f}, "
          f"heralded blind {ex['heralded_rate_blind']:.4f}, "
          f"heralded erasure {ex['heralded_rate_erasure']:.4f}")

    sw = sweep()
    print("Exact logical failure rate, erasure-aware / blind, by hot-spot size k and time t (us):")
    print("  k \\ t  " + "".join(f"{t:>14.0f}" for t in SWEEP_TIMES_US))
    for k in SWEEP_SIZES:
        print(f"  {k:<6}" + "".join(
            f"{sw[(k, t)]['rate_erasure']:>7.4f}/{sw[(k, t)]['rate_blind']:<6.4f}" for t in SWEEP_TIMES_US))
    print("Imperfect detection at t=1ms: erasure-aware failure rate (blind in last column), rows fn, columns fp:")
    for k in (2, 3, 4):
        print(f"  k={k}   fp=" + "".join(f"{fp:>9.2f}" for fp in (0.0, 0.01, 0.05, 0.2)) + "    blind")
        for fn in (0.0, 0.1, 0.3, 0.5, 1.0):
            row = [exact_rates_imperfect(k, P_HOTSPOT_PEAK, fn, fp) for fp in (0.0, 0.01, 0.05, 0.2)]
            print(f"   fn={fn:<4}     " + "".join(f"{r['rate_erasure']:>9.4f}" for r in row)
                  + f"    {row[0]['rate_blind']:.4f}")
    print("Detection latency, k=3, perfect flags, failure rate by decoding time t (us):")
    for L in (0.0, 100.0, 1000.0, 10000.0):
        print(f"  latency {L:>7.0f}: " + "".join(
            f"{rate_with_latency(3, t, L):>9.4f}" for t in (10.0, 100.0, 1000.0, 1500.0, 10000.0, 25000.0)))
    print("Exact failure rate conditioned on at least one hot-spot qubit hit, at t=1ms:")
    for k in SWEEP_SIZES:
        r = sw[(k, 1000.0)]
        print(f"  k={k}: P(herald)={r['p_herald']:.4f}  blind {r['heralded_rate_blind']:.4f}  "
              f"erasure {r['heralded_rate_erasure']:.4f}")
