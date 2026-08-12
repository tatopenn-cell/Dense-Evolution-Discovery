import pathlib
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.healing import calculate_delta_preemp

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE ZNE-BEFORE-PSR: scripts/zne_stabilized_psr_gradient.py found that
# the STATIC 2-point Richardson correction (E_zne = 2*E1 - E2), applied to
# every single-gate PSR term before the chain-rule combination, cuts bias and
# RMSE roughly in half away from a gradient zero-crossing (theta=0.2, 0.38,
# 1.0) but makes bias, variance, AND RMSE all WORSE near one (theta=0.62) --
# there's little systematic error left to correct there, so Richardson's
# variance amplification just adds noise.
#
# This script asks: does ATTENUATING the correction when per-shift confidence
# is low -- instead of always applying the full static correction -- recover
# the theta=0.62 case without giving up the wins elsewhere?
#
# Reuses dense_evolution.healing.calculate_delta_preemp(current_sigma,
# target_sigma_ideal), previously only prototyped in scratch code (never in
# this repo) in tests/test_zne_predictive_healing.py, which combined it with
# a 3-point Richardson and a SYNTHETIC "sigma" (an ad-hoc coherence-like
# proxy, not a real statistical quantity). Here current_sigma is a REAL,
# measured standard error of the mean (SEM = std(per-shot samples)/
# sqrt(n_shots)) of the scale=1.0 shot-averaged estimate for each individual
# single-gate PSR term -- a genuine confidence indicator, not a synthetic
# stand-in. target_sigma_ideal is calibrated for THIS SEM's scale (typically
# ~0.01-0.1), not the package default of 10.0 (which was paired with that
# unrelated synthetic proxy and would be meaningless here).
#
# Extends the already-merged 2-point Richardson (not the 3-point pattern
# from the scratch code, which would need a third noise scale per gate-shift
# and triple the shot cost):
#
#   delta_p = calculate_delta_preemp(SEM_at_scale_1, target_sigma_ideal)
#   attenuation = min(1.0, K_SENSITIVITY * delta_p)
#   c1 = 2.0 - attenuation
#   c2 = -1.0 + attenuation          # c1 + c2 == 1 always, no renormalization
#   E_adaptive = c1*E1 + c2*E2
#
# At delta_p=0 (SEM at or better than the ideal target) this is EXACTLY the
# static correction already merged. As delta_p grows (SEM worse than the
# target -- less confidence), the attenuation grows toward 1 and
# E_adaptive -> E1: no extrapolation, fall back to the raw noisy scale=1.0
# estimate -- a sensible fallback when there isn't enough confidence to
# trust the correction.
#
# NoiseModel.apply_to_sv is always post-hoc on an already-computed clean
# statevector (see the top-of-file note in zne_stabilized_psr_gradient.py
# for the full verification) -- shot-averaging is cheap, one circuit run
# per gate-shift regardless of shot count.
#
# HONEST FINDING -- NEGATIVE RESULT, reported in full because a documented
# negative result has the same scientific value as a positive one here: it
# saves anyone else from re-walking this exact path. SEM-based confidence
# attenuation does NOT recover theta=0.62 without giving up the wins
# elsewhere -- there is no (target_sigma_ideal, k_sensitivity) pair tried
# that Pareto-dominates, and with the calibration actually shipped, the
# adaptive correction does not win outright at ANY of the 4 tested theta --
# it is a linear compromise dial, not a fix.
#
# CALIBRATION SWEEP (exploratory, small budget: 15 trials/100 shots, 3
# thetas) used to pick the shipped default -- RMSE vs. the exact gradient:
#
#   target  k     theta=0.38 (naive/static/adaptive)   theta=0.62            theta=1.0
#   0.03    2.0   2.22 / 0.94 / 2.02                   0.11 / 0.24 / 0.15    1.46 / 0.62 / 1.24
#   0.04    1.0   2.22 / 0.94 / 1.87                   0.11 / 0.24 / 0.33    1.46 / 0.62 / 0.96
#   0.025   3.0   2.22 / 0.94 / 2.16                   0.11 / 0.24 / 0.16    1.46 / 0.62 / 1.35
#
# Every (target, k) pair traces the SAME tradeoff curve: turning the
# attenuation up helps theta=0.62 (closer to naive, sometimes better) but
# drags theta=0.38/1.0 back toward naive too, destroying most of the static
# correction's win there. Turning it down does the reverse.
#
# FINAL RESULT at the shipped default (TARGET_SIGMA_IDEAL=0.03,
# K_SENSITIVITY=2.0), full budget (40 trials, 200 shots, all 4 thetas) --
# RMSE vs. the exact gradient:
#
#   theta   naive    static   adaptive
#   0.20    3.131    1.425    2.501
#   0.38    2.180    0.961    2.170
#   0.62    0.106    0.248    0.175
#   1.00    1.406    0.684    1.139
#
# At full statistics, the adaptive correction does NOT win outright at ANY
# theta: it is always strictly between naive and static (closer to naive),
# meaning it is dominated by whichever of naive/static is best at that
# theta -- static at 0.20/0.38/1.00, naive at 0.62 (0.175 > naive's 0.106,
# so it doesn't even recover the zero-crossing case relative to doing
# nothing). It reduces static's damage at 0.62 without eliminating it, and
# gives back a large fraction of static's win everywhere else. Net: a
# genuine compromise, not a solution.
#
# WHY: verified directly (see the calibration note below) that the measured
# SEM at N_SHOTS=200 sits in the SAME narrow range (~0.016-0.025) at every
# theta tested, including both the "far from zero-crossing" and "near
# zero-crossing" cases. SEM is driven by shot count and NoiseModel's
# depolarizing probability, not by where theta sits relative to a gradient
# zero-crossing -- it carries NO information about the actual quantity that
# determines whether Richardson correction helps or hurts here, which is
# the SIZE OF THE SYSTEMATIC BIAS being corrected relative to the noise, not
# the raw noise level of one measurement in isolation. A single scalar SEM
# threshold therefore cannot distinguish "large bias, correction pays off"
# from "near-zero bias, correction just adds noise" -- it just dials the
# SAME attenuation in everywhere, roughly at random with respect to which
# regime actually needs it.
#
# NOT implemented here (a more promising direction for anyone picking this
# up): a confidence signal built from the CORRECTION TERM's own
# signal-to-noise ratio -- e.g. |k1 - k2| relative to the combined SEM of
# k1 and k2 -- which would directly measure "is there enough systematic
# bias here, relative to noise, to justify extrapolating" instead of just
# "how noisy is a single measurement." That was not tried in this PR; this
# one sticks to the SEM-of-a-single-estimate design that was actually
# built and tested, and reports honestly that it doesn't work.
#
# DEFAULT CALIBRATION SHIPPED (TARGET_SIGMA_IDEAL=0.03, K_SENSITIVITY=2.0):
# the most balanced of the three explored in the small-budget sweep. It is
# a compromise point, not a solution -- do not present it as one in any
# downstream summary of this work.
#
# RE-VERIFIED 2026-08-12 after dense-evolution 8.1.57 (PR #49) fixed
# NoiseModel.apply_to_sv's depolarizing channel (one Pauli decision per
# qubit per shot, applied globally, instead of one independent decision
# per computational-basis branch -- see zne_stabilized_psr_gradient.py's
# own re-verification note for the full mechanism). Re-measured at
# n_trials=40, n_shots=60 (smaller budget than the table above, for CI
# speed -- compare ratios, not raw magnitudes):
#
#   theta   naive    static   adaptive
#   0.38    0.7966   0.4120   0.7125
#   1.00    0.6015   0.6028   0.5784
#
# theta=0.38 is qualitatively unchanged: static wins big, adaptive is a
# worse compromise much closer to naive -- same story as the original
# table.
#
# theta=1.00 is NOT unchanged, and it changes the blanket claim above ("the
# adaptive correction does NOT win outright at ANY theta"). That claim no
# longer holds: static's own RMSE advantage over naive at theta=1.00 has
# collapsed under the corrected noise (0.6028 vs 0.6015 -- see
# zne_stabilized_psr_gradient.py's parallel finding, where the same
# 2-point-Richardson correction's bias win at theta=1.00 got fully offset
# by a variance increase). With static no longer clearly ahead of naive
# there, adaptive's partial attenuation -- which always sits between the
# two -- ends up with the lowest RMSE of the three at theta=1.00, a real
# outright win, not a small-sample artifact (reproduced consistently across
# n_trials in {15, 40, 80} at fixed n_shots=60). The mechanism (SEM being a
# poor confidence proxy, independent of the actual bias/noise ratio) is
# unchanged; what changed is that static stopped being a strictly-better
# reference to be dominated BY at this particular theta.
# ═══════════════════════════════════════════════════════════════════════════

N_Q = 6
sim = de.DenseSVSimulator(n_qubits=N_Q, use_gpu=False, use_float32=False)
N_BONDS = N_Q - 1                  # 5
N_PARAMS = 2 * N_BONDS              # 10
B_COEFF = -1.0                       # theta_B = -theta_A, same ansatz as elsewhere in this repo

BASE_P = 0.06  # same per-noise-scale-1 probability convention as zne_mitigation.py


def _base_row(theta: float) -> np.ndarray:
    row = np.empty(N_PARAMS, dtype=np.float64)
    row[0::2] = theta
    row[1::2] = B_COEFF * theta
    return row


def _kinetic_from_sv(sv: np.ndarray) -> float:
    """Same open-chain <XX+YY> kinetic term used throughout this repo's
    molecular VQE scripts."""
    dim = len(sv)
    idx = np.arange(dim)
    kinetic = 0.0
    for q in range(N_BONDS):
        mask = (1 << q) | (1 << (q + 1))
        pf = sv[idx ^ mask]
        xx = np.real(np.sum(np.conj(sv) * pf))
        bi = (idx & (1 << q)) >> q
        bj = (idx & (1 << (q + 1))) >> (q + 1)
        yy = np.real(np.sum(np.conj(sv) * pf * np.where(bi == bj, -1.0, 1.0)))
        kinetic += xx + yy
    return float(kinetic)


def _run_circuit_direct(param_row: np.ndarray) -> np.ndarray:
    """Run the staircase Givens circuit with an explicit N_PARAMS-length
    gate-value vector (not batched -- one direct circuit execution), return
    the clean (noise-free) statevector."""
    ops = [['x', 0]]
    for q in range(N_BONDS):
        ops += [['cx', q + 1, q], ['ry', q + 1, float(param_row[2 * q])], ['cx', q, q + 1],
                ['ry', q + 1, float(param_row[2 * q + 1])], ['cx', q + 1, q]]
    sim.set_initial_state()
    sim.run_circuit_jit_beast_mode(ops)
    return np.asarray(sim.get_statevector()).copy()


def _noisy_kinetic_samples(clean_sv: np.ndarray, p_error: float, n_shots: int, rng: np.random.Generator) -> np.ndarray:
    """Raw per-shot noisy kinetic samples (not just their mean) -- needed to
    compute the SEM confidence indicator, not just the averaged estimate."""
    samples = np.empty(n_shots)
    for i in range(n_shots):
        noisy_sv = NoiseModel.apply_to_sv(clean_sv, n=N_Q, model='depolarizing', p=p_error, rng=rng)
        samples[i] = _kinetic_from_sv(noisy_sv)
    return samples


def _sem(samples: np.ndarray) -> float:
    """Standard error of the mean -- a real, measured confidence indicator
    for a shot-averaged estimate (unlike the synthetic 'sigma' proxy used in
    the scratch prototype this generalizes)."""
    return float(np.std(samples, ddof=1) / np.sqrt(len(samples)))


def noisy_kinetic_averaged(clean_sv: np.ndarray, p_error: float, n_shots: int, rng: np.random.Generator) -> float:
    """Shot-averaged noisy kinetic expectation at a given noise probability."""
    return float(np.mean(_noisy_kinetic_samples(clean_sv, p_error, n_shots, rng)))


def zne_corrected_kinetic(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """Static 2-point Richardson ZNE (same formula as zne_mitigation.py /
    zne_stabilized_psr_gradient.py), for direct comparison against the
    adaptive version below."""
    k1 = noisy_kinetic_averaged(clean_sv, base_p * 1.0, n_shots, rng)
    k2 = noisy_kinetic_averaged(clean_sv, base_p * 2.0, n_shots, rng)
    return 2.0 * k1 - k2


def adaptive_zne_corrected_kinetic(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator,
                                    target_sigma_ideal: float, k_sensitivity: float) -> float:
    """Confidence-attenuated 2-point Richardson: computes the SEM of the
    scale=1.0 shot samples, converts it to a normalized deviation-from-ideal
    via calculate_delta_preemp, and attenuates the Richardson correction
    strength accordingly (full correction at delta_p=0, falling back to the
    raw scale=1.0 estimate as delta_p grows)."""
    samples_1 = _noisy_kinetic_samples(clean_sv, base_p * 1.0, n_shots, rng)
    k1 = float(np.mean(samples_1))
    sem_1 = _sem(samples_1)
    k2 = noisy_kinetic_averaged(clean_sv, base_p * 2.0, n_shots, rng)

    delta_p = float(calculate_delta_preemp(sem_1, target_sigma_ideal))
    attenuation = min(1.0, k_sensitivity * delta_p)
    c1 = 2.0 - attenuation
    c2 = -1.0 + attenuation
    return c1 * k1 + c2 * k2


def _chain_rule_gradient(kinetic_plus: np.ndarray, kinetic_minus: np.ndarray) -> float:
    """Combines N_PARAMS individual single-gate PSR partial derivatives
    into the exact chain-rule gradient w.r.t. the shared scalar theta."""
    grad = 0.0
    for k in range(N_PARAMS):
        partial = 0.5 * (kinetic_plus[k] - kinetic_minus[k])
        dtheta_dt = 1.0 if (k % 2 == 0) else B_COEFF
        grad += partial * dtheta_dt
    return grad


def exact_kinetic_gradient(theta: float) -> float:
    """Exact, noise-free chain-rule PSR gradient -- ground truth."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = _kinetic_from_sv(_run_circuit_direct(plus))
        k_minus[k] = _kinetic_from_sv(_run_circuit_direct(minus))
    return _chain_rule_gradient(k_plus, k_minus)


def psr_gradient_naive_noisy(theta: float, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """Chain-rule PSR gradient with NO ZNE correction -- the naive baseline."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = noisy_kinetic_averaged(_run_circuit_direct(plus), base_p, n_shots, rng)
        k_minus[k] = noisy_kinetic_averaged(_run_circuit_direct(minus), base_p, n_shots, rng)
    return _chain_rule_gradient(k_plus, k_minus)


def psr_gradient_zne_static(theta: float, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """Chain-rule PSR gradient with the STATIC 2-point Richardson correction
    (already merged in zne_stabilized_psr_gradient.py) -- reproduced here
    for a direct, same-script three-way comparison."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = zne_corrected_kinetic(_run_circuit_direct(plus), base_p, n_shots, rng)
        k_minus[k] = zne_corrected_kinetic(_run_circuit_direct(minus), base_p, n_shots, rng)
    return _chain_rule_gradient(k_plus, k_minus)


def psr_gradient_adaptive_zne(theta: float, base_p: float, n_shots: int, rng: np.random.Generator,
                               target_sigma_ideal: float, k_sensitivity: float) -> float:
    """Chain-rule PSR gradient with the confidence-ATTENUATED Richardson
    correction -- the new adaptive version under study."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = adaptive_zne_corrected_kinetic(_run_circuit_direct(plus), base_p, n_shots, rng,
                                                    target_sigma_ideal, k_sensitivity)
        k_minus[k] = adaptive_zne_corrected_kinetic(_run_circuit_direct(minus), base_p, n_shots, rng,
                                                     target_sigma_ideal, k_sensitivity)
    return _chain_rule_gradient(k_plus, k_minus)


THETAS = [0.2, 0.38, 0.62, 1.0]
N_TRIALS = 40
N_SHOTS = 200
TARGET_SIGMA_IDEAL = 0.03   # see HONEST FINDING above: no calibration tried dominates on
                             # every theta: this is the most balanced of the 3 explored
                             # (target, k) pairs, not a "solution" -- a compromise point
K_SENSITIVITY = 2.0


def _rmse(vals: np.ndarray, exact: float) -> float:
    bias = vals.mean() - exact
    return float(np.sqrt(bias ** 2 + vals.std() ** 2))


def _stable_seed(*parts) -> int:
    """Deterministic, cross-process-stable seed -- Python's built-in hash()
    is NOT stable across process invocations for tuples containing strings
    (hash randomization, PYTHONHASHSEED defaults to random), so trial data
    seeded via hash((tag, theta, trial)) % 2**32 silently differed between
    runs despite looking deterministic. zlib.crc32 has no such
    randomization."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 32)


def _run_stabilization_study():
    print("============================================================")
    print("ADAPTIVE ZNE-PRE-PSR GRADIENT STABILIZATION STUDY")
    print(f"   base_p={BASE_P}, n_shots={N_SHOTS}, n_trials={N_TRIALS}, "
          f"target_sigma_ideal={TARGET_SIGMA_IDEAL}, k_sensitivity={K_SENSITIVITY}")
    print("============================================================")

    rows = []
    for theta in THETAS:
        exact = exact_kinetic_gradient(theta)

        naive_vals = np.empty(N_TRIALS)
        static_vals = np.empty(N_TRIALS)
        adaptive_vals = np.empty(N_TRIALS)
        for trial in range(N_TRIALS):
            rng_naive = np.random.default_rng(_stable_seed("naive", theta, trial))
            naive_vals[trial] = psr_gradient_naive_noisy(theta, BASE_P, N_SHOTS, rng_naive)
            rng_static = np.random.default_rng(_stable_seed("static", theta, trial))
            static_vals[trial] = psr_gradient_zne_static(theta, BASE_P, N_SHOTS, rng_static)
            rng_adaptive = np.random.default_rng(_stable_seed("adaptive", theta, trial))
            adaptive_vals[trial] = psr_gradient_adaptive_zne(theta, BASE_P, N_SHOTS, rng_adaptive,
                                                              TARGET_SIGMA_IDEAL, K_SENSITIVITY)

        row = {
            "Theta": theta,
            "Gradiente_Esatto": exact,
            "Naive_Media": naive_vals.mean(), "Naive_Std": naive_vals.std(),
            "Naive_Bias": naive_vals.mean() - exact, "RMSE_Naive": _rmse(naive_vals, exact),
            "Statico_Media": static_vals.mean(), "Statico_Std": static_vals.std(),
            "Statico_Bias": static_vals.mean() - exact, "RMSE_Statico": _rmse(static_vals, exact),
            "Adattivo_Media": adaptive_vals.mean(), "Adattivo_Std": adaptive_vals.std(),
            "Adattivo_Bias": adaptive_vals.mean() - exact, "RMSE_Adattivo": _rmse(adaptive_vals, exact),
        }
        rows.append(row)
        print(f"theta={theta:.2f} | esatto={exact:+.5f}")
        print(f"   naive:    media={row['Naive_Media']:+.5f} std={row['Naive_Std']:.5f} "
              f"bias={row['Naive_Bias']:+.5f} rmse={row['RMSE_Naive']:.5f}")
        print(f"   statico:  media={row['Statico_Media']:+.5f} std={row['Statico_Std']:.5f} "
              f"bias={row['Statico_Bias']:+.5f} rmse={row['RMSE_Statico']:.5f}")
        print(f"   adattivo: media={row['Adattivo_Media']:+.5f} std={row['Adattivo_Std']:.5f} "
              f"bias={row['Adattivo_Bias']:+.5f} rmse={row['RMSE_Adattivo']:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "zne_adaptive_psr_gradient.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(THETAS))
    width = 0.25
    ax.bar(x - width, df["RMSE_Naive"], width=width, color='#888888', label='PSR ingenuo')
    ax.bar(x, df["RMSE_Statico"], width=width, color='#FF007F', label='ZNE statico (PR #6)')
    ax.bar(x + width, df["RMSE_Adattivo"], width=width, color='#00FFFF', label='ZNE adattivo (delta_preemp)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in THETAS])
    ax.set_ylabel("RMSE vs. gradiente esatto")
    ax.set_title("RMSE: ingenuo vs. ZNE statico vs. ZNE adattivo", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "zne_adaptive_psr_gradient.png", dpi=300)

    print("============================================================")
    print("COMPLETATO")
    print(df.to_string(index=False))
    print("============================================================")


if __name__ == "__main__":
    _run_stabilization_study()
