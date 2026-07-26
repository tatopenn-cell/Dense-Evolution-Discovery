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
# ADAPTIVE ZNE v2 -- confidence from the CORRECTION TERM's own SNR, not the
# raw noise of a single measurement.
#
# scripts/zne_adaptive_psr_gradient.py (merged, negative result) attenuated
# the static Richardson correction (zne_stabilized_psr_gradient.py) using
# the standard error of the mean (SEM) of the scale=1.0 estimate as a
# confidence signal. That failed: SEM sits in the same narrow band
# regardless of theta, so it carries no information about proximity to a
# gradient zero-crossing -- the actual thing that determines whether
# Richardson correction helps or hurts.
#
# This script tries the more promising direction flagged there: the
# correction term (k1 - k2) itself, and whether it is STATISTICALLY
# RESOLVED relative to its own combined noise, or indistinguishable from
# noise. Richardson extrapolation implicitly assumes (k1 - k2) is a
# meaningful estimate of the noise-scaling slope; if that difference is
# itself dominated by shot noise, extrapolating it just amplifies garbage.
#
#   sem_1 = SEM(scale=1.0 samples), sem_2 = SEM(scale=2.0 samples)
#   sem_diff = sqrt(sem_1^2 + sem_2^2)        # combined error of (k1 - k2), independent draws
#   snr = |k1 - k2| / sem_diff                 # "how many standard errors" is the correction
#
# Unlike SEM alone (confidence GROWS as SEM shrinks), confidence here GROWS
# as SNR grows -- the opposite direction. Reusing
# dense_evolution.healing.calculate_delta_preemp honestly (not forcing it
# through the wrong end like the caution note in the adaptive-v1 script
# warns against) requires clamping first:
#
#   current_sigma_clamped = min(snr, SNR_TARGET)
#   delta_p = calculate_delta_preemp(current_sigma_clamped, SNR_TARGET)
#           = (SNR_TARGET - min(snr, SNR_TARGET)) / SNR_TARGET   # in [0,1], DEcreasing in snr
#   attenuation = min(1.0, K_SENSITIVITY * delta_p)
#   c1 = 2.0 - attenuation
#   c2 = -1.0 + attenuation
#   E_adaptive = c1*k1 + c2*k2
#
# At snr >= SNR_TARGET (correction is a well-resolved, "statistically
# significant" effect) this is EXACTLY the static correction. At snr=0
# (correction indistinguishable from noise) it falls back fully to the raw
# scale=1.0 estimate, same fallback behavior as adaptive-v1, different (and
# better-motivated) trigger condition.
#
# HONEST FINDING -- HYPOTHESIS REJECTED. Modulating the correction's
# attenuation from a fixed SNR threshold does not deliver a clean win over
# the static calibration from zne_stabilized_psr_gradient.py, and the
# algorithm is hyper-sensitive to SNR_TARGET.
#
# Full budget (40 trials, 200 shots), RMSE vs. the exact gradient:
#
#   theta   naive    static   SNR-adaptive (target=3.0, k=1.0)
#   0.20    3.151    1.380    1.421   (slightly worse than static)
#   0.38    2.178    0.920    0.862   (BEATS static -- a genuine win)
#   0.62    0.107    0.263    0.307   (worse than static -- the critical
#                                      case is NOT fixed, it regresses)
#   1.00    1.427    0.673    0.665   (essentially tied with static)
#
# Raising SNR_TARGET to 6.0 (small-budget sweep, 15 trials/60 shots) does
# NOT selectively fix theta=0.62 -- it makes EVERY theta worse, including
# the critical one (0.62: 0.680 vs. static's 0.510 at that budget), because
# more gate-shifts fall below the higher threshold and get attenuated
# everywhere, not just where it would help. Increasing the threshold moves
# the whole profile in the wrong direction; it does not isolate the
# zero-crossing case.
#
# WHY (verified directly, not the recursive-adaptive-filter framing this
# might superficially resemble -- there is no time-varying state, gain
# history, or feedback loop here; snr_adaptive_zne_corrected_kinetic is
# memoryless, recomputed from scratch on every call, so "persistent
# excitation" / "covariance wind-up" in the classical adaptive-control
# sense do not literally apply): SNR is computed from the difference
# between two NOISE SCALES at the SAME gate-shift (k1 at p, k2 at 2p). The
# PSR gradient near theta=0.62 is small because of a near-cancellation
# between two GATE-SHIFTS at the SAME noise scale (k_plus - k_minus across
# the +-pi/2 shift). These are two different differences of two different
# quantities -- there's no causal reason the first should be small when
# the second is. On top of that, |k1-k2| is a biased estimator of the true
# noise-scaling difference: the expectation of an absolute value of a
# noisy, near-zero-mean quantity does not go to zero as the true value
# does (it's bounded below by something proportional to the noise itself,
# a Rayleigh-like floor) -- which is exactly why SNR was measured at a
# misleadingly high ~3.3-6.6 at EVERY theta tested, including 0.62, instead
# of dropping there as the original hypothesis expected.
#
# DEVELOPMENT HALTED on this SNR-threshold direction per this finding.
# Future adaptive-ZNE work returns to refining the SEM-based model
# (zne_adaptive_psr_gradient.py), which -- despite never winning outright
# either -- specifically improved on static AT the critical theta=0.62 case
# (0.175 vs. static's 0.248), something this SNR-based attempt did not
# reproduce (it made that case worse, 0.307 vs. 0.263).
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
    """Raw per-shot noisy kinetic samples -- needed to compute both the
    mean and the SEM of each noise-scale estimate."""
    samples = np.empty(n_shots)
    for i in range(n_shots):
        noisy_sv = NoiseModel.apply_to_sv(clean_sv, n=N_Q, model='depolarizing', p=p_error, rng=rng)
        samples[i] = _kinetic_from_sv(noisy_sv)
    return samples


def _sem(samples: np.ndarray) -> float:
    return float(np.std(samples, ddof=1) / np.sqrt(len(samples)))


def noisy_kinetic_averaged(clean_sv: np.ndarray, p_error: float, n_shots: int, rng: np.random.Generator) -> float:
    return float(np.mean(_noisy_kinetic_samples(clean_sv, p_error, n_shots, rng)))


def zne_corrected_kinetic(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """Static 2-point Richardson ZNE, for direct comparison."""
    k1 = noisy_kinetic_averaged(clean_sv, base_p * 1.0, n_shots, rng)
    k2 = noisy_kinetic_averaged(clean_sv, base_p * 2.0, n_shots, rng)
    return 2.0 * k1 - k2


def _correction_snr(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator):
    """Returns (snr, k1, k2): the correction term's own signal-to-noise
    ratio, |k1-k2| relative to the combined SEM of the two independent
    noise-scale estimates."""
    samples_1 = _noisy_kinetic_samples(clean_sv, base_p * 1.0, n_shots, rng)
    samples_2 = _noisy_kinetic_samples(clean_sv, base_p * 2.0, n_shots, rng)
    k1 = float(np.mean(samples_1))
    k2 = float(np.mean(samples_2))
    sem_diff = float(np.sqrt(_sem(samples_1) ** 2 + _sem(samples_2) ** 2))
    snr = abs(k1 - k2) / sem_diff if sem_diff > 0 else 0.0
    return snr, k1, k2


def snr_adaptive_zne_corrected_kinetic(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator,
                                        snr_target: float, k_sensitivity: float) -> float:
    """Confidence-attenuated 2-point Richardson where confidence GROWS with
    the correction term's own SNR (opposite direction from adaptive-v1's
    SEM-based signal). At snr>=snr_target: no attenuation (matches static).
    At snr=0: full attenuation (falls back to the raw scale=1.0 estimate)."""
    snr, k1, k2 = _correction_snr(clean_sv, base_p, n_shots, rng)
    current_sigma_clamped = min(snr, snr_target)
    delta_p = float(calculate_delta_preemp(current_sigma_clamped, snr_target))
    attenuation = min(1.0, k_sensitivity * delta_p)
    c1 = 2.0 - attenuation
    c2 = -1.0 + attenuation
    return c1 * k1 + c2 * k2


def _chain_rule_gradient(kinetic_plus: np.ndarray, kinetic_minus: np.ndarray) -> float:
    grad = 0.0
    for k in range(N_PARAMS):
        partial = 0.5 * (kinetic_plus[k] - kinetic_minus[k])
        dtheta_dt = 1.0 if (k % 2 == 0) else B_COEFF
        grad += partial * dtheta_dt
    return grad


def exact_kinetic_gradient(theta: float) -> float:
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
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = zne_corrected_kinetic(_run_circuit_direct(plus), base_p, n_shots, rng)
        k_minus[k] = zne_corrected_kinetic(_run_circuit_direct(minus), base_p, n_shots, rng)
    return _chain_rule_gradient(k_plus, k_minus)


def psr_gradient_snr_adaptive_zne(theta: float, base_p: float, n_shots: int, rng: np.random.Generator,
                                   snr_target: float, k_sensitivity: float) -> float:
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = snr_adaptive_zne_corrected_kinetic(_run_circuit_direct(plus), base_p, n_shots, rng,
                                                        snr_target, k_sensitivity)
        k_minus[k] = snr_adaptive_zne_corrected_kinetic(_run_circuit_direct(minus), base_p, n_shots, rng,
                                                         snr_target, k_sensitivity)
    return _chain_rule_gradient(k_plus, k_minus)


THETAS = [0.2, 0.38, 0.62, 1.0]
N_TRIALS = 40
N_SHOTS = 200
SNR_TARGET = 3.0        # calibrated -- see HONEST FINDING above (raising to 6.0 makes every theta worse)
K_SENSITIVITY = 1.0


def _rmse(vals: np.ndarray, exact: float) -> float:
    bias = vals.mean() - exact
    return float(np.sqrt(bias ** 2 + vals.std() ** 2))


def _stable_seed(*parts) -> int:
    """Deterministic, cross-process-stable seed -- Python's built-in hash()
    is NOT stable across process invocations for tuples containing strings
    (hash randomization, PYTHONHASHSEED defaults to random), so a previous
    version of this function using hash((tag, theta, trial)) % 2**32 gave
    different trial data on every fresh run despite looking deterministic.
    zlib.crc32 has no such randomization."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 32)


def _run_stabilization_study():
    print("============================================================")
    print("SNR-ADAPTIVE ZNE-PRE-PSR GRADIENT STABILIZATION STUDY")
    print(f"   base_p={BASE_P}, n_shots={N_SHOTS}, n_trials={N_TRIALS}, "
          f"snr_target={SNR_TARGET}, k_sensitivity={K_SENSITIVITY}")
    print("============================================================")

    rows = []
    for theta in THETAS:
        exact = exact_kinetic_gradient(theta)

        naive_vals = np.empty(N_TRIALS)
        static_vals = np.empty(N_TRIALS)
        snr_adaptive_vals = np.empty(N_TRIALS)
        for trial in range(N_TRIALS):
            rng_naive = np.random.default_rng(_stable_seed("naive", theta, trial))
            naive_vals[trial] = psr_gradient_naive_noisy(theta, BASE_P, N_SHOTS, rng_naive)
            rng_static = np.random.default_rng(_stable_seed("static", theta, trial))
            static_vals[trial] = psr_gradient_zne_static(theta, BASE_P, N_SHOTS, rng_static)
            rng_snr = np.random.default_rng(_stable_seed("snr", theta, trial))
            snr_adaptive_vals[trial] = psr_gradient_snr_adaptive_zne(
                theta, BASE_P, N_SHOTS, rng_snr, SNR_TARGET, K_SENSITIVITY)

        row = {
            "Theta": theta, "Gradiente_Esatto": exact,
            "Naive_Media": naive_vals.mean(), "Naive_Std": naive_vals.std(), "RMSE_Naive": _rmse(naive_vals, exact),
            "Statico_Media": static_vals.mean(), "Statico_Std": static_vals.std(), "RMSE_Statico": _rmse(static_vals, exact),
            "SNR_Adattivo_Media": snr_adaptive_vals.mean(), "SNR_Adattivo_Std": snr_adaptive_vals.std(),
            "RMSE_SNR_Adattivo": _rmse(snr_adaptive_vals, exact),
        }
        rows.append(row)
        print(f"theta={theta:.2f} | esatto={exact:+.5f} | rmse naive={row['RMSE_Naive']:.4f} "
              f"statico={row['RMSE_Statico']:.4f} snr_adattivo={row['RMSE_SNR_Adattivo']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "zne_snr_adaptive_psr_gradient.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(THETAS))
    width = 0.25
    ax.bar(x - width, df["RMSE_Naive"], width=width, color='#888888', label='PSR ingenuo')
    ax.bar(x, df["RMSE_Statico"], width=width, color='#FF007F', label='ZNE statico (PR #6)')
    ax.bar(x + width, df["RMSE_SNR_Adattivo"], width=width, color='#00FF88', label='ZNE adattivo-SNR (nuovo)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t}" for t in THETAS])
    ax.set_ylabel("RMSE vs. gradiente esatto")
    ax.set_title("RMSE: ingenuo vs. ZNE statico vs. ZNE adattivo-SNR", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "zne_snr_adaptive_psr_gradient.png", dpi=300)

    print("============================================================")
    print("COMPLETATO")
    print(df.to_string(index=False))
    print("============================================================")


if __name__ == "__main__":
    _run_stabilization_study()
