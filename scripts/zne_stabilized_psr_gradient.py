import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dense_evolution as de
from dense_evolution.registry import NoiseModel

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# ZNE-BEFORE-PSR: does correcting each individual single-gate +pi/2 / -pi/2
# shifted energy with Zero-Noise Extrapolation, BEFORE combining them via the
# Parameter-Shift-Rule (PSR) chain rule, stabilize the resulting gradient
# under NoiseModel -- as opposed to computing a raw noisy PSR gradient first
# and correcting that afterwards?
#
# Isolated on the simplest ansatz in this repo (the single shared-theta
# staircase Givens circuit from vqe_silicon_molecular_optimized.py, N_Q=6,
# replicated locally here -- scripts in this repo don't import each other,
# only tests do) rather than the per-bond extreme-geometry benchmark: that
# ansatz already has an exact, independently-verified PSR gradient (README
# Section 9b, finite differences to ~1e-9) to use as ground truth, and its
# small statevector (64 amplitudes) keeps this noise+shot-averaging study
# cheap and easy to reason about.
#
# CAUTION -- an earlier version of this script shifted the shared SCALAR
# theta by +-pi/2 directly, exactly the mistake already documented and fixed
# elsewhere in this repo (README Section 6, "Correction, audit finding"):
# each bond drives TWO gates from theta (theta_A=theta, theta_B=-theta), so
# PSR (only exact for a single gate's own parameter) must shift each of the
# N_PARAMS=2*N_BONDS individual gate values one at a time and recombine via
# the chain rule (dtheta_A/dtheta=1, dtheta_B/dtheta=B_COEFF=-1) -- exactly
# what vqe_silicon_molecular_optimized.py's batched_kinetic_and_exact_gradient
# already does correctly. A finite-difference test (which shifts the scalar
# directly -- valid for a classical derivative, unlike scalar-shift PSR)
# caught the bug: the broken version gave a completely different gradient
# (wrong magnitude AND sign region). Fixed here by adopting the same
# per-gate PSR pattern.
#
# NoiseModel.apply_to_sv is ALWAYS post-hoc on an already-computed clean
# statevector (verified by reading dense_evolution/simulator.py in full --
# no noise/RNG anywhere in the unitary simulation path). One call is one
# stochastic Kraus realization ("one shot"); estimating an expectation value
# needs averaging over many shots, same convention as zne_mitigation.py's
# NUM_SHOTS-based averaging. Because noise is applied to an ALREADY-COMPUTED
# statevector, shot-averaging here is cheap: each of the N_PARAMS individual
# gate-shifted circuits only runs once, and every shot just reapplies
# NoiseModel.apply_to_sv (a plain numpy op on a 64-length array) to a fresh
# copy of that same clean state. "ZNE before PSR" now means: ZNE-correct
# each of the N_PARAMS individual single-gate PSR terms BEFORE summing them
# via the chain rule -- the correct atomic level to apply it, since PSR
# itself is only exact at that same single-gate level.
#
# ZNE correction used: the static 2-point Richardson formula already
# validated in zne_mitigation.py, E_zne = 2*E(scale=1.0) - E(scale=2.0) --
# not the adaptive delta_preemp-weighted variant (that needs a per-shift
# "sigma" metric that doesn't exist yet; deliberately out of scope here to
# avoid mixing two new design questions at once).
#
# HONEST FINDING -- NOT what a naive "ZNE = more stable" intuition predicts,
# and NOT uniform across theta either. Measured (40 trials, 200 shots,
# base_p=0.06) after the per-gate-PSR fix above:
#
#   theta  exact     naive: bias    std     rmse    zne: bias    std     rmse
#   0.20   +7.354    -3.117  0.072  3.118    -1.387  0.142  1.394
#   0.38   +5.071    -2.164  0.113  2.167    -1.021  0.202  1.041
#   0.62   -0.127    +0.015  0.114  0.115    +0.053  0.256  0.261
#   1.00   -3.323    +1.409  0.121  1.414    +0.666  0.262  0.716
#
# Away from a gradient zero-crossing (theta=0.20, 0.38, 1.00), ZNE-pre-PSR
# cuts the bias roughly in half to a third and, despite ~2x higher std
# (the textbook Richardson bias/variance tradeoff -- 2*E1-E2 amplifies
# statistical noise in exchange for cancelling the leading systematic
# error), nets a clearly lower RMSE (roughly 2-2.2x better).
#
# BUT at theta=0.62, where the exact gradient is itself near zero, ZNE-pre-
# PSR is WORSE on every axis: bias +0.053 vs naive's +0.015, std 0.256 vs
# 0.114, RMSE 0.261 vs 0.115. Near a zero-crossing there is very little
# systematic bias left to correct, so Richardson's variance amplification
# has nothing to buy against -- it just adds noise. "ZNE stabilizes the
# gradient" is therefore a REGIME-DEPENDENT claim, true where the true
# gradient magnitude is large relative to the noise level, not a universal
# property of ZNE-pre-PSR. See tests/test_zne_stabilized_psr_gradient.py
# for both the "wins away from zero" and "loses near zero" as explicit,
# separate regression guards -- do not collapse this into one blanket claim.
#
# OUT OF SCOPE for this script (explicit, for a later step): hooking this
# noisy/ZNE-corrected gradient into an actual Adam optimization loop; noise
# models other than depolarizing; the adaptive delta_preemp Richardson
# variant; whether increasing shot count for the ZNE side specifically
# could close or reverse its variance disadvantage while keeping the bias
# win (plausible but unverified here).
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


def noisy_kinetic_single_shot(clean_sv: np.ndarray, p_error: float, rng: np.random.Generator) -> float:
    """One stochastic depolarizing-noise realization applied to an
    already-computed clean statevector, then measured -- one "shot"."""
    noisy_sv = NoiseModel.apply_to_sv(clean_sv, n=N_Q, model='depolarizing', p=p_error, rng=rng)
    return _kinetic_from_sv(noisy_sv)


def noisy_kinetic_averaged(clean_sv: np.ndarray, p_error: float, n_shots: int, rng: np.random.Generator) -> float:
    """Shot-averaged noisy kinetic expectation at a given noise probability."""
    return float(np.mean([noisy_kinetic_single_shot(clean_sv, p_error, rng) for _ in range(n_shots)]))


def zne_corrected_kinetic(clean_sv: np.ndarray, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """2-point Richardson ZNE (same formula as zne_mitigation.py) applied to
    ONE gate-shifted evaluation's shot-averaged kinetic, at noise scales 1.0
    and 2.0 (probabilities base_p and 2*base_p)."""
    k1 = noisy_kinetic_averaged(clean_sv, base_p * 1.0, n_shots, rng)
    k2 = noisy_kinetic_averaged(clean_sv, base_p * 2.0, n_shots, rng)
    return 2.0 * k1 - k2


def _chain_rule_gradient(kinetic_plus: np.ndarray, kinetic_minus: np.ndarray) -> float:
    """Combines N_PARAMS individual single-gate PSR partial derivatives
    (kinetic_plus[k], kinetic_minus[k] for gate k shifted by +-pi/2) into
    the exact chain-rule gradient w.r.t. the shared scalar theta."""
    grad = 0.0
    for k in range(N_PARAMS):
        partial = 0.5 * (kinetic_plus[k] - kinetic_minus[k])
        dtheta_dt = 1.0 if (k % 2 == 0) else B_COEFF
        grad += partial * dtheta_dt
    return grad


def exact_kinetic_gradient(theta: float) -> float:
    """Exact, noise-free chain-rule PSR gradient of the kinetic term w.r.t.
    the shared scalar theta -- ground truth, same math already verified
    elsewhere in this repo (vqe_silicon_molecular_optimized.py, agreement
    with finite differences to ~1e-9)."""
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
    """Chain-rule PSR gradient where each of the N_PARAMS individual
    single-gate shifted evaluations is a noisy (scale=1.0) shot-averaged
    kinetic value -- NO ZNE correction. The naive baseline."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = noisy_kinetic_averaged(_run_circuit_direct(plus), base_p, n_shots, rng)
        k_minus[k] = noisy_kinetic_averaged(_run_circuit_direct(minus), base_p, n_shots, rng)
    return _chain_rule_gradient(k_plus, k_minus)


def psr_gradient_zne_stabilized(theta: float, base_p: float, n_shots: int, rng: np.random.Generator) -> float:
    """Chain-rule PSR gradient where each of the N_PARAMS individual
    single-gate shifted evaluations is ZNE-corrected BEFORE being combined
    via the chain rule -- ZNE applied at the same atomic (single-gate)
    level where PSR itself is exact, not after the full gradient is formed."""
    base = _base_row(theta)
    k_plus = np.empty(N_PARAMS)
    k_minus = np.empty(N_PARAMS)
    for k in range(N_PARAMS):
        plus = base.copy(); plus[k] += np.pi / 2
        minus = base.copy(); minus[k] -= np.pi / 2
        k_plus[k] = zne_corrected_kinetic(_run_circuit_direct(plus), base_p, n_shots, rng)
        k_minus[k] = zne_corrected_kinetic(_run_circuit_direct(minus), base_p, n_shots, rng)
    return _chain_rule_gradient(k_plus, k_minus)


THETAS = [0.2, 0.38, 0.62, 1.0]
N_TRIALS = 40
N_SHOTS = 200


def _run_stabilization_study():
    print("============================================================")
    print("ZNE-PRE-PSR GRADIENT STABILIZATION STUDY")
    print(f"   base_p={BASE_P}, n_shots={N_SHOTS}, n_trials={N_TRIALS}")
    print("============================================================")

    rows = []
    for theta in THETAS:
        exact = exact_kinetic_gradient(theta)

        naive_vals = np.empty(N_TRIALS)
        zne_vals = np.empty(N_TRIALS)
        for trial in range(N_TRIALS):
            rng_naive = np.random.default_rng(hash(("naive", theta, trial)) % (2 ** 32))
            naive_vals[trial] = psr_gradient_naive_noisy(theta, BASE_P, N_SHOTS, rng_naive)
            rng_zne = np.random.default_rng(hash(("zne", theta, trial)) % (2 ** 32))
            zne_vals[trial] = psr_gradient_zne_stabilized(theta, BASE_P, N_SHOTS, rng_zne)

        naive_bias = naive_vals.mean() - exact
        zne_bias = zne_vals.mean() - exact
        row = {
            "Theta": theta,
            "Gradiente_Esatto": exact,
            "Naive_Media": naive_vals.mean(),
            "Naive_Std": naive_vals.std(),
            "Naive_Bias": naive_bias,
            "RMSE_Naive": float(np.sqrt(naive_bias ** 2 + naive_vals.std() ** 2)),
            "ZNE_Media": zne_vals.mean(),
            "ZNE_Std": zne_vals.std(),
            "ZNE_Bias": zne_bias,
            "RMSE_ZNE": float(np.sqrt(zne_bias ** 2 + zne_vals.std() ** 2)),
        }
        rows.append(row)
        print(f"theta={theta:.2f} | esatto={exact:+.5f} | "
              f"naive: media={row['Naive_Media']:+.5f} std={row['Naive_Std']:.5f} bias={row['Naive_Bias']:+.5f} "
              f"rmse={row['RMSE_Naive']:.5f} | "
              f"zne: media={row['ZNE_Media']:+.5f} std={row['ZNE_Std']:.5f} bias={row['ZNE_Bias']:+.5f} "
              f"rmse={row['RMSE_ZNE']:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "zne_stabilized_psr_gradient.csv", index=False)

    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6))

    x = np.arange(len(THETAS))
    ax1.errorbar(x - 0.05, df["Naive_Media"], yerr=df["Naive_Std"], fmt='o', color='#888888',
                 label='PSR ingenuo (rumoroso)', capsize=4)
    ax1.errorbar(x + 0.05, df["ZNE_Media"], yerr=df["ZNE_Std"], fmt='o', color='#FF007F',
                 label='PSR con ZNE pre-differenza', capsize=4)
    ax1.plot(x, df["Gradiente_Esatto"], marker='*', markersize=14, linestyle='none',
             color='#00FFFF', label='Gradiente esatto (no rumore)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"θ={t}" for t in THETAS])
    ax1.set_ylabel("d(cinetica)/dθ")
    ax1.set_title("Gradiente: media ± std tra trial", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax1.legend(loc="best", fontsize=8)

    width = 0.35
    ax2.bar(x - width / 2, df["Naive_Std"], width=width, color='#888888', label='PSR ingenuo')
    ax2.bar(x + width / 2, df["ZNE_Std"], width=width, color='#FF007F', label='PSR con ZNE pre-differenza')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"θ={t}" for t in THETAS])
    ax2.set_ylabel("Deviazione standard tra trial")
    ax2.set_title("Varianza tra trial", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax2.legend(loc="best", fontsize=8)

    ax3.bar(x - width / 2, df["RMSE_Naive"], width=width, color='#888888', label='PSR ingenuo')
    ax3.bar(x + width / 2, df["RMSE_ZNE"], width=width, color='#FF007F', label='PSR con ZNE pre-differenza')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"θ={t}" for t in THETAS])
    ax3.set_ylabel("RMSE vs. gradiente esatto")
    ax3.set_title("Errore complessivo (bias+varianza)", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.2, color='#444444')
    ax3.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "zne_stabilized_psr_gradient.png", dpi=300)

    print("============================================================")
    print("COMPLETATO")
    print(df.to_string(index=False))
    print("============================================================")


if __name__ == "__main__":
    _run_stabilization_study()
