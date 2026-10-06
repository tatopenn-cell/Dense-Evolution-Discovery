"""Single-qubit coherence-L1 nudge for ZNE (issue #243, variant 2).

The library's `coherence_predictive_zne_density_matrix` nudges the 3-point
Richardson coefficients when the l1 coherence (Baumgratz, Cramer, Plenio,
PRL 113, 140401 (2014)) changes non-linearly across the noise scales, then
projects onto a physical state. Its validation (Dense-Evolution-Discovery #208)
used the full GHZ(4) density matrix. Here the same core runs on one qubit: the
2x2 matrix rebuilt from single-qubit tomography (X, Y, Z, K shots each), so the
signal is 2|rho_01| and the cost stays three measurement settings per qubit.

Fair baseline, as in the original validation: the same core with
nudge_scale = 0, i.e. plain Richardson followed by the physical projection
(variant 1, single_qubit_bloch_ball_zne.py). Only seeds where the nudge
activates (rectified nonlinearity > 0) can differ; they are counted as in #208.

Known limit, by construction: in a GHZ state each single-qubit reduced state is
I/2, whose coherence is zero at every scale, so the local signal never fires
there. This script uses Haar-random pure single-qubit states instead.

Control (part 2): the same nudges reassigned at random across seeds, and one
constant nudge applied to every seed. If the coherence signal carried the gain,
both controls would do worse than the signal.
"""
import numpy as np
from scipy import stats

from dense_evolution.config import ensure_x64
from dense_evolution.mitigation import project_to_physical
from dense_evolution.mitigation.zne import _coherence_predictive_zne_density_matrix_core
from single_qubit_bloch_ball_zne import BASE_P, FACTORS, bloch, kraus, rho_of

SEEDS = 200
SHOTS = (100, 400, 1600)


def trial(channel, shots, rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    psi = np.outer(v, v.conj())
    stack = []
    for f in FACTORS:
        r = bloch(sum(K @ psi @ K.conj().T for K in kraus(channel, BASE_P * f)))
        stack.append(rho_of(2 * rng.binomial(shots, (1 + r) / 2) / shots - 1))
    stack = np.array(stack)
    coh = [2 * abs(m[0, 1]) for m in stack]
    g12, g23 = abs(coh[0] - coh[1]), abs(coh[1] - coh[2])
    active = (g23 - g12) / (g23 + g12 + 1e-12) > 0
    plain = np.asarray(_coherence_predictive_zne_density_matrix_core(stack, nudge_scale=0.0))
    nudged = np.asarray(_coherence_predictive_zne_density_matrix_core(stack, nudge_scale=0.5))
    n = bloch(psi)
    return active, (np.linalg.norm(bloch(plain) - n) - np.linalg.norm(bloch(nudged) - n)) / 2


def permutation_p(x, rng, n=20000):
    signs = rng.choice((-1.0, 1.0), size=(n, len(x)))
    return float(np.mean((signs * x).mean(axis=1) >= x.mean()))


def extrapolate(stack, rectified, nudge_scale=0.5):
    c = np.array([3 - nudge_scale * rectified, -3 + 2 * nudge_scale * rectified, 1 - nudge_scale * rectified])
    return np.asarray(project_to_physical(np.tensordot(c, stack, 1) / c.sum()))


def controls(channel, shots, rng, seeds=400):
    stacks, rects, ideal = [], [], []
    for _ in range(seeds):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        psi = np.outer(v, v.conj())
        st = []
        for f in FACTORS:
            r = bloch(sum(K @ psi @ K.conj().T for K in kraus(channel, BASE_P * f)))
            st.append(rho_of(2 * rng.binomial(shots, (1 + r) / 2) / shots - 1))
        st = np.array(st)
        coh = [2 * abs(m[0, 1]) for m in st]
        g12, g23 = abs(coh[0] - coh[1]), abs(coh[1] - coh[2])
        stacks.append(st)
        rects.append(max((g23 - g12) / (g23 + g12 + 1e-12), 0.0))
        ideal.append(bloch(psi))
    rects = np.array(rects)
    shuffled = rng.permutation(rects)
    constant = rects[rects > 0].mean()

    def err(rect_of):
        return np.array([np.linalg.norm(bloch(extrapolate(s, r)) - n) / 2 for s, r, n in zip(stacks, rect_of, ideal)])

    base = err(np.zeros(seeds))
    return (base - err(rects)).mean(), (base - err(shuffled)).mean(), (base - err(np.full(seeds, constant))).mean()


def main():
    ensure_x64()
    rng = np.random.default_rng(2432)
    print(f"{'channel':<18}{'shots':>6}{'active':>9}{'mean gain (active)':>20}{'better/worse':>14}{'t-test p':>11}{'perm p':>9}")
    for channel in ("phase_flip", "amplitude_damping", "depolarizing"):
        for shots in SHOTS:
            res = [trial(channel, shots, rng) for _ in range(SEEDS)]
            d = np.array([g for a, g in res if a])
            better, worse = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            t = stats.ttest_1samp(d, 0.0).pvalue if len(d) > 1 else float("nan")
            perm = permutation_p(d, rng) if len(d) > 1 else float("nan")
            print(f"{channel:<18}{shots:>6}{len(d):>5}/{SEEDS}{d.mean() if len(d) else float('nan'):>20.5f}"
                  f"{better:>8}/{worse:<5}{t:>11.2e}{perm:>9.4f}")
    print()
    print("Control, 400 seeds, mean gain over all seeds vs nudge_scale = 0:")
    print(f"{'channel':<18}{'shots':>6}{'signal':>10}{'shuffled':>10}{'constant':>10}")
    for channel in ("phase_flip", "amplitude_damping", "depolarizing"):
        for shots in SHOTS:
            sig, shu, con = controls(channel, shots, rng)
            print(f"{channel:<18}{shots:>6}{sig:>10.5f}{shu:>10.5f}{con:>10.5f}")


if __name__ == "__main__":
    main()
