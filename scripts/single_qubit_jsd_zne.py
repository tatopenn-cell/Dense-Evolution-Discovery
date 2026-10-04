"""Single-qubit population (JSD) nudge for ZNE (issue #243, variant 3).

The library's `jsd_predictive_zne_density_matrix` nudges the 3-point
Richardson coefficients when the Jensen-Shannon divergence of the diagonal
populations changes non-linearly across the noise scales, then projects onto a
physical state; it was validated on amplitude damping (photon loss). Here the
same core runs on one qubit, on the 2x2 matrix rebuilt from single-qubit
tomography (X, Y, Z, K shots each): the signal uses only the Z populations.

Baseline: the same core with nudge_scale = 0 (Richardson plus projection).
Controls, which exposed variant 2 (single_qubit_coherence_zne.py): the same
nudges shuffled across seeds, and one constant nudge on every seed.
"""
import numpy as np
from scipy import stats

from dense_evolution.config import ensure_x64
from dense_evolution.mitigation.zne import _js_divergence, _jsd_predictive_zne_density_matrix_core
from single_qubit_bloch_ball_zne import BASE_P, FACTORS, bloch, kraus, rho_of
from single_qubit_coherence_zne import extrapolate

SEEDS = 400
SHOTS = (100, 400, 1600)


def sample(channel, shots, rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    psi = np.outer(v, v.conj())
    stack = []
    for f in FACTORS:
        r = bloch(sum(K @ psi @ K.conj().T for K in kraus(channel, BASE_P * f)))
        stack.append(rho_of(2 * rng.binomial(shots, (1 + r) / 2) / shots - 1))
    stack = np.array(stack)
    pops = [np.real(np.diag(m)) for m in stack]
    j12, j23 = float(_js_divergence(pops[0], pops[1])), float(_js_divergence(pops[1], pops[2]))
    return stack, max((j23 - j12) / (j23 + j12 + 1e-12), 0.0), bloch(psi)


def main():
    ensure_x64()
    rng = np.random.default_rng(2433)
    print(f"{'channel':<18}{'shots':>6}{'active':>9}{'signal':>10}{'shuffled':>10}{'constant':>10}"
          f"{'better/worse (signal)':>23}{'signal-shuffled p':>19}")
    for channel in ("amplitude_damping", "depolarizing", "phase_flip"):
        for shots in SHOTS:
            runs = [sample(channel, shots, rng) for _ in range(SEEDS)]
            stacks = [s for s, _, _ in runs]
            rects = np.array([r for _, r, _ in runs])
            ideal = [n for _, _, n in runs]
            library = np.array([np.asarray(_jsd_predictive_zne_density_matrix_core(s, nudge_scale=0.5)) for s in stacks])
            assert np.allclose(library, [extrapolate(s, r) for s, r in zip(stacks, rects)], atol=1e-10)
            shuffled = rng.permutation(rects)
            constant = np.full(SEEDS, rects[rects > 0].mean() if (rects > 0).any() else 0.0)

            def err(rect_of):
                return np.array([np.linalg.norm(bloch(extrapolate(s, r)) - n) / 2 for s, r, n in zip(stacks, rect_of, ideal)])

            base = err(np.zeros(SEEDS))
            g_sig, g_shu, g_con = base - err(rects), base - err(shuffled), base - err(constant)
            p = stats.ttest_1samp(g_sig - g_shu, 0.0).pvalue
            print(f"{channel:<18}{shots:>6}{(rects > 0).sum():>5}/{SEEDS}{g_sig.mean():>10.5f}{g_shu.mean():>10.5f}"
                  f"{g_con.mean():>10.5f}{int((g_sig > 1e-12).sum()):>13}/{int((g_sig < -1e-12).sum()):<9}{p:>19.2e}")


if __name__ == "__main__":
    main()
