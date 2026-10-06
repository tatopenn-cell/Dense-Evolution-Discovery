"""Re-check of the coherence-L1 ZNE validation (#208) with the controls that
exposed a confound in the single-qubit version (single_qubit_coherence_zne.py).

Same setup as jsd_zne_noise_generalization.py Part 4: GHZ(4), phase flip,
base_p = 0.05, noise factors 1, 2, 3, 150 stochastic trajectories per scale,
seeds 900000 + i, Uhlmann fidelity against the ideal state, baseline = the same
extrapolation with nudge_scale = 0 followed by the physical projection.

Three ways to set the nudge on the same density matrices:
- signal: rectified coherence nonlinearity, the promoted method;
- shuffled: the same nudge values reassigned at random across seeds;
- constant: the mean active nudge applied to every seed.
If the signal carries the gain, shuffled and constant must both do worse.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from dense_evolution.config import ensure_x64
from dense_evolution.mitigation import project_to_physical, uhlmann_fidelity
from dense_evolution.mitigation.zne import _coherence_l1
from jsd_zne_noise_generalization import (FACTORS, N_QUBITS, N_TRIALS, build_ghz_statevector,
                                          density_matrix_stochastic)

SEEDS = 200
BASE_P = 0.05


def rectified(rhos):
    c = [float(jnp.real(_coherence_l1(rhos[i]))) for i in range(3)]
    g12, g23 = abs(c[0] - c[1]), abs(c[1] - c[2])
    return max((g23 - g12) / (g23 + g12 + 1e-12), 0.0)


def extrapolate(rhos, rect, nudge_scale=0.5):
    c = jnp.array([3 - nudge_scale * rect, -3 + 2 * nudge_scale * rect, 1 - nudge_scale * rect])
    return project_to_physical(jnp.tensordot(c, rhos, 1) / jnp.sum(c))


def main():
    ensure_x64()
    sv = build_ghz_statevector(N_QUBITS)
    rho_ideal = jnp.outer(sv, jnp.conj(sv))
    stacks, rects = [], []
    for seed in range(SEEDS):
        key = jax.random.PRNGKey(900000 + seed)
        rhos = []
        for f in FACTORS:
            key, sub = jax.random.split(key)
            rhos.append(density_matrix_stochastic(sv, N_QUBITS, "phaseflip", BASE_P, f, N_TRIALS, sub))
        rhos = jnp.stack(rhos)
        stacks.append(rhos)
        rects.append(rectified(rhos))
    rects = np.array(rects)
    rng = np.random.default_rng(208)
    shuffled = rng.permutation(rects)
    constant = np.full(SEEDS, rects[rects > 0].mean())

    def fid(rect_of):
        return np.array([float(uhlmann_fidelity(rho_ideal, extrapolate(s, r))) for s, r in zip(stacks, rect_of)])

    base = fid(np.zeros(SEEDS))
    print(f"active seeds (signal > 0): {(rects > 0).sum()}/{SEEDS}, mean active nudge {rects[rects > 0].mean():.3f}")
    print(f"{'nudge':<10}{'mean gain':>12}{'better/worse':>15}{'t-test p':>11}")
    gains = {}
    for name, rect_of in (("signal", rects), ("shuffled", shuffled), ("constant", constant)):
        d = fid(rect_of) - base
        gains[name] = d
        nz = d[np.abs(d) > 1e-12]
        p = stats.ttest_1samp(nz, 0.0).pvalue if len(nz) > 1 else float("nan")
        print(f"{name:<10}{d.mean():>12.5f}{int((d > 1e-12).sum()):>9}/{int((d < -1e-12).sum()):<5}{p:>11.2e}")
    diff = gains["signal"] - gains["shuffled"]
    print(f"signal - shuffled: mean {diff.mean():+.5f}, paired t-test p {stats.ttest_1samp(diff, 0.0).pvalue:.2e}")
    diff = gains["signal"] - gains["constant"]
    print(f"signal - constant: mean {diff.mean():+.5f}, paired t-test p {stats.ttest_1samp(diff, 0.0).pvalue:.2e}")


if __name__ == "__main__":
    main()
