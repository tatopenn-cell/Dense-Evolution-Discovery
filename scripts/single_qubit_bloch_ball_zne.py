"""Single-qubit density-matrix ZNE with Bloch-ball projection (issue #243, variant 1).

A single-qubit state is fixed by its Bloch vector r = (<X>, <Y>, <Z>), which
single-qubit tomography measures with 3 settings. Zero-noise extrapolation of
each component (3-point Richardson, factors 1, 2, 3) can return |r| > 1, an
unphysical state. For a 2x2 density matrix the eigenvalues are (1 +- |r|)/2, so
projecting them onto the probability simplex (Smolin, Gambetta, Smith,
arXiv:1106.5458) is the same as shrinking r to length 1. That equivalence is
checked here against the library's `project_to_physical`.

Experiment: Haar-random pure states, three channels (depolarizing, amplitude
damping, phase flip) at base_p = 0.05 scaled by the factor, K shots per basis
per scale. Paired comparison at the same shots: plain Richardson vs Richardson
followed by the projection. Metric: trace distance to the ideal state, |r - n|/2,
used only for grading.

The projection is a metric projection onto a convex set (the Bloch ball) that
contains the ideal state, so it can never increase the distance to it; the
question measured here is how often it acts and by how much.

Second baseline, from Miranskyy, Sorrenti, Thind, Gravel (arXiv:2604.24475,
Sect. III.D and VI.A): post hoc clipping of each extrapolated Pauli expectation
to [-1, 1]. Clipping keeps r inside the cube, not the ball, so the state can
still be unphysical; the paired column compares the ball projection with it.
"""
import numpy as np
from scipy import stats

from dense_evolution.config import ensure_x64
from dense_evolution.mitigation import project_to_physical, richardson_extrapolate

FACTORS = (1.0, 2.0, 3.0)
BASE_P = 0.05
SEEDS = 200
SHOTS = (100, 400, 1600)
PAULI = (np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex))


def kraus(channel, p):
    if channel == "depolarizing":
        return [np.sqrt(1 - 3 * p / 4) * np.eye(2)] + [np.sqrt(p / 4) * P for P in PAULI]
    if channel == "amplitude_damping":
        return [np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, np.sqrt(p)], [0, 0]])]
    if channel == "phase_flip":
        return [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * PAULI[2]]
    raise ValueError(channel)


def bloch(rho):
    return np.real([np.trace(rho @ P) for P in PAULI])


def rho_of(r):
    return 0.5 * (np.eye(2) + sum(c * P for c, P in zip(r, PAULI)))


def project_ball(r):
    n = np.linalg.norm(r)
    return r / n if n > 1 else r


def run(channel, shots, rng):
    gains, versus_clip, active = [], [], 0
    for _ in range(SEEDS):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        n_ideal = bloch(np.outer(v, v.conj()))
        estimates = []
        for f in FACTORS:
            rho = sum(K @ np.outer(v, v.conj()) @ K.conj().T for K in kraus(channel, BASE_P * f))
            r = bloch(rho)
            plus = rng.binomial(shots, (1 + r) / 2)
            estimates.append(2 * plus / shots - 1)
        r_zne = np.asarray(richardson_extrapolate(np.array(estimates), np.array(FACTORS)))
        r_proj = project_ball(r_zne)
        r_clip = np.clip(r_zne, -1.0, 1.0)
        active += np.linalg.norm(r_zne) > 1
        gains.append((np.linalg.norm(r_zne - n_ideal) - np.linalg.norm(r_proj - n_ideal)) / 2)
        versus_clip.append((np.linalg.norm(r_clip - n_ideal) - np.linalg.norm(r_proj - n_ideal)) / 2)
    return np.array(gains), np.array(versus_clip), active


def check_equivalence(rng):
    worst = 0.0
    for _ in range(200):
        r = rng.normal(size=3) * rng.uniform(0.2, 1.6)
        lib = np.asarray(project_to_physical(rho_of(r)))
        worst = max(worst, np.abs(lib - rho_of(project_ball(r))).max())
    return worst


def main():
    ensure_x64()
    rng = np.random.default_rng(243)
    print(f"Bloch-ball shrink vs project_to_physical, max |difference| over 200 matrices: {check_equivalence(rng):.1e}")
    print(f"{'channel':<18}{'shots':>6}{'active':>8}{'mean gain':>11}{'max gain':>10}{'ball - clip':>12}{'ball better/worse':>19}{'p (Wilcoxon)':>14}")
    for channel in ("depolarizing", "amplitude_damping", "phase_flip"):
        for shots in SHOTS:
            gains, vc, active = run(channel, shots, rng)
            nz = vc[np.abs(vc) > 1e-12]
            p = stats.wilcoxon(nz, alternative="greater").pvalue if len(nz) > 1 else float("nan")
            better, worse = int((vc > 1e-12).sum()), int((vc < -1e-12).sum())
            print(f"{channel:<18}{shots:>6}{active:>5}/{SEEDS}{gains.mean():>11.5f}{gains.max():>10.4f}"
                  f"{vc.mean():>12.5f}{better:>11}/{worse:<7}{p:>14.2e}")
            assert gains.min() >= -1e-12


if __name__ == "__main__":
    main()
