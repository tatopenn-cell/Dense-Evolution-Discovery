# Single-Qubit ZNE with Bloch-Ball Projection

Variant 1 of [issue #243](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/issues/243). Script: `scripts/noise_mitigation_validation/single_qubit_bloch_ball_zne.py`.

A single-qubit state is fixed by its Bloch vector `r = (<X>, <Y>, <Z>)`, which needs only three measurement settings. Zero-noise extrapolation of each component can return `|r| > 1`, a state that does not exist. For one qubit the eigenvalues of the density matrix are `(1 ± |r|)/2`, so the eigenvalue projection onto the simplex of Smolin, Gambetta and Smith (arXiv:1106.5458) is exactly "shrink `r` to length 1". The script checks this against the library's `project_to_physical`: maximum difference `7.8e-16` over 200 random matrices.

## Setup

- Haar-random pure single-qubit states, 200 seeds per row.
- Channels: depolarizing, amplitude damping, phase flip, `base_p = 0.05`, noise factors 1, 2, 3.
- `K` shots per basis per noise factor (binomial sampling of each Pauli expectation).
- Estimators at the same shots: plain 3-point Richardson; Richardson then clipping each component to `[-1, 1]` (the post hoc clipping discussed by Miranskyy, Sorrenti, Thind, Gravel, arXiv:2604.24475, Sect. III.D and VI.A); Richardson then Bloch-ball projection.
- Metric: trace distance to the ideal state, `|r - n|/2`, used only for grading.

## Results

| Channel | K | Projection acts | Mean gain vs Richardson | Max gain | Mean gain vs clipping | Ball better / worse than clipping | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| depolarizing | 100 | 137/200 | 0.0537 | 0.417 | 0.0353 | 132 / 5 | 4.1e-23 |
| depolarizing | 400 | 117/200 | 0.0223 | 0.231 | 0.0193 | 117 / 0 | 3.1e-21 |
| depolarizing | 1600 | 113/200 | 0.0061 | 0.095 | 0.0057 | 113 / 0 | 1.4e-20 |
| amplitude damping | 100 | 125/200 | 0.0483 | 0.345 | 0.0326 | 118 / 7 | 3.7e-21 |
| amplitude damping | 400 | 129/200 | 0.0237 | 0.137 | 0.0199 | 126 / 3 | 1.4e-22 |
| amplitude damping | 1600 | 114/200 | 0.0068 | 0.069 | 0.0066 | 114 / 0 | 9.6e-21 |
| phase flip | 100 | 143/200 | 0.0631 | 0.348 | 0.0427 | 137 / 6 | 1.6e-24 |
| phase flip | 400 | 117/200 | 0.0143 | 0.151 | 0.0117 | 115 / 2 | 6.5e-21 |
| phase flip | 1600 | 112/200 | 0.0085 | 0.086 | 0.0079 | 108 / 4 | 5.1e-20 |

## Reading the table

- The projection acts on more than half of the seeds at every shot count: extrapolating a nearly pure state often overshoots the Bloch sphere.
- The gain over plain Richardson is never negative. This is guaranteed, not measured luck: the Bloch ball is convex and contains the ideal state, and a metric projection onto a convex set never moves a point further from any point of that set. The table measures how often and how much it helps.
- Against component clipping the ball projection wins in most active seeds and loses in a few (at most 7/200). Clipping only enforces each `|<P>| ≤ 1`, so the clipped vector can still lie outside the ball.
- The gain shrinks as shots grow (about 0.05 at K = 100, under 0.01 at K = 1600): the projection mostly removes overshoot caused by shot noise amplified by extrapolation.

## Details

- `project_to_physical` gives the same matrices only with JAX 64-bit enabled; without it the library computed in 32-bit and differed by `4.5e-07`. The script calls `ensure_x64()` first. The library's mitigation functions do not call it themselves, which is worth fixing in Dense-Evolution.
- Not yet tested: the multi-qubit product-state case (one projection per qubit) and real-device noise.
