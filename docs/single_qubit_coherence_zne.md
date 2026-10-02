# Single-Qubit Coherence Nudge for ZNE: the Gain Is Not the Signal (the Multi-Qubit One Holds)

Variant 2 of [issue #243](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/issues/243). Script: `scripts/single_qubit_coherence_zne.py`. Negative result.

The library's `coherence_predictive_zne_density_matrix` lowers the weight of the 3-point Richardson extrapolation when the l1 coherence (Baumgratz, Cramer, Plenio, PRL 113, 140401 (2014)) changes non-linearly across the noise scales, then projects onto a physical state. Here the same core runs on one qubit, on the 2×2 matrix rebuilt from single-qubit tomography (X, Y, Z, `K` shots each), so the signal is `2|ρ₀₁|`.

## First test: looks like a clear win

Haar-random pure states, 200 seeds, `base_p = 0.05`, factors 1, 2, 3. Baseline: the same core with `nudge_scale = 0` (Richardson plus projection), as in the original validation of the multi-qubit version. Counted on seeds where the nudge activates, as there.

| Channel | K | Active | Mean gain (active) | Better / worse | t-test p |
|---|---|---|---|---|---|
| phase flip | 100 | 96/200 | 0.0308 | 94 / 2 | 1.0e-23 |
| phase flip | 400 | 100/200 | 0.0185 | 100 / 0 | 3.0e-20 |
| phase flip | 1600 | 95/200 | 0.0064 | 94 / 1 | 3.1e-12 |
| amplitude damping | 100 | 105/200 | 0.0260 | 103 / 2 | 7.4e-18 |
| amplitude damping | 400 | 106/200 | 0.0144 | 101 / 5 | 1.5e-15 |
| amplitude damping | 1600 | 97/200 | 0.0084 | 95 / 2 | 2.0e-20 |
| depolarizing | 100 | 104/200 | 0.0286 | 101 / 3 | 2.8e-20 |
| depolarizing | 400 | 105/200 | 0.0176 | 102 / 3 | 9.8e-21 |
| depolarizing | 1600 | 102/200 | 0.0087 | 101 / 1 | 5.7e-22 |

## Control: the same nudges at random, and one fixed nudge

400 seeds, mean gain over all seeds against `nudge_scale = 0`.

| Channel | K | Signal | Same nudges shuffled across seeds | One constant nudge on every seed |
|---|---|---|---|---|
| phase flip | 100 | 0.0149 | 0.0117 | 0.0216 |
| phase flip | 400 | 0.0090 | 0.0070 | 0.0141 |
| phase flip | 1600 | 0.0028 | 0.0023 | 0.0044 |
| amplitude damping | 100 | 0.0133 | 0.0126 | 0.0225 |
| amplitude damping | 400 | 0.0079 | 0.0070 | 0.0147 |
| amplitude damping | 1600 | 0.0042 | 0.0037 | 0.0074 |
| depolarizing | 100 | 0.0148 | 0.0118 | 0.0232 |
| depolarizing | 400 | 0.0084 | 0.0069 | 0.0139 |
| depolarizing | 1600 | 0.0038 | 0.0032 | 0.0062 |

## Reading it

- Shuffled nudges keep most of the gain, and a constant nudge on every seed does better than the signal. The improvement comes from extrapolating less aggressively, which amplifies shot noise less, not from the coherence signal choosing when to act.
- The signal adds only a little over random placement (0.0005 to 0.003).
- A constant nudge trades variance for bias: it moves the estimate toward the noisy data, so at higher noise or with many more shots it can lose. It is not a free improvement and is not promoted.
- The multi-qubit version passes the same controls (next section): there the signal does carry the gain.

## Re-check of the multi-qubit validation with the same controls

Script: `scripts/coherence_zne_shuffled_control.py`. Exactly the setup of #208 (GHZ(4), phase flip, `base_p = 0.05`, 150 trajectories per scale, seeds 900000 + i, Uhlmann fidelity); it reproduces the original 63/200 active, 63/63 positive, p = 1.07e-8.

| Nudge | Mean gain | Better / worse | t-test p |
|---|---|---|---|
| coherence signal (library) | 0.00469 | 63 / 0 | 1.1e-08 |
| same nudges shuffled across seeds | 0.00072 | 17 / 14 | 0.16 |
| constant nudge on every seed | 0.00243 | 65 / 43 | 4.5e-06 |

The signal beats shuffled placement by 0.0040 (paired p = 5.3e-06) and the constant nudge by 0.0023 (p = 3.9e-06). On the full GHZ(4) state, where the coherence sits in the `|0000><1111|` element, the l1 signal really chooses when to act; on a single qubit, with one off-diagonal element, it does not.

## Details

- In a GHZ state each single-qubit reduced state is `I/2`, so the local signal is zero at every scale and never fires; that is why this test uses single-qubit pure states.
- Requires `ensure_x64()` (called by the script) until Dense-Evolution#343 is merged.
