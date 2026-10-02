# Single-Qubit Population (JSD) Nudge for ZNE: No Better Than Random

Variant 3 of [issue #243](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/issues/243). Script: `scripts/single_qubit_jsd_zne.py`. Negative result.

The library's `jsd_predictive_zne_density_matrix` nudges the 3-point Richardson coefficients when the Jensen-Shannon divergence of the diagonal populations changes non-linearly across the noise scales. Here the same core runs on one qubit (the script checks that it matches the library to 1e-10), on the 2×2 matrix from single-qubit tomography: the signal uses only the Z populations. Controls as in [variant 2](single_qubit_coherence_zne.md): the same nudges shuffled across seeds, and one constant nudge on every seed.

Haar-random pure states, 400 seeds per row, `base_p = 0.05`, factors 1, 2, 3; mean gain in trace distance against `nudge_scale = 0`.

| Channel | K | Active | Signal | Shuffled | Constant | Signal better / worse | Signal vs shuffled, p |
|---|---|---|---|---|---|---|---|
| amplitude damping | 100 | 179/400 | 0.0166 | 0.0150 | 0.0341 | 172 / 7 | 0.34 |
| amplitude damping | 400 | 201/400 | 0.0113 | 0.0102 | 0.0204 | 193 / 8 | 0.27 |
| amplitude damping | 1600 | 210/400 | 0.0053 | 0.0049 | 0.0096 | 202 / 8 | 0.45 |
| depolarizing | 100 | 172/400 | 0.0156 | 0.0141 | 0.0324 | 166 / 6 | 0.33 |
| depolarizing | 400 | 190/400 | 0.0102 | 0.0093 | 0.0202 | 187 / 3 | 0.34 |
| depolarizing | 1600 | 187/400 | 0.0049 | 0.0044 | 0.0097 | 186 / 1 | 0.33 |
| phase flip | 100 | 171/400 | 0.0147 | 0.0149 | 0.0354 | 164 / 7 | 0.91 |
| phase flip | 400 | 187/400 | 0.0088 | 0.0088 | 0.0197 | 179 / 8 | 0.99 |
| phase flip | 1600 | 184/400 | 0.0046 | 0.0049 | 0.0108 | 175 / 9 | 0.56 |

## Reading it

- Taken alone, the signal column looks like a win (most active seeds improve). Against shuffled placement it is never significantly better (p from 0.27 to 0.99), and a constant nudge gives about twice the gain.
- On phase flip the populations do not change with the noise, so the signal fires only on shot noise, and it matches shuffled exactly.
- Same conclusion as the single-qubit coherence signal: the gain is from extrapolating less aggressively. Combining the two single-qubit signals cannot help, since neither beats random placement.
- The multi-qubit coherence version passed these controls (see variant 2). The multi-qubit population version, validated on photon loss (`photonic_predictive_zne.py`), has not been re-checked with them yet.
