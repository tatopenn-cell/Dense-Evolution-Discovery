# Classical Shadows: a Real Bug Fix, and Shadow-Estimated Magic Entropy

A prior Colab session (following Huang, Kueng, Preskill 2020, "Predicting Many Properties of a Quantum System from Very Few Measurements") proposed a `dense_evolution/circuits/shadows.py` module with a `ClassicalShadow` class and `predict_renyi_entropy`. Checking it against the real paper turned up a real bug, and also a genuine, well-supported extension worth building: using the same shadow machinery to estimate Experiment 30's magic entropy from measurement snapshots instead of the exact state.

## The bug: a missing transpose

The purity estimator's cross-trace between independent shadow snapshots was computed as:

```python
jnp.einsum('ijk,mjk->im', matrices, matrices)
```

This contracts matching indices with no transpose — `sum_jk A_i[j,k]*A_m[j,k]` — not `Tr(A_i @ A_m) = sum_jk A_i[j,k]*A_m[k,j]`. Verified directly on two fixed Hermitian test matrices with complex off-diagonal entries: the buggy contraction gives `23.0`, the true `Tr(A@B)` is `11.0`.

The bug is **silent whenever every snapshot happens to be real-valued** — a Z-basis-only demo, which is exactly why the Colab's own Bell-state `S2=1.000000` output never caught it. It is wrong whenever X/Y-basis snapshots (complex entries) are mixed in, which is the normal case for a real random-Pauli shadow protocol. `predict_renyi_entropy` inherits the bug since it calls the buggy purity function.

Fixed on a genuinely complex case (a `|T>`-state shadow run, which does exercise all three Pauli bases): the buggy estimator stays biased at **0.48** even at 100,000 snapshots (`Tr[rho^2]=1` is the true value for a pure state) — a systematic bias, not statistical noise, so more samples do not fix it. The corrected contraction converges to `1.01`.

[![Classical shadows: purity bug fix and shadow-estimated magic entropy](assets/quantum_shadows_magic_entropy/quantum_shadows_magic_entropy.png)](assets/quantum_shadows_magic_entropy/quantum_shadows_magic_entropy.png)

## The extension: shadow-estimated magic entropy

Huang et al.'s headline nonlinear example is estimating `Tr[rho^2]` (purity) from **two independent** shadow copies via a U-statistic — and they state explicitly that "this approach readily generalizes to higher order polynomials." Experiment 30's magic entropy needs exactly a higher-order polynomial: the reduced convolution matrix `R = Tr_{2,3}[V(rho (x) rho (x) rho)V^dagger]` is a **linear functional of `rho^{(x)3}`** (3 copies), since partial trace and unitary conjugation are both linear. Each entry can be written `R_ab = Tr[O_ab . rho^{(x)3}]` for a fixed operator `O_ab` built directly from the Key Unitary `V` -- exactly the shape the paper's own generalization statement covers, not an invented extension.

So: group the single-qubit shadow snapshots into disjoint triples, estimate each entry of `R` as the average over triples of `Tr[O_ab . (rho_hat_i (x) rho_hat_j (x) rho_hat_k)]` (unbiased, since each snapshot in a triple is an independent unbiased estimator of `rho`), then compute the von Neumann entropy of the **estimated** `R` classically (with Hermitization, eigenvalue clipping, and trace renormalization to absorb estimation noise). The entropy step itself is never shadow-estimated directly -- Huang et al. don't do this for their own Rényi-2 entanglement entropy example either; only the linear reduced-matrix reconstruction is shadow-based, and entropy is computed classically afterward on the small estimated matrix.

Validated against Experiment 30's exact values: a `|T>`-state (exact magic entropy `0.811` bits) and a `|+>`-state (exact `0.000`, a stabilizer state). At 300,000 snapshots the shadow estimate lands within `0.03` bits of both exact values, though convergence is visibly noisy along the way (a real, honestly-reported observation, not a smooth curve) -- reflecting real single-qubit shadow variance, not a flaw in the estimator's construction (confirmed unbiased separately: the `O_ab` operators exactly reproduce Experiment 30's reduced matrix when fed exact, noiseless copies of `rho^{(x)3}` instead of shadow estimates).

## A second, smaller bug caught along the way

Building and testing the `O_ab` operators surfaced an unrelated index-ordering slip in this experiment's own first draft: `Tr[(|a><b| (x) I) M] = R_ba`, not `R_ab` (confirmed by direct index expansion, not just the cyclic-trace shortcut, which looks right on paper but hides the swap). It happened not to affect the entropy result here (a Hermitian matrix and its transpose-conjugate share eigenvalues, so the final magic-entropy numbers above were unaffected either way) but would have broken any other use of the reconstructed matrix `R` -- caught by a dedicated unit test that checks matrix entries directly rather than only the downstream entropy.

## Making it robust: median-of-means

The original version above used plain averaging over triples/pairs — unbiased, but with an unbounded failure mode: a systematic fault affecting one contiguous stretch of a measurement run (e.g. a calibration drift, not independent per-sample noise) drags a plain mean proportionally to how much of the run it corrupted. Huang et al.'s own protocol uses median-of-means (MoM) instead: split the samples into `n_groups` batches, average each batch, then take the **median** of those batch means.

Verified directly, not just cited: with `n_groups=20`, corrupting a growing contiguous fraction of a `|T>`-state purity run (true value `1.0`) —

| % corrupted | naive mean | median-of-means |
|---|---|---|
| 0% | 1.01 | 1.00 |
| 20% | -9.18 | 1.00 |
| 40% | -19.39 | 0.97 |
| 45% | -21.94 | 0.93 |

— the naive mean is dragged linearly away from the truth from the first corrupted sample on; median-of-means stays within `0.5` of the true value all the way to 40% corruption (a real, asserted property, not just a plot), since fewer than half of the 20 groups are ever fully corrupted below that point. Past ~50% corrupted, MoM's own guarantee breaks down too — checked honestly rather than oversold: with 60% of samples corrupted, the median itself is forced to land on a corrupted group's mean.

Complex-valued entries (the magic-entropy reduced matrix `R_ab`) get their real and imaginary parts median-of-means'd independently — the median of a set of complex numbers has no single standard definition, but doing real/imaginary parts separately is the standard practical choice.

Trade-off, also checked directly rather than assumed: MoM has somewhat higher variance than plain averaging on *uncorrupted* data, since it only sees `n_groups` batch means instead of every sample individually. Across 10 seeds at 150,000 snapshots, the `|+>`-state's magic entropy (true value `0`) stayed under `0.09` bits for 9 of them but reached `0.16` for one — an honest, real spread, not a bug (see the test suite's own comments).

[![Classical shadows: purity bug fix and shadow-estimated magic entropy](assets/quantum_shadows_magic_entropy/quantum_shadows_magic_entropy.png)](assets/quantum_shadows_magic_entropy/quantum_shadows_magic_entropy.png)

## Status

`estimate_purity_fixed` and `estimate_magic_entropy_from_shadows` (both now median-of-means-based, `n_groups=20` default, configurable) are implemented and validated in `scripts/quantum_shadows_magic_entropy.py`, not yet promoted to `dense_evolution`. The robustness gap that blocked promotion is now closed. Two things still open before promotion: the API shape needs its own design (measurement snapshots in, not a density matrix — unlike every other function in `dense_evolution.mitigation`), and a proper sample-complexity guide (how many snapshots for what confidence bound) hasn't been derived yet.

## Reproduce

```bash
python scripts/quantum_shadows_magic_entropy.py
```

Produces `data/quantum_shadows_purity_bugfix.csv`, `data/quantum_shadows_magic_entropy_convergence.csv`, `data/quantum_shadows_median_of_means_robustness.csv`.
