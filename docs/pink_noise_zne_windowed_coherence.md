# Pink (1/f) Noise and ZNE: a Coherence-Windowed Filter That Doesn't Survive Its Own Negative Control

!!! note
    Negative result. `dense_evolution.noise.pink_noise_p_eff` (real 1/f-spectrum noise, Timmer & Koenig 1995 spectral synthesis) is a shipped library feature. This page documents an attempt to build a predictive ZNE variant for it, specifically so nobody re-runs the same dead end: two independent methodological traps were found along the way (documented below), and the final, methodologically-sound version of the idea still doesn't beat plain `zne_density_matrix`.

## Why 1/f noise looked worth trying

Real superconducting qubits are dominated by 1/f flux and charge noise, not the smoothly-scaling synthetic noise (`depolarizing`, `phaseflip`, `amplitude_damping`) the library's shipped predictive ZNE methods (`jsd_predictive_zne_density_matrix`, `coherence_predictive_zne_density_matrix`) were built and validated for. 1/f noise is temporally correlated -- nearby trials in a Monte Carlo ensemble share similar noise levels, distant ones don't -- which is, in principle, exploitable: split each noise scale's trial ensemble into small windows, flag windows whose local behavior looks anomalous, downweight them.

## Trap 1: a per-trial signal is structurally blind here

The first instinct -- compute a per-trial diagnostic (coherence-L1, or diagonal population, of a single-shot statevector) and use it to detect "noisy" trials directly -- is dead on arrival. A single Kraus outcome under phaseflip noise is always a **pure** state: the noise is a probabilistic choice of unitary per shot (apply `Z` or don't), never a mixing operation within one shot. Any purity-sensitive per-trial signal is therefore identically constant regardless of how much noise that particular shot actually experienced -- verified directly: standard deviation across 150 trials was exactly `0.0`. The noise only becomes visible once trials are averaged into an ensemble (a window); it cannot be seen in any single shot.

## Trap 2: an oracle-based signal looks great and means nothing

The next attempt used `|<ideal|trial>|^2` -- the overlap with the (unknown, in a real experiment) target state -- to decide which trials to keep. This produced a dramatic-looking result: 60-seed mean fidelity 0.983 versus 0.941 for plain `zne_density_matrix`. A negative control -- shuffle the trial order before windowing, destroying the pink noise's temporal structure -- reproduced the *same* result almost exactly (0.993 shuffled vs. 0.983 real). The effect had nothing to do with pink noise or temporal structure: it was pure survivorship bias from using the answer to select which data counts as the answer. No real experiment has this oracle available. This variant was never a candidate technique, only a bug caught by the negative control it should always be checked against.

## The real, oracle-free version -- and why it still doesn't work

The legitimate signal is windowed coherence-L1: split the trial ensemble into windows of size 5 (close to the pink trace's own measured 1/e autocorrelation length of 4 trials), compute each window's ensemble-averaged density matrix, take its coherence-L1, and use `dense_armor.utility.robust_filters.hampel_filter` to flag anomalously low-coherence (high local noise) windows for exclusion before the final average. This signal is real and does correlate with the true local noise level (measured directly: correlation -0.68 between a window's coherence-L1 and its true mean noise probability) -- it just doesn't translate into a fidelity improvement once fed through the full pipeline.

Tested with two experimental designs, both against the same real-vs-shuffled negative control, 150 seeds each, GHZ(3), phaseflip, `base_p=0.05`:

- **LOCAL scaling** -- an independent pink-noise realization drawn fresh at each of the 3 ZNE noise-scale factors. Analogous to what Schultz et al. ([arXiv:2201.11792](https://arxiv.org/abs/2201.11792), *Analyzing the impact of time-correlated noise on zero-noise extrapolation*) call local noise scaling -- their result: unreliable for time-correlated noise, since it can't preserve the noise's spectral/correlation structure while changing intensity.
- **GLOBAL folding** -- one shared underlying pink-noise realization; each scale factor's per-trial noise level is the mean of a longer contiguous stretch of that *same* realization. The correlation-preserving analog of the paper's recommended global unitary folding, adapted from a unitary-circuit-folding context to a stochastic-noise-source one.

| Design | Real gain (mean, t-test p) | Shuffled gain (mean, t-test p) | Real > shuffled (Mann-Whitney p) |
|---|---|---|---|
| Local scaling | +0.01157, p=0.0015 | +0.01264, p=0.0002 | p=0.76 (not significant; wrong direction) |
| Global folding | -0.00401, p=0.53 | -0.00030, p=0.96 | p=0.42 (not significant) |

Local scaling shows a real-looking gain against zero -- but the shuffled control is just as significant, ruling out any real dependence on the pink noise's temporal structure. Global folding, the design [arXiv:2201.11792](https://arxiv.org/abs/2201.11792) identifies as the one that actually preserves correlated noise's spectral properties under scaling, shows no effect at all in either condition. Consistent with that paper's core finding -- time-correlated noise is described there as "beyond the scope of ZNE in principle" for the general case -- this specific attempt to work around that limitation with an oracle-free coherence signal does not succeed.

**Not promoted.** `pink_noise_p_eff` ships as a noise-generation primitive; no predictive ZNE variant for it is added to `dense_evolution.mitigation` on the strength of this result.

Produced by `scripts/pink_noise_zne_windowed_coherence.py`.

## References

1. J. Timmer, M. Koenig, *On generating power law noise*, Astronomy and Astrophysics 300, 707 (1995).
2. K. Schultz, R. LaRose, A. Mari, G. Quiroz, N. Shammah, B. D. Clader, W. J. Zeng, *Analyzing the impact of time-correlated noise on zero-noise extrapolation*, [arXiv:2201.11792](https://arxiv.org/abs/2201.11792).
