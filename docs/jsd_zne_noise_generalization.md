# Does the Shipped JSD-Predictive ZNE Generalize? And a Real Extension

!!! note
    Both signals on this page are now in the main library:
    [`dense_evolution.mitigation`](https://tatopenn-cell.github.io/Dense-Evolution/) --
    `jsd_predictive_zne_density_matrix` (classical JSD, validated on
    photon-loss/amplitude-damping, see
    [Photonic Predictive ZNE](photonic_predictive_zne.md)) and
    `coherence_predictive_zne_density_matrix` (coherence-L1, validated
    here on phase-type noise, `base_p<=0.10`). This page is the
    experimental log for both, including the negative results (QJSD, the
    coherent-error case) kept in rather than discarded.

**In plain terms**: the shipped method nudges the standard zero-noise-extrapolation formula only when it detects something "surprising" happening between noise levels, using a signal built from the *populations* of a quantum state. That works well for photon loss. This page asks whether it works for other kinds of noise too, finds a real, structural reason it can't see one entire category (dephasing), and finds a different signal that can.

## Part 1: screening across every standard noise channel

Same fairness discipline as `jsd_zne_oscillating_noise.py` (its own screening results are Part 1 of this page): both methods see the identical 3 noise scales, the baseline is the library's own plain 3-point Richardson (not a reimplementation), and the treatment is the real, shipped function. Screened across `depolarizing`, `bitflip`, `phaseflip`, `amplitude_damping`, `combined`, and a deterministic coherent (Rz over-rotation) error, 6 seeds per configuration:

- **amplitude_damping and combined** show a lead at low noise (`base_p=0.05`) -- both already inside the shipped signal's validated domain.
- **bitflip, depolarizing, phaseflip, and the coherent case** show no reliable effect at 6 seeds.

## Part 2: phaseflip and coherent noise are exactly zero, not just weak -- and why

`_jsd_predictive_zne_density_matrix_core`'s signal is the diagonal of the density matrix -- population probabilities only. Phaseflip (`K1=√p·Z`) and a coherent Rz rotation are both diagonal in the computational basis: they move phase (off-diagonal coherence), never populations. The classical JSD signal is **blind by construction**, not merely weak -- verified directly: the fidelity difference is exactly `0.0` at every tested noise strength and rotation angle.

## Part 3: quantum JSD does not fix this cleanly

A natural fix: use the quantum generalization of Jensen-Shannon divergence (von Neumann entropy of the full density matrix instead of Shannon entropy of the diagonal -- Lamberti et al. 2008). Tested with the identical nudge structure:

- On phaseflip: picks up a nonzero signal, but noisy and not significant at 6 seeds.
- On amplitude_damping/combined: **weakens** the already-working classical-JSD result in the same test.
- On the coherent case: still no useful effect. The reason is structural, not about which divergence is used -- the nudge fires on *nonlinearity between consecutive noise scales* (`jsd_23` vs `jsd_12`), and a smooth, deterministic function of the scale factor has `jsd_12≈jsd_23` regardless of which divergence measures it. Verified: the residual is exactly `0.0` for classical JSD and QJSD, and floating-point-noise-level (`~1e-6`) for the coherence signal below -- three orders of magnitude under any of that signal's real active-case effects.

QJSD is not adopted. A real negative result, kept in rather than discarded.

## Part 4: a coherence signal that works, measured the right way

The `l1`-norm of coherence (`Σ|ρ_ij|`, `i≠j` -- Baumgratz, Cramér & Plenio, PRL 113, 140401, 2014) targets what phaseflip actually destroys directly. Screening showed a lead too noisy to trust at 6 seeds -- the fix wasn't more seeds blindly, it was counting correctly: most seeds never trigger the nudge at all (`rectified=0`, diff exactly `0.0`, zero risk by construction), so a fair comparison restricts to the seeds where it *does* activate, the same convention [`photonic_predictive_zne.py`](photonic_predictive_zne.md) already established for its own validation.

At 200 seeds on phaseflip (`base_p=0.05`): **63/200 active (31.5%)**, and among those, **63/63 positive** -- mean fidelity gain `+0.014892`, one-sample t-test `p=1.07×10⁻⁸`, a permutation test (20,000 resamples) finding no resample matching or exceeding the observed effect (`p<0.00005`). Effect sizes among active points range from `+0.0001` to `+0.074`, median `+0.0094`.

## Part 5: confirmation sweep, matching the bar the promoted method met

Before treating this as more than a single lucky configuration, the same scope the photon-loss signal was validated against before promotion -- a noise-level sweep and a second circuit family:

| `base_p` | active/100 | wins | t-test `p` | permutation `p` |
|---|---|---|---|---|
| 0.03 | 35 | 33/35 | 8.8×10⁻⁵ | <0.00001 |
| 0.05 | 23 | 23/23 | 2.0×10⁻³ | <0.00001 |
| 0.08 | 21 | 21/21 | 5.7×10⁻³ | <0.00001 |
| 0.10 | 11 | 11/11 | 4.6×10⁻³ | 0.00100 |
| **0.15** | 17 | 14/17 | 0.288 | 0.301 |

The effect is real and significant by both tests from `base_p=0.03` through `0.10`, with a 100% win rate among active points at every one of those levels. At `base_p=0.15` it is **no longer significant** -- a real, honest upper boundary, not a universal effect at any noise strength.

A second circuit family (hardware-efficient VQE-style ansatz, 2 layers, identical construction to [`photonic_zne_multi_circuit_postselection.py`](photonic_predictive_zne.md)'s own) at `base_p=0.05`: **69/150 active (46%, a higher activation rate than GHZ)**, **67/69 positive**, `p=4.4×10⁻⁶` -- confirms the effect is not specific to GHZ states.

## Honest conclusion

The shipped classical-JSD signal is genuinely blind to phase-type noise, for a specific, verified, structural reason -- not a gap left uninvestigated. A coherence-based signal covers exactly that gap, with a large-sample result stronger than the original photon-loss validation (`p=1.07×10⁻⁸` vs. `p=0.0003`), confirmed across a noise-level sweep and a second circuit family with the same scope the original validation required before promotion. Deterministic coherent errors remain out of reach for this entire family of methods: detecting *nonlinearity between noise scales* cannot work on a noise process that has none by construction, regardless of which divergence measures it. `coherence_predictive_zne_density_matrix` was promoted to `dense_evolution.mitigation` with its validated scope stated directly, not glossed over: phaseflip/dephasing-dominated noise, `base_p<=0.10` -- see [`Draft and Verification`](https://github.com/tatopenn-cell/quantum-rag/blob/main/docs/draft_verification_methodology.md) for the process this promotion followed.

## Reproducing this

```bash
python scripts/jsd_zne_noise_generalization.py
```

Real data: [`data/jsd_zne_noise_generalization.csv`](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/blob/main/data/jsd_zne_noise_generalization.csv)-equivalent (generated locally, `/data/` is gitignored -- re-run the script above to reproduce).
