# Does the Shipped JSD-Predictive ZNE Generalize? And a Real Extension

!!! note
    `jsd_predictive_zne_density_matrix` (the classical-JSD signal) is
    promoted and lives in the main library:
    [`dense_evolution.mitigation`](https://tatopenn-cell.github.io/Dense-Evolution/),
    validated on photon-loss/amplitude-damping (see
    [Photonic Predictive ZNE](photonic_predictive_zne.md)). Everything on
    this page is draft-stage: a coherence-L1 signal with a real,
    large-sample-confirmed result on phase-flip noise, not yet promoted --
    see this repo's own README for the draft/verification distinction.

**In plain terms**: the shipped method nudges the standard zero-noise-extrapolation formula only when it detects something "surprising" happening between noise levels, using a signal built from the *populations* of a quantum state. That works well for photon loss. This page asks whether it works for other kinds of noise too, finds a real, structural reason it can't see one entire category (dephasing), and finds a different signal that can.

## Part 1: screening across every standard noise channel

Same fairness discipline as [`jsd_zne_oscillating_noise.py`](jsd_zne_oscillating_noise.md): both methods see the identical 3 noise scales, the baseline is the library's own plain 3-point Richardson (not a reimplementation), and the treatment is the real, shipped function. Screened across `depolarizing`, `bitflip`, `phaseflip`, `amplitude_damping`, `combined`, and a deterministic coherent (Rz over-rotation) error, 6 seeds per configuration:

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

## Honest conclusion

The shipped classical-JSD signal is genuinely blind to phase-type noise, for a specific, verified, structural reason -- not a gap left uninvestigated. A coherence-based signal covers exactly that gap, with a large-sample result stronger than the original photon-loss validation (`p=1.07×10⁻⁸` vs. `p=0.0003`). Deterministic coherent errors remain out of reach for this entire family of methods: detecting *nonlinearity between noise scales* cannot work on a noise process that has none by construction, regardless of which divergence measures it. This is a real, useful extension, held here in draft rather than promoted -- see [`Draft and Verification`](https://github.com/tatopenn-cell/quantum-rag/blob/main/docs/draft_verification_methodology.md) for why that distinction is deliberate, not a delay.

## Reproducing this

```bash
python scripts/jsd_zne_noise_generalization.py
```

Real data: [`data/jsd_zne_noise_generalization.csv`](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/blob/main/data/jsd_zne_noise_generalization.csv)-equivalent (generated locally, `/data/` is gitignored -- re-run the script above to reproduce).
