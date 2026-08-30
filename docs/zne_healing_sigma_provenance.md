# ZNE Healing-Branch Sigma Provenance: a Confound, Not a New Win

`dense_evolution.zero_noise_extrapolation`'s "healing-adapted" branch (`sigma_at_base_noise`, see [docs/api/healing.md](https://tatopenn-cell.github.io/Dense-Evolution/healing/)) perturbs the 3 Richardson coefficients via `calculate_delta_preemp(sigma_at_base_noise, target_sigma_ideal)`. The library's own `calculate_advanced_sigma` (`kappa*H*Psi*Omega_sync*tau_K`) was meant to produce that sigma, but its 5 inputs have no defined provenance in a ZNE context: `dashboard_core`'s `run_zne_mitigation` only ever has scalar Pauli expectation values, never a density matrix, so entropy/purity-style inputs have no real data to come from at that integration point -- deliberately left unwired rather than plumbed in speculatively.

**What's being tested**: bypass the unresolved `calculate_advanced_sigma` provenance question entirely and test the *mechanism* directly with the most literal, oracle-free candidate for "sigma at the base noise level" -- the real empirical standard deviation of the `n_trials` stochastic Kraus-draw ensemble at `noise_factor=1`. No density matrix, no ideal-state comparison (`uhlmann_fidelity`'s own docstring bans that: "never to feed into" a correction).

## Setup

2-qubit Bell state, `<ZZ>` (ideal = 1.0), 3 noise channels (depolarizing, bitflip, amplitude_damping) x 3 noise levels (p=0.02/0.05/0.10) x 5 seeds x 300 trials per noise scale -- 45 configurations total. `target_sigma_ideal` is measured per-config (empirical std at the sweep's smallest real noise level, independently seeded), not guessed.

## Result: looks like a win at first

| | mean error delta (plain − healing) | win rate |
|---|---|---|
| Real pairing (real `base_std`) | +0.000458 | 88.9% (40/45) |

Every single (noise_model, noise_p) combination came out net positive -- exactly the shape of result that looked convincing before Experiment 25's own JSD-ZNE nudge turned out to be a confound.

## Negative control: shuffle `sigma_at_base_noise` across runs

Same 45 configurations, same `means`/`ideal` pairs, but `base_std` **permuted at random** across rows before being fed into the healing branch:

| | mean error delta | win rate |
|---|---|---|
| Negative control (shuffled `base_std`) | +0.000491 | 86.7% (39/45) |

**Statistically indistinguishable from the real pairing.** The win doesn't depend on `base_std` reflecting anything real about that specific run -- any positive-valued sigma produces essentially the same small structural nudge to the Richardson coefficients.

## Verdict

**Confound, not a real effect** -- the same failure mode as Experiment 25, on a different part of the codebase. The healing-adapted branch's coefficient perturbation (`c1 = 3.0 - 0.01*delta_p`, `c2 = -3.0 + 0.02*delta_p`, `c3 = 1.0 - 0.01*delta_p`) is a small enough nudge away from the exact-interpolation Richardson coefficients that it mildly regularizes the extrapolation regardless of what `delta_p` actually is -- consistent with `polynomial_extrapolate`'s own documented finding that exact 3-point Richardson coefficients are numerically fragile under real statistical noise. The apparent benefit is about *perturbing away from the exact coefficients at all*, not about *what the perturbation is driven by*.

**What this means for `calculate_advanced_sigma`**: even a fully-designed, physically-grounded set of `kappa`/`H`/`Psi`/`Omega_sync`/`tau_K` inputs would not make the healing-adapted ZNE branch meaningfully better than feeding it any other positive scalar -- the consumer doesn't discriminate signal from noise at the tested scale. Completing `calculate_advanced_sigma`'s provenance is not blocked on missing inputs anymore; it's blocked on the downstream mechanism itself not using its input, which is a different (and more fundamental) problem than the one originally suspected.

## Status

Not promoted. `calculate_advanced_sigma` remains publicly exported (backward-compat) but unused by any real pipeline, same as before this experiment -- now with a verified reason why completing its provenance wouldn't help, rather than an open question.

## Reproduce

```bash
python scripts/zne_healing_sigma_provenance.py
```