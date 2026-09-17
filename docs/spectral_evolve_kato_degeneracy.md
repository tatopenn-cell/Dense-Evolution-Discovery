# Gauge-safe spectral-function gradients at exact eigenvalue degeneracy

`jnp.linalg.eigh`'s reverse-mode gradient divides by `lambda_i - lambda_j`
for every eigenvector pair. When two eigenvalues are exactly degenerate,
this does not raise and does not always produce NaN -- it can silently
return a finite, WRONG gradient, since the eigenvectors spanning a
degenerate eigenspace aren't themselves uniquely defined.

## The fix: Kato's divided-difference formula

`matrix_function_eigh` computes `V f(Lambda) V^dagger` (e.g.
`exp(-iHt)`) via a `jax.custom_jvp` based on Kato's divided-difference
formula for matrix functions (Kato, *Perturbation Theory for Linear
Operators*, 1995, Ch. II.5.6), following Kasim (arXiv:2011.04366, 2020)
-- both citations verified against the actual paper text (arXiv PDF
downloaded and read directly), not trusted at face value from an
AI-drafted citation string. Kasim's Eq. 4.72 (a compatibility condition
an earlier, discarded `custom_vjp`-based attempt needed) was confirmed
present in the real paper text at that exact equation number.

Because the divided-difference formula never passes eigenvectors through
as an intermediate OUTPUT, it needs no compatibility condition: the
contribution from a degenerate block uses `f'(lambda)` directly.

## Measured, not assumed

On a real Hamiltonian with four exact doubly-degenerate eigenvalue pairs:

| method | gradient error vs. central finite differences |
|---|---|
| plain `jnp.linalg.eigh` | 0.98 (wrong by nearly an order of magnitude) |
| Kato (`matrix_function_eigh`) | 4e-10 (correct to numerical precision) |

Verified independently on a real Kaggle CPU kernel (not just Colab,
where this was first prototyped): 7/7 checks pass, including a guard
test (`test_std_eigh_fails_at_degeneracy`) that fails loudly if
`jnp.linalg.eigh`'s own gradient ever becomes correct at degeneracy.

## A real bug found by testing, not reading

The first Colab-drafted version passed `f`/`f_prime` (Python closures) as
ordinary positional arguments to a `@jax.custom_jvp`-decorated function
-- crashes immediately (`TypeError: ... is not a valid JAX type`), since
JAX tries to trace every positional argument by default. Fixed with
`nondiff_argnums=(1, 2)`. The physics and the citations were sound; the
JAX API usage had a real, silent, immediately-fatal mistake that only
running the code (not reading it) revealed.

## Tested on a real Discovery Hamiltonian: no measurable benefit there

The gap this doc originally left open -- "not yet wired into a real
Discovery use case with genuine exact degeneracy" -- is now closed, with
an honest answer. Scanned the wormhole/SYK teleportation signal's
`dSignal/dg` (N_MAJ=8, dim=256) as `g -> 0`, where `H_total`'s minimum
eigenvalue gap dips below the 1e-8 degeneracy tolerance:

![Degeneracy and gradient agreement vs. coupling g](assets/kato_syk_wormhole/degeneracy_vs_g.png)

`max|Kato - fd|` and `max|std - fd|` agree to 4 significant figures
(~2.4e-06) across the entire scan, down to `g=1e-6` where `min_gap`
(~2.8e-14) is far below the tolerance. Plain `jnp.linalg.eigh` never
diverges from finite differences here.

Why this differs from the synthetic case above (four EXACT
doubly-degenerate pairs, std error 0.98): that `H` was built as
`U diag([1,1,2,2,...]) U^dagger`, a mathematically exact tie. SYK's
random disorder generates near-degeneracies that are numerically small
but never mathematically exact -- `eigh`'s division by a merely-small
gap doesn't catastrophically fail the way division by an exactly-zero
gap does. The 1e-8 tolerance is a useful diagnostic threshold for "is
this worth worrying about," not the precision floor where the gradient
actually breaks.

`spectral_evolve` remains correct and validated for the case it targets;
plain `eigh` stays correct and faster for this real wormhole protocol as
actually used in this repo.

A real usage gotcha found writing this test: `spectral_evolve` calls
`ensure_x64()` internally, but that can't retroactively upcast arrays
the CALLER already built in float32/complex64 before ever calling it --
omitting an explicit `jax.config.update("jax_enable_x64", True)` in the
caller produced a spurious NaN on the first (JIT-tracing) call only.
Callers building their own Hamiltonians before calling `spectral_evolve`
still need to enable x64 themselves, up front.

## Status

Promoted to Dense-Evolution as `dense_evolution.physics.spectral`
(`has_exact_degeneracy`, `matrix_function_eigh`, `spectral_evolve`).

Scripts: `scripts/spectral_evolve_kato_degeneracy.py`,
`scripts/kato_syk_wormhole_real_usecase.py`.
