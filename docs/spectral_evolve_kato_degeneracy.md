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

## Status

Promoted to Dense-Evolution as `dense_evolution.physics.spectral`
(`has_exact_degeneracy`, `matrix_function_eigh`, `spectral_evolve`).

Script: `scripts/spectral_evolve_kato_degeneracy.py`.
