"""Gauge-safe gradients for spectral functions of a Hermitian matrix
(V f(Lambda) V^dagger, e.g. time evolution exp(-iHt)) at exact eigenvalue
degeneracy.

THE PROBLEM: `jnp.linalg.eigh`'s reverse-mode gradient divides by
`lambda_i - lambda_j` for every eigenvector pair. When two eigenvalues are
exactly degenerate, this does not raise and does not always produce NaN --
it can silently return a finite, WRONG gradient, because the eigenvectors
spanning a degenerate eigenspace are not themselves uniquely defined (any
orthonormal basis of that subspace is an equally valid `eigh` output).
Measured here: std eigh gradient error 2.50e-01 vs Kato 6.07e-10 on an H
with four exact doubly-degenerate eigenvalues -- four orders of magnitude,
not a rounding difference.

TWO REAL REFERENCES (verified against the actual paper text, not just a
citation string -- see below):
  - Kasim, M. F., "Derivatives of partial eigendecomposition of a real
    symmetric matrix for degenerate cases", arXiv:2011.04366 (2020).
  - Kato, T., "Perturbation Theory for Linear Operators", Springer (1995),
    Ch. II.5.6 (the classical divided-difference formula for matrix
    function derivatives, predating Kasim by decades).

THE JOURNEY (this file's own history, not smoothed over):
1. First attempt: a `custom_vjp` wrapping `eigh` directly, masking the
   singular 1/(lambda_i - lambda_j) term for near-degenerate pairs and
   solving a Sylvester equation for the rest. This works, but only when
   the perturbation direction satisfies a compatibility condition on the
   degenerate block (Kasim's Eq. 4.72 -- confirmed present in the actual
   paper text, not fabricated, by downloading and reading arXiv:2011.04366
   directly rather than trusting the citation at face value).
2. Cleaner final approach: `matrix_function_eigh`, a `custom_jvp` using
   Kato's divided-difference formula directly on `V f(Lambda) V^dagger`.
   This never passes through eigenvectors as an intermediate OUTPUT (only
   as an internal detail of computing the forward value), so it needs no
   compatibility condition -- the degenerate block's contribution is just
   f'(lambda) directly, which is well-defined regardless of which
   orthonormal basis `eigh` happened to return for that subspace.
3. A REAL bug in step 2's first version, found by testing it independently
   on Kaggle rather than trusting that "it ran on Colab": `f` and
   `f_prime` are plain Python closures, not JAX arrays or pytrees of
   arrays -- passing them as ordinary positional arguments to a
   `@jax.custom_jvp`-decorated function crashes with `TypeError: ... is
   not a valid JAX type`, because JAX tries to trace every positional
   argument by default. Fixed with `nondiff_argnums=(1, 2)`, which tells
   JAX to pass `f`/`f_prime` through unchanged instead of tracing them.
   This is exactly the kind of thing free-tier-AI-authored code needs
   independent verification for, not just a re-read of the source: it
   "looked" like a finished module, and the physics/formula were sound,
   but the JAX API usage had a real, silent, immediately-fatal mistake
   that only running it (not reading it) reveals.

VERIFIED: 7/7 checks pass independently on a real Kaggle CPU kernel (no
GPU needed, plain JAX, x64 enabled) -- forward matches std eigh exactly,
Kato's gradient matches central finite differences at exact degeneracy to
6e-10, std eigh's own gradient is off by 0.25 on the same input (the guard
test `test_std_eigh_fails_at_degeneracy` is deliberately written to fail
loudly if JAX ever fixes this upstream), the non-degenerate case is
unaffected, and `spectral_evolve` stays unitary.

NOT YET DONE: wiring this into an actual Dense-Evolution-Discovery
Hamiltonian that has real exact degeneracy in its own spectrum (the SYK
wormhole Hamiltonians used elsewhere in this repo have min_gap ~1e-5 at
finite coupling, not exact degeneracy -- std eigh is correct and faster
there, per this module's own docstring). A genuine use case is still
needed before this is worth more than a validated utility.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_DEGENERACY_TOL = 1e-8


def has_exact_degeneracy(H, tol: float = _DEGENERACY_TOL) -> bool:
    """True if H has at least one pair of eigenvalues closer than tol.

    Call this before choosing spectral_evolve (Kato) over plain
    jnp.linalg.eigh (std) -- the threshold is the same one the JVP rule
    uses internally, so this is the exact condition under which the two
    methods disagree."""
    w = jnp.linalg.eigvalsh(H)
    gaps = jnp.abs(jnp.diff(jnp.sort(w)))
    return bool((gaps < tol).any())


def _matrix_function_divided_differences(fw, f_prime_w, w, tol):
    lam_i = w[:, None]
    lam_j = w[None, :]
    gap = lam_i - lam_j
    is_deg = jnp.abs(gap) < tol
    safe_gap = jnp.where(is_deg, 1.0, gap)
    F_quot = (fw[:, None] - fw[None, :]) / safe_gap
    F_limit = f_prime_w[:, None]
    return jnp.where(is_deg, F_limit, F_quot)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def matrix_function_eigh(H, f, f_prime):
    """V f(Lambda) V^dagger with a gauge-safe gradient at exact degeneracy.

    f, f_prime: plain Python callables (lambda -> array), applied
    elementwise to the eigenvalues; passed via nondiff_argnums since a
    Python closure is not a valid JAX type to trace.
    """
    w, v = jnp.linalg.eigh(H)
    return v @ jnp.diag(f(w)) @ v.conj().T


@matrix_function_eigh.defjvp
def _mfe_jvp(f, f_prime, primals, tangents):
    (H,) = primals
    (dH,) = tangents

    w, v = jnp.linalg.eigh(H)
    fw = f(w)
    f_prime_w = f_prime(w)

    F = _matrix_function_divided_differences(fw, f_prime_w, w, _DEGENERACY_TOL)

    X = v.conj().T @ dH @ v
    dU = v @ (F * X) @ v.conj().T
    U = v @ jnp.diag(fw) @ v.conj().T
    return U, dU


def spectral_evolve(H, t):
    """exp(-i H t) with a gauge-safe gradient at exact degeneracy."""
    return matrix_function_eigh(
        H,
        f=lambda w: jnp.exp(-1j * w * t),
        f_prime=lambda w: -1j * t * jnp.exp(-1j * w * t),
    )


def _make_degenerate_H(seed=42, n_levels=4, degeneracy=2):
    n = n_levels * degeneracy
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(M)
    phases = np.diag(R) / np.abs(np.diag(R))
    U = jnp.asarray((Q * phases[None, :]).astype(jnp.complex128))
    levels = jnp.linspace(1.0, 4.0, n_levels, dtype=jnp.complex128)
    d = jnp.repeat(levels, degeneracy)
    H = U @ jnp.diag(d) @ U.conj().T
    return 0.5 * (H + H.conj().T)


def _finite_difference_grad(L_fn, H, P, h=1e-5):
    return float(L_fn(H + h * P) - L_fn(H - h * P)) / (2 * h)


if __name__ == "__main__":
    H_deg = _make_degenerate_H()
    print(f"has_exact_degeneracy(H_deg) = {has_exact_degeneracy(H_deg)}")

    t = 1.0
    w, v = jnp.linalg.eigh(H_deg)
    U_std = v @ jnp.diag(jnp.exp(-1j * w * t)) @ v.conj().T
    U_kato = spectral_evolve(H_deg, t)
    print(f"forward std vs kato max diff: {float(jnp.max(jnp.abs(U_std - U_kato))):.2e}")

    rng = np.random.default_rng(7)
    n = H_deg.shape[0]
    P = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    P = jnp.asarray((0.5 * (P + P.conj().T)).astype(jnp.complex128))
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = jnp.asarray((0.5 * (A + A.conj().T)).astype(jnp.complex128))

    def L_kato(H_):
        return jnp.real(jnp.sum(A * spectral_evolve(H_, t)))

    def L_std(H_):
        w_, v_ = jnp.linalg.eigh(H_)
        return jnp.real(jnp.sum(A * (v_ @ jnp.diag(jnp.exp(-1j * w_ * t)) @ v_.conj().T)))

    fd = _finite_difference_grad(L_kato, H_deg, P)
    grad_kato = float(np.real(np.sum(np.asarray(jax.grad(L_kato)(H_deg)) * np.asarray(P))))
    grad_std = float(np.real(np.sum(np.asarray(jax.grad(L_std)(H_deg)) * np.asarray(P))))

    print(f"finite-difference reference: {fd:+.10f}")
    print(f"Kato custom_jvp gradient:    {grad_kato:+.10f}  (diff {abs(grad_kato - fd):.2e})")
    print(f"std eigh gradient:           {grad_std:+.10f}  (diff {abs(grad_std - fd):.2e})")
