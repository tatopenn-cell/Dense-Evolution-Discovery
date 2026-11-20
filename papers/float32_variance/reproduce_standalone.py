"""Variance in float32: the standalone program of the paper.

Everything the paper claims on the examples and on the accuracy runs is computed here, with no
library of ours: the kernel of ``stable_var`` is written out below, copied from the code of the
library (``dense_armor/core/signals/stats/stable.py``), and needs only JAX and NumPy.

    python reproduce_standalone.py

It prints, in order: the problem, the stock index, the shift, the TwoSum, the compensated tree, the
Kahan sum, the variance of identical values, the four datasets and a small stack of layers. The last
digits can differ with the version of JAX and with the machine. The times are not measured here.
"""

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6
OFFSET, SPREAD = 1000.0, 0.004
WSCALE = 0.7


# the kernel of stable_var, copied from the library

def _as_float_array(x):
    """Return ``x`` as a JAX array with a float dtype.

    Complex input is not handled here: :func:`stable_var` and
    :func:`stable_moments` split it into real and imaginary part and
    return ``var(real) + var(imag)``, which is ``E|z - mean(z)|**2``.
    """
    a = jnp.asarray(x)
    kind = a.dtype.kind
    if a.dtype == jnp.bfloat16:
        kind = "f"
    if kind == "c":
        raise TypeError(
            "stable_var: complex input is handled by stable_var and "
            "stable_moments directly, not by _as_float_array"
        )
    if kind == "f":
        if a.dtype in (jnp.float16, jnp.bfloat16):
            a = a.astype(jnp.float32)
    elif kind in ("i", "u", "b"):
        a = a.astype(jnp.float32)
    else:
        raise TypeError(f"stable_var: unsupported dtype {a.dtype}")
    return a


def _result_dtype(original):
    """Return the dtype of the result: input for floats, float32 else."""
    d = getattr(original, "dtype", None)
    if d is None:
        d = np.asarray(original).dtype
    else:
        d = np.dtype(d)
    if d.kind == "f" or d == jnp.bfloat16:
        return d
    return np.float32


def _cast_out(a, out_dtype):
    """Cast ``a`` to ``out_dtype``, down to float32 when x64 is off."""
    d = jnp.dtype(out_dtype)
    if d == jnp.float64 and not jax.config.jax_enable_x64:
        return a.astype(jnp.float32)
    return a.astype(d)


def _check_axis(a, axis):
    if axis is None:
        return tuple(range(a.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    nd = a.ndim
    for ax in axis:
        if not (-nd <= ax < nd):
            raise ValueError(f"stable_var: axis {ax} out of bounds for ndim {nd}")
    return tuple(sorted({ax % nd for ax in axis}))


def _prepare(a, axes):
    """Transpose reduction axes to the end, flatten them, return (b_flat, batch_shape, n)."""
    nd = a.ndim
    other_axes = [i for i in range(nd) if i not in axes]
    perm = other_axes + list(axes)
    if perm != list(range(nd)):
        a = jnp.transpose(a, perm)
    batch_shape = a.shape[: len(other_axes)]
    red_shape = a.shape[len(other_axes) :]
    n = 1
    for s in red_shape:
        n *= s
    b_flat = a.reshape(-1, n) if batch_shape else a.reshape(1, n)
    return b_flat, batch_shape, n


def _two_sum(a, b):
    """``TwoSum``: ``a + b = s + err``, ``s = fl(a + b)``.

    Ogita, Rump, Oishi 2005, Algorithm 3.1, PDF page 7; Shewchuk 1997,
    Theorem 7, PDF page 12.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _compensated_tree(v):
    """Compensated tree of Kahan for a 1-D array: each merge uses TwoSum.

    The array is merged in ``log2(n)`` levels; when a level has an odd
    length, one zero is appended. At each level, pairs of partial sums
    ``(S, C)`` are combined with three ``TwoSum`` calls: one for the high
    parts, one for the carries, one for their sum. The result is
    ``S + C``.

    Args:
        v: a 1-D array.

    Returns:
        The compensated sum, a scalar of the same dtype.
    """
    S = v
    C = jnp.zeros_like(v)
    while S.shape[0] > 1:
        if S.shape[0] % 2 == 1:
            S = jnp.concatenate([S, jnp.zeros((1,), S.dtype)])
            C = jnp.concatenate([C, jnp.zeros((1,), C.dtype)])
        s1, e1 = _two_sum(S[0::2], S[1::2])
        s2, e2 = _two_sum(C[0::2], C[1::2])
        s3, e3 = _two_sum(s1, s2)
        S = s3
        C = e1 + e2 + e3
    return S[0] + C[0]


def _kernel_var(v, n):
    """Variance (biased) of one 1-D slice: shift, two-term, compensated tree on d^2."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    d = y - m
    s2 = _compensated_tree(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


def _kernel_mom(v, n):
    """Mean and biased variance of one 1-D slice: shift, two-term, compensated tree."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    mean = ref + m
    d = y - m
    s2 = _compensated_tree(d * d)
    s1 = jnp.sum(d)
    var_pop = jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))
    return mean, var_pop


def _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype):
    var_pop = var_pop.reshape(batch_shape)
    if keepdims:
        for ax in sorted(axes):
            var_pop = jnp.expand_dims(var_pop, axis=ax)
    if ddof == 0:
        var = var_pop
    else:
        denom = n - ddof
        if denom <= 0:
            var = jnp.where(
                var_pop == 0, jnp.zeros_like(var_pop), jnp.full_like(var_pop, jnp.nan)
            )
        else:
            factor = var_pop.dtype.type(n / denom)
            var = var_pop * factor
    return _cast_out(var, out_dtype)


def _finish_mean(mean, axes, batch_shape, keepdims, out_dtype):
    mean = mean.reshape(batch_shape)
    if keepdims:
        for ax in sorted(axes):
            mean = jnp.expand_dims(mean, axis=ax)
    return _cast_out(mean, out_dtype)


def stable_var(x, axis=None, ddof=0, keepdims=False):
    """Variance of ``x`` along ``axis``, computed in the input dtype.

    The shift ``y = x - x[0]``, the shifted-data algorithm with K = x[0]
    (Healy 2022, Algorithm 2), comes first. Then ``m = mean(y)``, ``d = y - m`` and ``var = mean(d**2) -
    mean(d)**2``, one slice at a time with ``vmap``, with ``d**2`` summed
    by a compensated tree of Kahan, where each merge uses
    ``TwoSum`` (``log2(n)`` vectorised levels).
    The result is exactly ``0.0`` on identical values up to ``1e38``, in
    eager mode and under ``jax.jit``. Complex input gives ``E|z -
    mean(z)|**2``, a real number. Integers, ``float16`` and ``bfloat16``
    are computed in ``float32``.

    Args:
        x: array-like. numpy and JAX arrays are accepted.
        axis: int, tuple of int, or ``None`` (default, all axes).
        ddof: delta degrees of freedom, default ``0``. When ``ddof >= n``
            the variance is ``nan``, or ``0.0`` if it is zero.
        keepdims: keep the reduced axes with size one, default ``False``.

    Returns:
        The variance, dtype of the input (or float32 for integers).

    Raises:
        ValueError: if the reduction axis is empty or out of bounds, or
            if ``ddof`` is negative.
        TypeError: if the dtype is not numeric.

    Example:
        >>> import numpy as np
        >>> from dense_armor.core.signals.stats.stable import stable_var
        >>> x = np.array([1e10, 1e10, 1e10], dtype=np.float32)
        >>> float(stable_var(x))
        0.0
        >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> round(float(stable_var(x)), 6)
        0.666667
    """
    if ddof < 0:
        raise ValueError(f"stable_var: ddof must be non-negative, got {ddof}")
    xa = jnp.asarray(x)
    if xa.dtype.kind == "c":
        re = xa.real
        im = xa.imag
        return stable_var(re, axis=axis, ddof=ddof, keepdims=keepdims) + stable_var(
            im, axis=axis, ddof=ddof, keepdims=keepdims
        )
    a = _as_float_array(x)
    out_dtype = _result_dtype(x)
    if a.ndim == 0:
        if ddof >= 1:
            return _cast_out(jnp.asarray(jnp.nan, dtype=a.dtype), out_dtype)
        return _cast_out(jnp.asarray(0.0, dtype=a.dtype), out_dtype)
    axes = _check_axis(a, axis)
    if len(axes) == 0:
        return _cast_out(a, out_dtype)
    for ax in axes:
        if a.shape[ax] == 0:
            raise ValueError("stable_var: empty reduction axis")

    b_flat, batch_shape, n = _prepare(a, axes)
    var_pop = jax.vmap(lambda v: _kernel_var(v, n))(b_flat)
    return _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype)


# the baseline, the cases and the error of the accuracy runs

def naive(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    return jnp.mean(d * d, axis=axis)


def base(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    return jnp.maximum(jnp.mean(d * d, axis=axis) - jnp.mean(d, axis=axis) ** 2, 0.0)


def _max_rel(err, ref):
    err = np.asarray(err, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mask = ref != 0.0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(err[mask] - ref[mask]) / np.abs(ref[mask])))


def identical_cases():
    return [
        ("f32_1e10", np.float32, 1e10),
        ("f32_7_5e10", np.float32, 7.5e10),
        ("f32_1e15", np.float32, 1e15),
        ("f32_1e38", np.float32, 1e38),
        ("c64_7_5e10", np.complex64, 7.5e10 + 7.5e10j),
    ]


def accuracy_cases(smoke):
    n = 100 if smoke else 1000
    m = 8 if smoke else 64
    rng = np.random.default_rng(0)
    return [
        ("offset_1e3", rng.normal(1e3, 0.01, (n, m)).astype(np.float32), 0),
        ("offset_1e4", rng.normal(1e4, 1.0, (m, n)).astype(np.float32), 1),
        ("standard", rng.normal(0.0, 1.0, (n, m)).astype(np.float32), 0),
        ("offset_1e6", rng.normal(1e6, 1.0, (n, m)).astype(np.float32), 0),
    ]


# the layers of the stack of layers

def ln_usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def ln_stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(stable_var(x, axis=-1, keepdims=True) + EPS)


def block(x, w, ln, spread):
    return OFFSET + spread * jnp.tanh(ln(x) @ w)


def make_trajectory(ln, spread):
    def trajectory(x0, W):
        def step(x, w):
            return block(x, w, ln, spread), x

        return jax.lax.scan(step, x0, W)

    return trajectory


def make_grad(ln, spread):
    traj = make_trajectory(ln, spread)
    return jax.jit(jax.grad(lambda x0, W, c: jnp.sum(c * traj(x0, W)[0])))


def make_inputs(ln, spread):
    traj = make_trajectory(ln, spread)
    return jax.jit(lambda x0, W: traj(x0, W)[1])


def make_reference(spread):
    @jax.jit
    def reference_grad(xs, W, c):
        def step(g, inp):
            x, w = inp
            return jax.vjp(lambda a: block(a, w, ln_usual, spread), x)[1](g)[0], None

        return jax.lax.scan(step, c, (xs, W), reverse=True)[0]

    return reference_grad


def orthogonal(rng, L, D):
    q = np.linalg.qr(rng.normal(0.0, 1.0, (L, D, D)))[0]
    return (WSCALE * q).astype(np.float32)

def grid():
    print("0. the grid of float32 near mu^2, against sigma^2: the one-pass formula fails when sigma^2 is below one step")
    for name, mu, sigma in (("stock index", 4500.0, 0.5), ("robot joint", 1000.0, 0.01), ("problem", 1e6, 1.0)):
        step = float(np.spacing(np.float32(mu * mu)))
        print(f"   {name:12s} mu/sigma {mu / sigma:9.0f}   ulp(mu^2) {step:12.4g}   sigma^2 {sigma * sigma:10.4g}   below one step: {sigma * sigma < step}")


def problem():
    x32 = np.random.default_rng(0).normal(1e6, 1.0, 3000).astype(np.float32)
    x = jnp.asarray(x32)
    print("1. the problem: 3000 values around 10^6, float32")
    print("   float64 variance          ", float(np.var(x32.astype(np.float64))))
    print("   one-pass float32 formula  ", float(jnp.mean(x * x) - jnp.mean(x) ** 2))
    print("   stable_var                ", float(stable_var(x)))


def stock_index():
    rng = np.random.default_rng(0)
    quotes = np.round(rng.normal(4500.0, 0.5, 3000) * 4.0) / 4.0
    x = jnp.asarray(quotes.astype(np.float32))
    ref = float(np.var(np.asarray(x, dtype=np.float64)))
    naive_one = float(jnp.mean(x * x) - jnp.mean(x) ** 2)
    print("2. a stock index near 4500, steps of 0.25")
    print(f"   {ref:.4f} {naive_one:.4f} {float(stable_var(x)):.4f}   (float64, one-pass float32, stable_var)")


def shift():
    x = jnp.asarray([1000.0, 1000.01, 1000.02], dtype=jnp.float32)
    print("3. the shift of [1000.0, 1000.01, 1000.02]:", np.asarray(x - x[0]))


def two_sum_example():
    a, b = np.float32(16777216.0), np.float32(1.0)
    s, e = _two_sum(jnp.float32(a), jnp.float32(b))
    print("4. TwoSum of a = 16777216 and b = 1, float32: plain sum", float(a + b), "| s", float(s), "e", float(e))


def tree_example():
    v = jnp.asarray([1.0] + [1e-8] * 7, dtype=jnp.float32)
    exact = float(np.sum(np.asarray(v, dtype=np.float64)))
    print("5. eight numbers, 1.0 and seven of 1e-8: plain float32 sum", float(jnp.sum(v)),
          "| compensated tree", float(_compensated_tree(v)), "| exact", exact)


def kahan_example():
    def seq_sum(x):
        def step(carry, xi):
            return carry + xi, None
        s, _ = jax.lax.scan(step, jnp.zeros((), x.dtype), x)
        return s

    def kahan(x):
        def step(carry, xi):
            s, c = carry
            y = xi - c
            t = s + y
            c = (t - s) - y
            return (t, c), None
        (s, c), _ = jax.lax.scan(step, (jnp.zeros((), x.dtype), jnp.zeros((), x.dtype)), x)
        return s + c

    x = jnp.asarray([1.0] + [1e-7] * 100000, dtype=jnp.float32)
    print("6. 1.0 and 100000 numbers of 1e-7, float32: sequential sum", f"{float(seq_sum(x)):.8f}", "| Kahan", f"{float(kahan(x)):.8f}", "| exact 1.01")


def identical():
    print("7. the variance of identical values (stable_var, under jit)")
    for name, dt, val in identical_cases():
        x = jnp.asarray(np.full(3, val, dtype=dt))
        print(f"   {name:12s}", float(jax.jit(lambda a: stable_var(a))(x)))


def datasets():
    print("8. the four datasets, maximum relative error against float64: base eager, kernel eager, base jit, kernel jit")
    for name, data, axis in accuracy_cases(False):
        ref = np.var(data.astype(np.float64), axis=axis)
        x = jnp.asarray(data)
        row = [
            _max_rel(base(x, axis), ref),
            _max_rel(stable_var(x, axis=axis), ref),
            _max_rel(jax.jit(lambda a: base(a, axis))(x), ref),
            _max_rel(jax.jit(lambda a: stable_var(a, axis=axis))(x), ref),
        ]
        print(f"   {name:11s}", "  ".join(f"{v:.2e}" for v in row))


def layers():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    B, D, depths = 16, 64, [1, 4, 16, 64]
    rng = np.random.default_rng(0)
    c = rng.normal(0.0, 1.0, (B, D)).astype(np.float32)
    y0 = np.random.default_rng(1).normal(OFFSET, SPREAD, (B, D)).astype(np.float32)
    print("9. a stack of normalised layers, float32, offset 1000 and spread 0.004: error of the gradient, usual and with stable_var")
    ref_grad = make_reference(SPREAD)
    for L in depths:
        W = jnp.asarray(orthogonal(rng, L, D))
        a = (jnp.asarray(y0), W, jnp.asarray(c))
        res = []
        for ln in (ln_usual, ln_stable):
            xs = make_inputs(ln, SPREAD)(a[0], W)
            g64 = np.asarray(ref_grad(xs.astype(jnp.float64), W.astype(jnp.float64), a[2].astype(jnp.float64)))
            g32 = np.asarray(make_grad(ln, SPREAD)(*a), np.float64)
            res.append(np.linalg.norm(g32 - g64) / np.linalg.norm(g64))
        print(f"   {L:3d} layers   {res[0]:.2e}   {res[1]:.2e}")


if __name__ == "__main__":
    grid()
    problem()
    stock_index()
    shift()
    two_sum_example()
    tree_example()
    kahan_example()
    identical()
    datasets()
    layers()
