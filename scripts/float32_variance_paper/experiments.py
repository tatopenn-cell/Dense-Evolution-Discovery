"""Experiments of the paper "Variance in float32 with a first-element shift, a local centring and a
compensated tree sum". Self-contained: it needs only numpy and jax, and carries its own copy of the
kernel (the same operations as ``stable_var`` of the library, equal to it bit for bit on the datasets
D1 to D4 of the note).

    python experiments.py --out results_cpu.json            # every experiment
    python experiments.py --only kappa,layernorm --out r.json

Experiments (names for --only):
  datasets   the four datasets of the note, 30 seeds, maximum error in relative terms and in ulp
  kappa      a scan of the condition number kappa = |mean| / std of float32 data
  layernorm  a layer normalisation (forward and gradient) on rows with a large offset
  layernorm_centred  the same, with the mean subtracted on the data shifted by the first element
  longn      long arrays (up to 1e7 values) with a large offset
  bound      the bound of the compensated tree sum against the exact sum
  timing     speed on a [4096, 4096] array
"""

import argparse
import json
import math
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

U = 2.0**-24  # unit roundoff of float32


# ---------------------------------------------------------------- the kernel and its competitors
def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def comp_tree(v):
    """Sum of a one-dimensional array by a tree of TwoSum, with the carries carried up the tree."""
    S = v
    C = jnp.zeros_like(v)
    while S.shape[0] > 1:
        if S.shape[0] % 2 == 1:
            S = jnp.concatenate([S, jnp.zeros((1,), S.dtype)])
            C = jnp.concatenate([C, jnp.zeros((1,), C.dtype)])
        s1, e1 = two_sum(S[0::2], S[1::2])
        s2, e2 = two_sum(C[0::2], C[1::2])
        s3, e3 = two_sum(s1, s2)
        S = s3
        C = e1 + e2 + e3
    return S[0] + C[0]


def kahan_sum(x):
    def step(carry, v):
        s, c = carry
        y = v - c
        t = s + y
        return (t, (t - s) - y), None

    (s, _), _ = jax.lax.scan(step, (jnp.zeros((), x.dtype), jnp.zeros((), x.dtype)), x)
    return s


def over_axis(f1d):
    def f(x, axis):
        xm = jnp.moveaxis(x, axis, -1)
        flat = xm.reshape(-1, xm.shape[-1])
        return jax.vmap(f1d)(flat).reshape(xm.shape[:-1])

    return f


def kernel_1d(v):
    n = v.shape[0]
    y = v - v[0]
    m = jnp.sum(y) / n
    d = y - m
    s2 = comp_tree(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


def kahan_1d(v):
    n = v.shape[0]
    y = v - v[0]
    m = jnp.sum(y) / n
    d = y - m
    s2 = kahan_sum(d * d)
    return jnp.maximum(s2 / n, jnp.zeros_like(s2))


kernel = over_axis(kernel_1d)
kahan = over_axis(kahan_1d)


def naive(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    return jnp.mean(d * d, axis=axis)


def reference(x, axis):
    """The corrected two-pass formula of Chan, Golub and LeVeque (1983): sum of d^2 minus the square of the sum of d, no shift."""
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    return jnp.maximum(jnp.mean(d * d, axis=axis) - jnp.mean(d, axis=axis) ** 2, 0.0)


METHODS = {"naive": naive, "reference": reference, "kernel": kernel}


def run(f, x, axis, jit):
    if jit:
        return jax.jit(lambda a: f(a, axis))(x)
    return f(x, axis)


def truth(x, axis):
    return np.var(np.asarray(x).astype(np.float64), axis=axis)


def rel_err(v, ref):
    v = np.asarray(v, np.float64)
    with np.errstate(all="ignore"):
        e = np.abs(v - ref) / ref
    return np.where(np.isfinite(e), e, np.inf)


def ulp_err(v, ref):
    ulp = np.spacing(ref.astype(np.float32)).astype(np.float64)
    with np.errstate(all="ignore"):
        e = np.abs(np.asarray(v, np.float64) - ref) / ulp
    return np.where(np.isfinite(e), e, np.inf)


# ---------------------------------------------------------------- 1. the datasets of the note
SPEC = {
    "D1": (1e3, 0.01, (1000, 64), 0),
    "D2": (1e4, 1.0, (64, 1000), 1),
    "D3": (0.0, 1.0, (1000, 64), 0),
    "D4": (1e6, 1.0, (1000, 64), 0),
}


def make(name, seed):
    mu, sd, shape, axis = SPEC[name]
    rng = np.random.default_rng(seed)
    return np.asarray(jnp.asarray(rng.normal(mu, sd, shape), jnp.float32)), axis


def exp_datasets(seeds=30):
    out = {}
    for name in SPEC:
        row = {m: {"rel": [], "ulp": []} for m in METHODS}
        for seed in range(seeds):
            x, axis = make(name, seed)
            ref = truth(x, axis)
            for m, f in METHODS.items():
                v = run(f, x, axis, True)
                row[m]["rel"].append(float(rel_err(v, ref).max()))
                row[m]["ulp"].append(float(ulp_err(v, ref).max()))
        k, r = np.array(row["kernel"]["rel"]), np.array(row["reference"]["rel"])
        out[name] = {
            "mean_max_rel": {m: float(np.mean(row[m]["rel"])) for m in METHODS},
            "mean_max_ulp": {m: float(np.mean(row[m]["ulp"])) for m in METHODS},
            "kernel_le_reference_seeds": int((k <= r).sum()),
            "median_ratio_kernel_over_reference": float(np.median(k / r)),
            "seeds": seeds,
        }
        # eager against jit for seed 0
        x, axis = make(name, 0)
        ref = truth(x, axis)
        out[name]["seed0_eager_vs_jit_max_rel"] = {
            m: [float(rel_err(run(f, x, axis, j), ref).max()) for j in (False, True)]
            for m, f in METHODS.items()
        }
    return out


# ---------------------------------------------------------------- 2. the condition number scan
def exp_kappa():
    rows = []
    n, cols = 1000, 32
    for mu in (1e3, 1e4, 1e5, 1e6, 3e6, 1e7, 3e7, 1e8):
        spacing = float(np.spacing(np.float32(mu)))
        for mult in (1, 2, 4, 16, 64, 256, 1024):
            sd = spacing * mult
            rng = np.random.default_rng(0)
            x = np.asarray(jnp.asarray(rng.normal(mu, sd, (n, cols)), jnp.float32))
            ref = truth(x, 0)
            if not np.all(ref > 0):
                continue
            kappa = float(
                np.median(np.abs(x.astype(np.float64).mean(0)) / np.sqrt(ref))
            )
            row = {"mu": mu, "sd": sd, "kappa": kappa}
            for m, f in METHODS.items():
                v = run(f, x, 0, True)
                e = rel_err(v, ref)
                row[m] = {
                    "max_rel": float(e.max()),
                    "median_rel": float(np.median(e)),
                    "bad_columns": int(np.sum(e > 0.1)),
                }
            rows.append(row)
    return rows


# ---------------------------------------------------------------- 2b. long arrays
def exp_longn():
    rows = []
    for n in (10**4, 10**5, 10**6, 10**7):
        for mu, sd in ((1e4, 1.0), (1e6, 1.0), (1e6, 0.25)):
            rng = np.random.default_rng(0)
            x = np.asarray(jnp.asarray(rng.normal(mu, sd, (n, 4)), jnp.float32))
            ref = truth(x, 0)
            row = {"n": n, "mu": mu, "sd": sd}
            for m, f in METHODS.items():
                row[m] = float(rel_err(run(f, x, 0, True), ref).max())
            rows.append(row)
    return rows


# ---------------------------------------------------------------- 3. a layer normalisation
EPS = 1e-5


def ln_forward_with(var_fn):
    def f(x, w):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = var_fn(x)[..., None]
        xh = (x - mean) / jnp.sqrt(var + EPS)
        return jnp.sum(w * xh)

    return f


def ln_float64(x, w):
    x64 = x.astype(np.float64)
    mean = x64.mean(-1, keepdims=True)
    var = x64.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xh = (x64 - mean) * inv
    g = w.astype(np.float64)
    dx = inv * (g - g.mean(-1, keepdims=True) - xh * (g * xh).mean(-1, keepdims=True))
    return xh, dx


def exp_layernorm():
    rows = []
    B, D = 64, 1024
    for mu, mult in (
        (1e3, 64),
        (1e5, 64),
        (1e6, 16),
        (3e6, 4),
        (1e7, 4),
        (3e7, 1),
        (1e8, 1),
    ):
        spacing = float(np.spacing(np.float32(mu)))
        rng = np.random.default_rng(1)
        x = np.asarray(jnp.asarray(rng.normal(mu, spacing * mult, (B, D)), jnp.float32))
        w = rng.normal(0, 1, (B, D)).astype(np.float32)
        xh64, dx64 = ln_float64(x, w)
        row = {"mu": mu, "sd": spacing * mult}
        for name, var_fn in (
            ("reference", lambda a: reference(a, -1)),
            ("kernel", lambda a: kernel(a, -1)),
        ):
            f = ln_forward_with(var_fn)
            xj, wj = jnp.asarray(x), jnp.asarray(w)

            def fwd(a, b, var_fn=var_fn):
                mean = jnp.mean(a, axis=-1, keepdims=True)
                var = var_fn(a)[..., None]
                return (a - mean) / jnp.sqrt(var + EPS)

            xh = np.asarray(jax.jit(fwd)(xj, wj), np.float64)
            grad = np.asarray(jax.jit(jax.grad(f))(xj, wj), np.float64)
            std_out = xh.std(axis=-1)
            row[name] = {
                "forward_max_abs_err": float(np.max(np.abs(xh - xh64))),
                "rows_with_output_std_off_by_10pct": int(
                    np.sum(np.abs(std_out - 1.0) > 0.1)
                ),
                "nan_or_inf": int(np.sum(~np.isfinite(xh))),
                "grad_max_abs_err": float(np.max(np.abs(grad - dx64))),
                "grad_nan_or_inf": int(np.sum(~np.isfinite(grad))),
                "grad_scale": float(np.max(np.abs(dx64))),
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 3b. a layer normalisation centred on the shift
def ln_centred_with(var_fn):
    """The mean is subtracted on the shifted data: (x - x0) - mean(x - x0), exact when x0 is close to x."""

    def f(x):
        y = x - x[..., :1]
        d = y - jnp.mean(y, axis=-1, keepdims=True)
        return d / jnp.sqrt(var_fn(x)[..., None] + EPS)

    return f


def ln_standard_with(var_fn):
    def f(x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var_fn(x)[..., None] + EPS)

    return f


def exp_layernorm_centred():
    rows = []
    B, D = 64, 1024
    for mu, mult in (
        (1e3, 64),
        (1e5, 64),
        (1e6, 16),
        (3e6, 4),
        (1e7, 4),
        (3e7, 1),
        (1e8, 1),
    ):
        spacing = float(np.spacing(np.float32(mu)))
        rng = np.random.default_rng(1)
        x = np.asarray(jnp.asarray(rng.normal(mu, spacing * mult, (B, D)), jnp.float32))
        w = rng.normal(0, 1, (B, D)).astype(np.float32)
        xh64, dx64 = ln_float64(x, w)
        xj, wj = jnp.asarray(x), jnp.asarray(w)
        row = {"mu": mu, "sd": spacing * mult}
        for name, build, var_fn in (
            ("standard_reference", ln_standard_with, lambda a: reference(a, -1)),
            ("standard_kernel", ln_standard_with, lambda a: kernel(a, -1)),
            ("centred_kernel", ln_centred_with, lambda a: kernel(a, -1)),
        ):
            fwd = build(var_fn)
            xh = np.asarray(jax.jit(fwd)(xj), np.float64)
            grad = np.asarray(
                jax.jit(jax.grad(lambda a, b: jnp.sum(b * fwd(a))))(xj, wj), np.float64
            )
            row[name] = {
                "forward_max_abs_err": float(np.max(np.abs(xh - xh64))),
                "grad_max_abs_err": float(np.max(np.abs(grad - dx64))),
                "nan_or_inf": int(
                    np.sum(~np.isfinite(xh)) + np.sum(~np.isfinite(grad))
                ),
                "grad_scale": float(np.max(np.abs(dx64))),
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 4. the bound of the compensated tree
def np_two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def np_comp_tree(v):
    S = v.astype(np.float32)
    C = np.zeros_like(S)
    levels = 0
    while S.shape[0] > 1:
        if S.shape[0] % 2 == 1:
            S = np.concatenate([S, np.zeros(1, np.float32)])
            C = np.concatenate([C, np.zeros(1, np.float32)])
        s1, e1 = np_two_sum(S[0::2], S[1::2])
        s2, e2 = np_two_sum(C[0::2], C[1::2])
        s3, e3 = np_two_sum(s1, s2)
        S = s3
        C = (e1 + e2 + e3).astype(np.float32)
        levels += 1
    return np.float32(S[0] + C[0]), levels


def exp_bound(trials=300):
    rng = np.random.default_rng(0)
    families = {
        "positive": lambda n: rng.uniform(0.0, 1.0, n),
        "normal (cancellation)": lambda n: rng.normal(0.0, 1.0, n),
        "wide exponent range": lambda n: (
            rng.normal(0.0, 1.0, n) * 10.0 ** rng.integers(-6, 7, n)
        ),
        "large cancelling pairs": lambda n: np.concatenate(
            [np.array([1e6, -1e6]), rng.normal(0, 1, n - 2)]
        )[rng.permutation(n)],
        "alternating huge and small": lambda n: (
            np.where(np.arange(n) % 2 == 0, 1e5, -1e5) + rng.normal(0, 1, n)
        ),
    }
    out = []
    for fam, gen in families.items():
        for n in (16, 256, 4096):
            worst = -np.inf
            worst_plain = -np.inf
            for _ in range(trials):
                v = gen(n).astype(np.float32)
                exact = math.fsum(v.astype(np.float64))
                sabs = float(np.abs(v.astype(np.float64)).sum())
                res, L = np_comp_tree(v)
                err = abs(float(res) - exact)
                worst = max(worst, (err - U * abs(exact)) / (L * U * U * sabs))
                plain = float(np.sum(v, dtype=np.float32))
                worst_plain = max(worst_plain, abs(plain - exact) / (L * U * U * sabs))
            out.append(
                {
                    "family": fam,
                    "n": n,
                    "levels": L,
                    "max_ratio": float(worst),
                    "plain_sum_max_ratio": float(worst_plain),
                }
            )
    return out


# ---------------------------------------------------------------- 5. speed
def exp_timing(runs=15):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(1e3, 1.0, (4096, 4096)), jnp.float32)
    x.block_until_ready()
    out = {}
    for name, f in (
        ("reference", reference),
        ("kernel", kernel),
        ("kahan_scan", kahan),
    ):
        g = jax.jit(lambda a, f=f: f(a, 1))
        g(x).block_until_ready()
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            g(x).block_until_ready()
            ts.append((time.perf_counter() - t) * 1e3)
        out[name] = float(np.median(ts))
    out["ratio_kernel_over_reference"] = out["kernel"] / out["reference"]
    out["ratio_kahan_over_reference"] = out["kahan_scan"] / out["reference"]
    return out


EXPERIMENTS = {
    "datasets": exp_datasets,
    "kappa": exp_kappa,
    "longn": exp_longn,
    "layernorm": exp_layernorm,
    "layernorm_centred": exp_layernorm_centred,
    "bound": exp_bound,
    "timing": exp_timing,
}


def main():
    ap = argparse.ArgumentParser()
    import os

    ap.add_argument(
        "--out",
        default="/kaggle/working/results_gpu.json"
        if os.path.isdir("/kaggle/working")
        else "results.json",
    )
    ap.add_argument("--only", default=",".join(EXPERIMENTS))
    args = ap.parse_args()
    res = {
        "environment": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "x64": bool(jax.config.jax_enable_x64),
            "python": platform.python_version(),
            "machine": platform.machine(),
        }
    }
    for name in args.only.split(","):
        t = time.time()
        res[name] = EXPERIMENTS[name]()
        print(f"{name}: done in {time.time() - t:.0f} s", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
