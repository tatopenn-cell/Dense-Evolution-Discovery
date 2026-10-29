"""Reproduce every number of the paper on the float32 variance kernel.

Run with the checkout of Dense-Armor named in the paper on the path:

    PYTHONPATH=<checkout> python reproduce_float32_variance.py --out results.json

Contenders, all float32, eager and under jax.jit:

- naive: mean((x - mean(x))**2).
- base: max(mean(d**2) - mean(d)**2, 0), d = x - mean(x). The formula of
  tf.nn.moments in TensorFlow master, nn_impl.py lines 1480-1487, read on
  2026-10-29. A local copy: nothing is sent to TensorFlow.
- kernel: dense_armor.utility.stats.stable.stable_var.
- shift_kahan: the kernel written again here, to check that the paper
  describes the code (the two must agree bit for bit).
- shift_tree, base_kahan, shift_twoterm_tree: the variants of the ablation,
  with the same vmap-over-1D structure as the kernel.
"""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.stats.stable import stable_var


def kahan_sum(x):
    def step(carry, xi):
        s, c = carry
        y = xi - c
        t = s + y
        c = (t - s) - y
        return (t, c), None

    z = jnp.zeros((), x.dtype)
    (s, c), _ = jax.lax.scan(step, (z, z), x)
    return s + c


def over_axis(f1d):
    def f(x, axis):
        xm = jnp.moveaxis(x, axis, -1)
        shape = xm.shape[:-1]
        flat = xm.reshape(-1, xm.shape[-1])
        return jax.vmap(f1d)(flat).reshape(shape)

    return f


def naive(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    return jnp.mean(d * d, axis=axis)


def base(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    var = jnp.mean(d * d, axis=axis) - jnp.mean(d, axis=axis) ** 2
    return jnp.maximum(var, 0.0)


def _shift_center(v):
    y = v - v[0]
    m = jnp.sum(y) / v.shape[0]
    return y - m


def v_shift_kahan(v):
    d = _shift_center(v)
    s2 = kahan_sum(d * d)
    return jnp.maximum(s2 / v.shape[0], jnp.zeros_like(s2))


def v_shift_tree(v):
    d = _shift_center(v)
    s2 = jnp.sum(d * d)
    return jnp.maximum(s2 / v.shape[0], jnp.zeros_like(s2))


def v_shift_twoterm_tree(v):
    d = _shift_center(v)
    n = v.shape[0]
    s2 = jnp.sum(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


def v_base_kahan(v):
    n = v.shape[0]
    d = v - jnp.sum(v) / n
    s2 = kahan_sum(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


CONTENDERS = {
    "naive": naive,
    "base": base,
    "kernel": lambda x, axis: stable_var(x, axis=axis),
    "shift_kahan": over_axis(v_shift_kahan),
    "shift_tree": over_axis(v_shift_tree),
    "base_kahan": over_axis(v_base_kahan),
    "shift_twoterm_tree": over_axis(v_shift_twoterm_tree),
}


def run(f, x, axis, jit):
    if jit:
        return jax.jit(lambda a: f(a, axis))(x)
    return f(x, axis)


def datasets():
    rng = np.random.default_rng(0)
    spec = [
        ("D1", 1e3, 0.01, (1000, 64), 0),
        ("D2", 1e4, 1.0, (64, 1000), 1),
        ("D3", 0.0, 1.0, (1000, 64), 0),
        ("D4", 1e6, 1.0, (1000, 64), 0),
    ]
    out = {}
    for name, mu, sd, shape, axis in spec:
        x = np.asarray(jnp.asarray(rng.normal(mu, sd, shape), jnp.float32))
        out[name] = (x, axis, mu, sd, shape)
    return out


def max_rel_err(v, x, axis):
    ref = np.var(x.astype(np.float64), axis=axis)
    return float(np.max(np.abs(np.asarray(v, np.float64) - ref) / ref))


def timing(f, x, runs):
    g = jax.jit(lambda a: f(a, 1))
    g(x).block_until_ready()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        g(x).block_until_ready()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.subtract(*np.percentile(ts, [75, 25])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    shape_speed = (512, 512) if args.smoke else (4096, 4096)

    res = {
        "env": {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "x64": bool(jax.config.jax_enable_x64),
        }
    }

    ident = {}
    for val in (1e10, 7.5e10, 1e15, 1e38):
        x = jnp.full(3, val, jnp.float32)
        ident[f"f32_{val:g}"] = {
            k: {"eager": float(run(f, x, 0, False)), "jit": float(run(f, x, 0, True))}
            for k, f in CONTENDERS.items()
        }
    z = jnp.full(3, 7.5e10 + 7.5e10j, jnp.complex64)
    ident["complex64_7.5e10"] = {
        "kernel": {"eager": float(stable_var(z)), "jit": float(jax.jit(stable_var)(z))}
    }
    zc = np.array([1 + 1j, 2 + 3j, -1 + 0.5j, 4 - 2j], np.complex64)
    ident["complex_check"] = {
        "kernel": float(stable_var(zc)),
        "numpy": float(np.var(zc)),
    }
    res["identical"] = ident

    acc, same = {}, {}
    for name, (x, axis, mu, sd, shape) in datasets().items():
        xj = jnp.asarray(x)
        acc[name] = {
            "mu": mu,
            "sd": sd,
            "shape": list(shape),
            "axis": axis,
            **{
                k: {
                    "eager": max_rel_err(run(f, xj, axis, False), x, axis),
                    "jit": max_rel_err(run(f, xj, axis, True), x, axis),
                }
                for k, f in CONTENDERS.items()
            },
        }
        same[name] = bool(
            np.array_equal(
                np.asarray(run(CONTENDERS["kernel"], xj, axis, False)),
                np.asarray(run(CONTENDERS["shift_kahan"], xj, axis, False)),
            )
        )
        if name == "D4":
            y32 = x - x[:1]
            y64 = x.astype(np.float64) - x[:1].astype(np.float64)
            res["d4_shift_exact_fraction"] = float(np.mean(y32.astype(np.float64) == y64))
    res["accuracy"] = acc
    res["kernel_equals_reimplementation_bitwise"] = same

    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.normal(0, 1, shape_speed), jnp.float32)
    sessions = []
    for _ in range(args.sessions):
        sessions.append(
            {k: dict(zip(("median_ms", "iqr_ms"), timing(f, xs, args.runs)))
             for k, f in CONTENDERS.items()}
        )
    res["speed"] = {
        "shape": list(shape_speed),
        "axis": 1,
        "runs": args.runs,
        "sessions": sessions,
        "ratio_vs_base": {
            k: [s[k]["median_ms"] / s["base"]["median_ms"] for s in sessions]
            for k in CONTENDERS
        },
    }
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)
    print("D4 shift exact fraction:", res["d4_shift_exact_fraction"])
    print("kernel == reimplementation:", same)
    print("ratio kernel/base per session:", [round(r, 2) for r in res["speed"]["ratio_vs_base"]["kernel"]])


if __name__ == "__main__":
    main()
