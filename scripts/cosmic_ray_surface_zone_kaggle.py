import json
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pymatching", "dense-evolution"], check=True)

import numpy as np
import jax.numpy as jnp
import pymatching
from functools import lru_cache
from dense_evolution import cosmic_ray_burst_profile

D = 5
N = D * D
BASELINE = 0.01
K0 = 2
TAU_US = 300.0
N_SHOTS = 2000
SEED = 2026
RATIOS = (30.0, 100.0)
TABLE_TIMES = (0.0, 10.0, 100.0, 300.0, 1000.0, 1500.0, 3000.0, 10000.0, 25000.0, 100000.0)
GRID = np.unique(np.concatenate([TABLE_TIMES, np.geomspace(1.0, 1e5, 24)]))


def build_code(d):
    hx, hz = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            qs = [r * d + c for r, c in ((i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j))
                  if 0 <= r < d and 0 <= c < d]
            typ = "X" if (i + j) % 2 == 0 else "Z"
            if len(qs) == 4:
                keep = True
            elif len(qs) == 2:
                keep = (typ == "X") == (i in (0, d))
            else:
                keep = False
            if keep:
                row = np.zeros(d * d, dtype=np.uint8)
                row[qs] = 1
                (hx if typ == "X" else hz).append(row)
    return np.array(hx), np.array(hz)


def gf2_rank(m):
    m = m.copy() % 2
    rank = 0
    for c in range(m.shape[1]):
        piv = next((r for r in range(rank, m.shape[0]) if m[r, c]), None)
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def find_logical(h_commute, h_span):
    cands = []
    for k in range(D):
        row = np.zeros(N, dtype=np.uint8)
        row[k * D:(k + 1) * D] = 1
        col = np.zeros(N, dtype=np.uint8)
        col[k::D] = 1
        cands += [row, col]
    for v in cands:
        if not ((h_commute @ v) % 2).any() and gf2_rank(np.vstack([h_span, v])) > gf2_rank(h_span):
            return v
    raise RuntimeError("no logical operator found")


HX, HZ = build_code(D)
assert HX.shape[0] == 12 and HZ.shape[0] == 12, (HX.shape, HZ.shape)
assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()
LZ = find_logical(HX, HZ)
LX = find_logical(HZ, HX)
assert (LX @ LZ) % 2 == 1
print("code ok: 25 qubits, 12 X and 12 Z checks, logical weights", int(LX.sum()), int(LZ.sum()), flush=True)


def twirled(g):
    s = np.sqrt(1 - g)
    return ((1 + s) / 2) ** 2, g / 4, g / 4, ((1 - s) / 2) ** 2


def weights(g):
    _, px, py, pz = twirled(g)
    w = lambda p: np.log((1 - np.clip(p, 1e-9, 0.5)) / np.clip(p, 1e-9, 0.5))
    return w(px + py), w(py + pz)


def matchings(g):
    wx, wz = weights(g)
    return (pymatching.Matching.from_check_matrix(HZ, weights=wx),
            pymatching.Matching.from_check_matrix(HX, weights=wz))


def fails(ex, ez, mx, mz):
    cx = mx.decode_batch(((ex @ HZ.T) % 2).astype(np.uint8))
    cz = mz.decode_batch(((ez @ HX.T) % 2).astype(np.uint8))
    return ((((ex ^ cx) @ LZ) % 2) | (((ez ^ cz) @ LX) % 2)).astype(bool)


def sample(g, n, rng):
    pi, px, py, _ = twirled(g)
    u = rng.random((n, N))
    return ((u >= pi) & (u < pi + px + py)).astype(np.uint8), (u >= pi + px).astype(np.uint8)


mx0, mz0 = matchings(np.full(N, BASELINE))
for q in range(N):
    e = np.zeros((1, N), dtype=np.uint8)
    e[0, q] = 1
    z0 = np.zeros((1, N), dtype=np.uint8)
    assert not fails(e, z0, mx0, mz0)[0] and not fails(z0, e, mx0, mz0)[0] and not fails(e, e, mx0, mz0)[0], q
pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
e = np.zeros((len(pairs), N), dtype=np.uint8)
for i, (a, b) in enumerate(pairs):
    e[i, a] = e[i, b] = 1
z0 = np.zeros_like(e)
assert not fails(e, z0, mx0, mz0).any() and not fails(z0, e, mx0, mz0).any()
print("checks ok: every single and double X/Z/Y error is corrected (distance 5)", flush=True)
rng0 = np.random.default_rng(SEED)
ex, ez = sample(np.full(N, BASELINE), 40000, rng0)
print("baseline failure rate, 40000 shots:", float(fails(ex, ez, mx0, mz0).mean()), flush=True)

XY = np.array([(q // D, q % D) for q in range(N)])
ORDERS = [np.lexsort((np.arange(N), ((XY - XY[m]) ** 2).sum(1))) for m in range(N)]


@lru_cache(maxsize=None)
def burst(t, ratio):
    return float(cosmic_ray_burst_profile(jnp.array([t]), baseline_gamma=BASELINE,
                                          ratio_intermediate=ratio * 2.5 / 3.75, ratio_peak=ratio)[0])


def gammas(t, order, ratio, k=None):
    k = K0 + (N - K0) * (1 - np.exp(-t / TAU_US)) if k is None else k
    g = np.full(N, BASELINE)
    for r, q in enumerate(order):
        g[q] = BASELINE + (burst(t, ratio) - BASELINE) * min(max(k - r, 0.0), 1.0)
    return g


def run_point(t, ratio, rng, blind_m):
    tot = np.zeros(4)
    for m in range(N):
        true = gammas(t, ORDERS[m], ratio)
        ex, ez = sample(true, N_SHOTS, rng)
        static = matchings(gammas(t, ORDERS[m], ratio, k=K0))
        track = matchings(true)
        wrong = matchings(gammas(t, ORDERS[(m + 12) % N], ratio))
        for i, mm in enumerate((blind_m, static, track, wrong)):
            tot[i] += fails(ex, ez, *mm).sum()
    return tot / (N * N_SHOTS)


results = {}
t0 = time.time()
blind_m = (mx0, mz0)
for ratio in RATIOS:
    rng = np.random.default_rng(SEED + int(ratio))
    curve = np.array([run_point(t, ratio, rng, blind_m) for t in GRID])
    results[str(ratio)] = {"t_us": GRID.tolist(), "blind_static_tracking_mislocated": curve.tolist()}
    print(f"\nPeak ratio R={ratio}, 5x5 surface code, K0={K0}, tau={TAU_US} us, "
          f"{N * N_SHOTS} shots per point  ({time.time() - t0:.0f}s)", flush=True)
    print("  t (us)    blind    static   tracking   mislocated   tracking gain", flush=True)
    for t in TABLE_TIMES:
        r = curve[list(GRID).index(t)]
        gain = 1 - r[2] / r[0] if r[0] > 0 else float("nan")
        print(f"  {t:>8.0f}   {r[0]:.4f}   {r[1]:.4f}   {r[2]:.4f}     {r[3]:.4f}       {gain:6.1%}", flush=True)
    avg = (getattr(np, "trapezoid", None) or np.trapz)(curve, GRID, axis=0) / GRID[-1]
    print(f"  Average failure over the first 100 ms: blind {avg[0]:.4f}, static {avg[1]:.4f}, "
          f"tracking {avg[2]:.4f}, mislocated {avg[3]:.4f}", flush=True)
    print(f"  Tracking gain {1 - avg[2] / avg[0]:.1%}, static gain {1 - avg[1] / avg[0]:.1%}, "
          f"mislocated vs blind {avg[3] / avg[0]:.2f}x", flush=True)

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f)
print("done", flush=True)
