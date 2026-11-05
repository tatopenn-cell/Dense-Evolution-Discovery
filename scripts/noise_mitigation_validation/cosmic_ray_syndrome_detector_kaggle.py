import json
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pymatching", "scipy", "dense-evolution"], check=True)

import numpy as np
from scipy.optimize import nnls
import jax.numpy as jnp
import pymatching
from functools import lru_cache
from dense_evolution import cosmic_ray_burst_profile

D = 5
N = D * D
BASELINE = 0.01
K0 = 2
TAU_US = 300.0
SHOTS_PER_IMPACT = 480
WMAX = 100
WS = (10, 30, 100)
LAM = 0.1
TIMES = (10, 30, 100, 300, 1000)
SEED = 2026
RATIOS = (30.0, 100.0)


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
print("code checks ok (single errors corrected, distance 5 logical operators)", flush=True)

XY = np.array([(q // D, q % D) for q in range(N)])
ORDERS = [np.lexsort((np.arange(N), ((XY - XY[m]) ** 2).sum(1))) for m in range(N)]


@lru_cache(maxsize=None)
def burst(t, ratio):
    return float(cosmic_ray_burst_profile(jnp.array([t]), baseline_gamma=BASELINE,
                                          ratio_intermediate=ratio * 2.5 / 3.75, ratio_peak=ratio)[0])


def gammas(t, order, ratio):
    k = K0 + (N - K0) * (1 - np.exp(-t / TAU_US))
    g = np.full(N, BASELINE)
    for r, q in enumerate(order):
        g[q] = BASELINE + (burst(t, ratio) - BASELINE) * min(max(k - r, 0.0), 1.0)
    return g


def history(t, order, ratio):
    return np.array([gammas(t - (WMAX - 1 - j), order, ratio) if t - (WMAX - 1 - j) >= 0
                     else np.full(N, BASELINE) for j in range(WMAX)])


def marginals(g):
    pi, px, py, pz = twirled(g)
    return px + py, py + pz


P0X, P0Z = (float(v[0]) for v in marginals(np.array([BASELINE])))
A_AUG = {name: np.vstack([H.astype(float), np.sqrt(LAM) * np.eye(N)]) for name, H in (("x", HZ), ("z", HX))}


def estimate(freq, name, p0):
    y = -np.log(1 - 2 * np.clip(freq, 0.0, 0.49))
    b = np.concatenate([y, np.sqrt(LAM) * np.full(N, -np.log(1 - 2 * p0))])
    a, _ = nnls(A_AUG[name], b)
    return (1 - np.exp(-a)) / 2


def weight_of(p):
    p = np.clip(p, 1e-3, 0.49)
    return np.log((1 - p) / p)


def sample_rounds(gam, rng, shots):
    pi, px, py, _ = twirled(gam)
    u = rng.random((shots,) + gam.shape)
    ex = ((u >= pi) & (u < pi + px + py)).astype(np.uint8)
    ez = (u >= pi + px).astype(np.uint8)
    return ex, ez


rng = np.random.default_rng(SEED)
g_uni = np.full(N, 0.05)
ex, ez = sample_rounds(np.tile(g_uni, (400, 1)), rng, 20)
fx = ((ex @ HZ.T.astype(np.int64)) % 2).mean(axis=1)
est = np.array([estimate(f, "x", P0X) for f in fx]).mean()
true_px = float(marginals(g_uni)[0][0])
print(f"estimator check, uniform decay 0.05, 400 rounds: estimated X-flip prob {est:.4f} vs true {true_px:.4f}", flush=True)
assert abs(est - true_px) < 0.25 * true_px, (est, true_px)


def run_group(t, m, ratio, rng, shots):
    order = ORDERS[m]
    hist = history(t, order, ratio)
    ex_h, ez_h = sample_rounds(hist, rng, shots)
    sx = (np.einsum("swq,cq->swc", ex_h.astype(np.int64), HZ.astype(np.int64)) % 2)
    sz = (np.einsum("swq,cq->swc", ez_h.astype(np.int64), HX.astype(np.int64)) % 2)
    g_now = gammas(t, order, ratio)
    ex, ez = sample_rounds(g_now[None, :], rng, shots)
    ex, ez = ex[:, 0], ez[:, 0]
    wx, wz = weights(g_now)
    mo = (pymatching.Matching.from_check_matrix(HZ, weights=wx), pymatching.Matching.from_check_matrix(HX, weights=wz))
    blind = fails(ex, ez, mx0, mz0)
    oracle = fails(ex, ez, *mo)
    det = {}
    for W in WS:
        out = np.zeros(shots, dtype=bool)
        for s in range(shots):
            px_hat = estimate(sx[s, -W:].mean(axis=0), "x", P0X)
            pz_hat = estimate(sz[s, -W:].mean(axis=0), "z", P0Z)
            mm = (pymatching.Matching.from_check_matrix(HZ, weights=weight_of(px_hat)),
                  pymatching.Matching.from_check_matrix(HX, weights=weight_of(pz_hat)))
            out[s] = fails(ex[s:s + 1], ez[s:s + 1], *mm)[0]
        det[W] = out
    return blind.sum(), oracle.sum(), {W: det[W].sum() for W in WS}


t0 = time.time()
run_group(100, 12, RATIOS[0], np.random.default_rng(1), 20)
per_shot = (time.time() - t0) / 20
total = per_shot * SHOTS_PER_IMPACT * N * len(TIMES) * len(RATIOS)
print(f"timing: {per_shot * 1000:.1f} ms per shot, projected full run {total / 60:.0f} min", flush=True)
if total > 3 * 3600:
    raise SystemExit("projected run is too long, aborting before the full run")

results = {}
for ratio in RATIOS:
    rng = np.random.default_rng(SEED + int(ratio))
    rows = []
    print(f"\nPeak ratio R={ratio}, 5x5 surface code, syndrome-based zone estimate, "
          f"{N * SHOTS_PER_IMPACT} shots per point  ({time.time() - t0:.0f}s)", flush=True)
    print("  t (us)    blind    W=10     W=30     W=100    oracle   best-W gain  oracle gain", flush=True)
    for t in TIMES:
        tot = np.zeros(2 + len(WS))
        for m in range(N):
            b, o, d = run_group(t, m, ratio, rng, SHOTS_PER_IMPACT)
            tot += np.array([b, o] + [d[W] for W in WS])
        r = tot / (N * SHOTS_PER_IMPACT)
        rows.append([t] + r.tolist())
        best = min(r[2:])
        print(f"  {t:>6}   {r[0]:.4f}   {r[2]:.4f}   {r[3]:.4f}   {r[4]:.4f}   {r[1]:.4f}   "
              f"{1 - best / r[0]:8.1%}   {1 - r[1] / r[0]:8.1%}", flush=True)
    results[str(ratio)] = {"columns": ["t_us", "blind", "oracle", "W10", "W30", "W100"], "rows": rows}

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f)
print("done", flush=True)
