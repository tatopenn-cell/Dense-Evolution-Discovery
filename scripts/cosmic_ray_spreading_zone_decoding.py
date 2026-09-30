"""
Zone-weighted decoding when the hot zone spreads during the burst.

Extends cosmic_ray_zone_weighted_decoding.py (same Pauli-twirled decay errors,
same exact coset-level maximum-likelihood decoders) with the spatial evolution
reported in arXiv:2104.05219: the burst first hits a small patch, then the
hot zone grows until the whole patch is elevated (~1.5 ms), then the strength
recovers with a ~25 ms time constant.

Model, with the assumptions stated: the 7 qubits are visited by the front in a
spreading order starting at the impact qubit (the Steane block has no
geometry here, so the order is averaged over random permutations). The front
covers k(t) = K0 + (7-K0)*(1 - exp(-t/TAU_SPREAD_US)) qubits, K0 = 2 at impact;
the qubit at order r has decay probability
    baseline + (burst(t) - baseline) * clip(k(t) - r, 0, 1),
where burst(t) is cosmic_ray_burst_profile at peak ratio R. K0 and
TAU_SPREAD_US are chosen here, not fitted; the paper gives a ~180 us estimate
for the initial spread and saturation around 1-1.5 ms.

Decoders (all maximum-likelihood on stabilizer cosets, exact):
  blind:      baseline prior on every qubit.
  static:     knows the initial patch (first K0 qubits) and the current
              strength, but not that the zone has spread.
  tracking:   knows the current zone and strength at every time.
  mislocated: tracking prior built on a different, wrong spreading order.
"""
import numpy as np

import cosmic_ray_zone_weighted_decoding as z

K0 = 2
TAU_SPREAD_US = 300.0
N_ORDERS = 120
SEED = 2026
TIMES_US = (0.0, 10.0, 100.0, 300.0, 1000.0, 1500.0, 3000.0, 10000.0, 25000.0, 100000.0)
INTEGRAL_GRID_US = np.concatenate([[0.0], np.geomspace(1.0, 1e5, 24)])


def front(t_us):
    return K0 + (z.N - K0) * (1.0 - np.exp(-t_us / TAU_SPREAD_US))


def gammas_at(t_us, order, ratio, k=None):
    g_burst = z.hot_gamma(t_us, ratio)
    k = front(t_us) if k is None else k
    g = np.full(z.N, z.BASELINE_P)
    for r, q in enumerate(order):
        g[q] = z.BASELINE_P + (g_burst - z.BASELINE_P) * min(max(k - r, 0.0), 1.0)
    return g


def failures_at(t_us, order, wrong, ratio, q_blind):
    q_true = z.coset_probs(gammas_at(t_us, order, ratio))
    q_static = z.coset_probs(gammas_at(t_us, order, ratio, k=K0))
    q_wrong = z.coset_probs(gammas_at(t_us, wrong, ratio))
    return np.array([
        z.failure_probability(q_true, q_blind),
        z.failure_probability(q_true, q_static),
        z.failure_probability(q_true, q_true),
        z.failure_probability(q_true, q_wrong),
    ])


def orders():
    rng = np.random.default_rng(SEED)
    return [(rng.permutation(z.N), rng.permutation(z.N)) for _ in range(N_ORDERS)]


def curve(times, ratio, order_pairs):
    q_blind = z.coset_probs(z.gammas((), 0))
    out = np.zeros((len(times), 4))
    for i, t in enumerate(times):
        out[i] = np.mean([failures_at(t, o, w, ratio, q_blind) for o, w in order_pairs], axis=0)
    return out


if __name__ == "__main__":
    order_pairs = [(o, w) for o, w in orders() if not np.array_equal(o, w)]
    for ratio in (30.0, 100.0):
        print(f"\nPeak ratio R={ratio}, K0={K0}, spread time {TAU_SPREAD_US:.0f} us, "
              f"{len(order_pairs)} random spreading orders")
        print("  t (us)    zone size   blind    static   tracking   mislocated   tracking gain")
        res = curve(TIMES_US, ratio, order_pairs)
        for t, r in zip(TIMES_US, res):
            gain = 1 - r[2] / r[0] if r[0] > 0 else float('nan')
            print(f"  {t:>8.0f}   {front(t):>5.2f}      {r[0]:.4f}   {r[1]:.4f}   {r[2]:.4f}     "
                  f"{r[3]:.4f}       {gain:6.1%}")
        grid = curve(INTEGRAL_GRID_US, ratio, order_pairs)
        avg = np.trapezoid(grid, INTEGRAL_GRID_US, axis=0) / INTEGRAL_GRID_US[-1]
        print(f"  Average failure over the first 100 ms: blind {avg[0]:.4f}, static {avg[1]:.4f}, "
              f"tracking {avg[2]:.4f}, mislocated {avg[3]:.4f}")
        print(f"  Tracking gain {1 - avg[2] / avg[0]:.1%}, static gain {1 - avg[1] / avg[0]:.1%}, "
              f"mislocated vs blind {avg[3] / avg[0]:.2f}x")
