"""Can the Richardson nudge be driven by the measured expectation values alone?

The library's predictive ZNE nudges the 3-point Richardson coefficients from a
signal computed on the density matrix. Here the signal is computed from the
three measured expectation values y1, y2, y3 themselves, so no tomography is
needed and any number of qubits works:
    nonlinearity = (|y2 - y3| - |y1 - y2|) / (|y2 - y3| + |y1 - y2|), rectified.

Benchmark in the style of Majumdar et al. (arXiv:2307.05203, Sect. IV, Fig. 6):
brickwork circuits on 4 qubits, local noise after every gate (depolarizing or
amplitude damping) with probability p scaled by the noise factor, factors 1, 2,
3, 8000 shots per factor, observable Z on qubit 0, exact density-matrix
simulation. Baselines: Richardson, linear least squares, exponential with the
asymptote fixed to 0 (traceless observable, as in Giurgica-Tiron et al.,
arXiv:2005.10921, Eq. 35). Controls: the same nudges shuffled across circuits,
and one constant nudge on every circuit. Metric: RMSE against the exact
zero-noise value.

Also tested: a curvature test in the spirit of Majumdar et al.'s advice to
check whether the data vary enough before choosing an extrapolator. If the
second difference y1 - 2 y2 + y3 is within 2 shot-noise standard deviations of
zero, use the linear fit, otherwise Richardson.
"""
import numpy as np

N_QUBITS = 4
FACTORS = np.array([1.0, 2.0, 3.0])
SHOTS = 8000
I2 = np.eye(2)
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1.0, -1.0]).astype(complex)]
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def random_su2(rng):
    q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def kraus(channel, p):
    if channel == "depolarizing":
        return [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * P for P in PAULI]
    return [np.array([[1, 0], [0, np.sqrt(1 - p)]]), np.array([[0, np.sqrt(p)], [0, 0]])]


def apply(rho, op, qubits):
    k = len(qubits)
    t = rho.reshape([2] * (2 * N_QUBITS))
    op = op.reshape([2] * (2 * k))
    axes = list(qubits)
    t = np.tensordot(op, t, axes=(list(range(k, 2 * k)), axes))
    t = np.moveaxis(t, list(range(k)), axes)
    t = np.tensordot(t, op.conj(), axes=([N_QUBITS + q for q in qubits], list(range(k, 2 * k))))
    t = np.moveaxis(t, list(range(2 * N_QUBITS - k, 2 * N_QUBITS)), [N_QUBITS + q for q in qubits])
    return t.reshape(2 ** N_QUBITS, 2 ** N_QUBITS)


def channel_on(rho, ks, q):
    return sum(apply(rho, K, [q]) for K in ks)


def circuit(rng, depth):
    layers = []
    for d in range(depth):
        singles = [random_su2(rng) for _ in range(N_QUBITS)]
        pairs = [(i, i + 1) for i in range(d % 2, N_QUBITS - 1, 2)]
        layers.append((singles, pairs))
    return layers


def expectation(layers, channel, p):
    rho = np.zeros((2 ** N_QUBITS,) * 2, complex)
    rho[0, 0] = 1
    ks = kraus(channel, p) if p > 0 else None
    for singles, pairs in layers:
        for q, U in enumerate(singles):
            rho = apply(rho, U, [q])
            if ks:
                rho = channel_on(rho, ks, q)
        for a, b in pairs:
            rho = apply(rho, CZ, [a, b])
            if ks:
                rho = channel_on(channel_on(rho, ks, a), ks, b)
    z0 = np.kron(PAULI[2], np.eye(2 ** (N_QUBITS - 1)))
    return float(np.real(np.trace(rho @ z0)))


def coeffs(rect, s=0.5):
    return np.array([3 - s * rect, -3 + 2 * s * rect, 1 - s * rect])


def nudge_signal(y):
    g12, g23 = abs(y[0] - y[1]), abs(y[1] - y[2])
    return max((g23 - g12) / (g23 + g12 + 1e-12), 0.0)


def linear(y):
    return np.polyval(np.polyfit(FACTORS, y, 1), 0.0)


def exponential(y):
    s = np.sign(y.sum()) or 1.0
    z = np.log(np.maximum(np.abs(y), 1e-6))
    return s * np.exp(np.polyval(np.polyfit(FACTORS, z, 1), 0.0))


def curvature_select(y, z=2.0):
    sigma = np.sqrt(np.maximum(1 - y ** 2, 1e-12) / SHOTS)
    curvature = y[0] - 2 * y[1] + y[2]
    noise = np.sqrt(sigma[0] ** 2 + 4 * sigma[1] ** 2 + sigma[2] ** 2)
    return y @ coeffs(0.0) if abs(curvature) > z * noise else linear(y)


def run_cell(channel, depth, p, n_circuits, rng):
    exact, ys = [], []
    for _ in range(n_circuits):
        layers = circuit(rng, depth)
        exact.append(expectation(layers, channel, 0.0))
        e = np.array([expectation(layers, channel, p * f) for f in FACTORS])
        ys.append(2 * rng.binomial(SHOTS, (1 + np.clip(e, -1, 1)) / 2) / SHOTS - 1)
    exact, ys = np.array(exact), np.array(ys)
    rects = np.array([nudge_signal(y) for y in ys])
    shuffled = rng.permutation(rects)
    constant = rects[rects > 0].mean() if (rects > 0).any() else 0.0
    est = {
        "richardson": ys @ coeffs(0.0),
        "linear": np.array([linear(y) for y in ys]),
        "exponential": np.array([exponential(y) for y in ys]),
        "nudge": np.array([y @ coeffs(r) for y, r in zip(ys, rects)]),
        "shuffled": np.array([y @ coeffs(r) for y, r in zip(ys, shuffled)]),
        "constant": ys @ coeffs(constant),
        "curv_select": np.array([curvature_select(y) for y in ys]),
    }
    return {k: float(np.sqrt(np.mean((v - exact) ** 2))) for k, v in est.items()}, float((rects > 0).mean())


def main(n_circuits=100, cells=None):
    rng = np.random.default_rng(2307)
    cells = cells or [(c, d, p) for c in ("depolarizing", "amplitude_damping") for d in (4, 12) for p in (0.005, 0.02)]
    names = ["richardson", "linear", "exponential", "nudge", "shuffled", "constant", "curv_select"]
    print(f"{'channel':<18}{'depth':>6}{'p':>7}{'active':>8}" + "".join(f"{n:>13}" for n in names))
    for channel, depth, p in cells:
        rmse, active = run_cell(channel, depth, p, n_circuits, rng)
        print(f"{channel:<18}{depth:>6}{p:>7.3f}{active:>8.2f}" + "".join(f"{rmse[n]:>13.4f}" for n in names))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main(n_circuits=5, cells=[("depolarizing", 4, 0.02)])
    else:
        main()
