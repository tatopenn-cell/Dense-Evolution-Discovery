"""Does bounding the ZNE zero-noise estimate to the physically valid range
[-1, 1] actually help, on Dense-Evolution's own real noise model?

Real method (Miranskyy, Sorrenti, Thind, Gravel, arXiv:2604.24475, "Improving
Zero-Noise Extrapolation via Physically Bounded Models"): reparametrize the
exponential ZNE model so the zero-noise value is an explicit parameter,

    E(lambda) = a + (zeta - a) * exp(-c*lambda),   E(0) = zeta

then fit under constraints -1 <= a <= 1, -1 <= zeta <= 1, c > 0 (L-BFGS-B),
instead of the usual unconstrained a + b*exp(-c*lambda) fit. Dense-Evolution's
own `dense_evolution/mitigation/zne.py` has NO exponential-family ZNE model
today (only `polynomial_extrapolate`/`richardson_extrapolate`), so this is a
genuinely new comparison, not just "add a bound to what's already there".

Test case: real 2-qubit Bell circuit (Z0 Z1 observable, ideal value +1,
bounded in [-1, 1] exactly as the paper's setting), noise amplified via
Dense-Evolution's own depolarizing NoiseModel at p_eff = p_base * lambda
(the same scaling convention already used throughout this project's ZNE
examples). Few noise points (n=3, {1,2,3}) and a noisy regime (p_base high
enough that the exponential decay is not perfectly clean) are exactly the
conditions the paper says make the unbounded fit misbehave.
"""
import numpy as np
from scipy.optimize import curve_fit, minimize

from dense_evolution import DenseSVSimulator, QASMParser
from dense_evolution.registry import NoiseModel
from dense_evolution.observables import pauli_expectation

QASM = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
CIRCUIT = QASMParser().parse(QASM)
SCALES = np.array([1.0, 2.0, 3.0])
P_BASE = 0.25
N_TRAJECTORIES = 200


def ideal_sv():
    sim = DenseSVSimulator(2)
    sim.run_circuit_jit(CIRCUIT.to_tuples())
    return np.asarray(sim.get_statevector())


def noisy_expectation(sv0, p, rng):
    acc = 0.0
    for _ in range(N_TRAJECTORIES):
        sv_noisy = NoiseModel.apply_to_sv(sv0.copy(), 2, 'depolarizing', p, rng=rng)
        acc += pauli_expectation(np.asarray(sv_noisy), "ZZ")
    return acc / N_TRAJECTORIES


def unbounded_exponential_zero(scales, values):
    def model(lam, a, b, c):
        return a + b * np.exp(-c * lam)
    try:
        popt, _ = curve_fit(model, scales, values, p0=[0.0, values[0], 0.5], maxfev=20000)
        a, b, _ = popt
        return a + b
    except Exception:
        return float("nan")


def bounded_exponential_zero(scales, values):
    def loss(theta):
        a, zeta, c = theta
        pred = a + (zeta - a) * np.exp(-c * scales)
        return np.sum((values - pred) ** 2)
    res = minimize(loss, x0=[0.0, values[0], 0.5], method='L-BFGS-B',
                    bounds=[(-1.0, 1.0), (-1.0, 1.0), (1e-6, None)])
    return res.x[1]


def run():
    sv0 = ideal_sv()
    seeds = range(30)
    n_unbounded_out_of_range = 0
    n_bounded_out_of_range = 0
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        values = np.array([noisy_expectation(sv0, P_BASE * lam, rng) for lam in SCALES])
        zeta_unbounded = unbounded_exponential_zero(SCALES, values)
        zeta_bounded = bounded_exponential_zero(SCALES, values)
        out_u = not (-1.0 <= zeta_unbounded <= 1.0) if np.isfinite(zeta_unbounded) else True
        out_b = not (-1.0 <= zeta_bounded <= 1.0) if np.isfinite(zeta_bounded) else True
        n_unbounded_out_of_range += out_u
        n_bounded_out_of_range += out_b
        rows.append((seed, values, zeta_unbounded, zeta_bounded, out_u))

    print(f"Ideal E[ZZ] = 1.0 (real Bell state, {len(seeds)} seeds, p_base={P_BASE}, scales={SCALES.tolist()})")
    print(f"unbounded exponential: out-of-[-1,1] in {n_unbounded_out_of_range}/{len(seeds)} seeds")
    print(f"bounded exponential:   out-of-[-1,1] in {n_bounded_out_of_range}/{len(seeds)} seeds")
    print()
    worst = max(rows, key=lambda r: abs(r[2]) if np.isfinite(r[2]) else 1e18)
    seed, values, zu, zb, out_u = worst
    print(f"Worst unbounded case: seed={seed}, measured E(lambda)={np.round(values, 4).tolist()}")
    print(f"  unbounded zero-noise estimate = {zu:.4f} (out of range: {out_u})")
    print(f"  bounded zero-noise estimate   = {zb:.4f}")

    mae_u = np.mean([abs(r[2] - 1.0) for r in rows if np.isfinite(r[2])])
    mae_b = np.mean([abs(r[3] - 1.0) for r in rows if np.isfinite(r[3])])
    print()
    print(f"Mean absolute error vs ideal (1.0): unbounded={mae_u:.4f}, bounded={mae_b:.4f}")


if __name__ == "__main__":
    run()
