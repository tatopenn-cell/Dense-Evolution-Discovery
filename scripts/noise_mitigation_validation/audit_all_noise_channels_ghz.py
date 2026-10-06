"""
Scratch verification (not the final repo test) -- extend the depolarizing
audit's three-way methodology (apply_to_sv vs from-scratch one-decision-
per-qubit-per-shot sampler vs exact density-matrix Kraus sum) to ALL
non-ideal channels in NoiseModel.apply_to_sv, on a small GHZ-4 state, to
decide which channels besides depolarizing need the same fix.

Run against the CURRENT (pre-fix) registry.py in the worktree under test
by editing DE_ROOT below.
"""
import sys
import numpy as np

DE_ROOT = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Admin\Desktop\Dense-Evolution-main\Dense-Evolution-main'
sys.path.insert(0, DE_ROOT)
from dense_evolution import DenseSVSimulator
from dense_evolution.observables import pauli_expectation
from dense_evolution.registry import NoiseModel

N = 4

_PAULI_2X2 = {
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _embed(pauli_2x2, qubit, n):
    mats = [pauli_2x2 if i == qubit else np.eye(2, dtype=complex) for i in range(n)]
    full = mats[0]
    for m in mats[1:]:
        full = np.kron(full, m)
    return full


def _embed_string(pauli_str):
    n = len(pauli_str)
    mats = [_PAULI_2X2.get(c, np.eye(2, dtype=complex)) for c in pauli_str]
    full = mats[0]
    for m in mats[1:]:
        full = np.kron(full, m)
    return full


def _apply_pauli_msb(sv, n, qubit, pauli):
    dim = len(sv)
    phys = n - 1 - qubit
    step = 1 << phys
    idx = np.arange(dim)
    idx0 = idx[(idx & step) == 0]
    idx1 = idx0 | step
    v0, v1 = sv[idx0].copy(), sv[idx1].copy()
    out = sv.copy()
    if pauli == 'X':
        out[idx0], out[idx1] = v1, v0
    elif pauli == 'Y':
        out[idx0], out[idx1] = -1j * v1, 1j * v0
    elif pauli == 'Z':
        out[idx1] = -v1
    return out


def scratch_depolarizing(sv, n, p, rng):
    out = sv.copy()
    for q in range(n):
        if rng.random() < p:
            pauli = ('X', 'Y', 'Z')[int(rng.random() * 3.0)]
            out = _apply_pauli_msb(out, n, q, pauli)
    return out / np.linalg.norm(out)


def scratch_bitflip(sv, n, p, rng):
    out = sv.copy()
    for q in range(n):
        if rng.random() < p:
            out = _apply_pauli_msb(out, n, q, 'X')
    return out / np.linalg.norm(out)


def scratch_phaseflip(sv, n, p, rng):
    out = sv.copy()
    for q in range(n):
        if rng.random() < p:
            out = _apply_pauli_msb(out, n, q, 'Z')
    return out / np.linalg.norm(out)


def scratch_amplitude_damping(sv, n, gamma, rng):
    out = sv.copy()
    dim = len(out)
    for q in range(n):
        phys = n - 1 - q
        step = 1 << phys
        idx = np.arange(dim)
        idx0 = idx[(idx & step) == 0]
        idx1 = idx0 | step
        v0, v1 = out[idx0].copy(), out[idx1].copy()
        p1 = gamma * np.sum(np.abs(v1) ** 2)
        new = out.copy()
        if rng.random() < p1:
            # K1 jump, global renormalization by sqrt(p1)
            new[idx0] = v1 * np.sqrt(gamma) / np.sqrt(p1)
            new[idx1] = 0.0
        else:
            new[idx0] = v0 / np.sqrt(1.0 - p1)
            new[idx1] = v1 * np.sqrt(1.0 - gamma) / np.sqrt(1.0 - p1)
        out = new
    return out / np.linalg.norm(out)


def exact_dm(sv, n, model, p):
    rho = np.outer(sv, sv.conj())
    for q in range(n):
        if model == 'depolarizing':
            X, Y, Z = (_embed(_PAULI_2X2[k], q, n) for k in 'XYZ')
            rho = (1 - p) * rho + (p / 3.0) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
        elif model == 'bitflip':
            X = _embed(_PAULI_2X2['X'], q, n)
            rho = (1 - p) * rho + p * (X @ rho @ X)
        elif model == 'phaseflip':
            Z = _embed(_PAULI_2X2['Z'], q, n)
            rho = (1 - p) * rho + p * (Z @ rho @ Z)
        elif model == 'amplitude_damping':
            gamma = p
            K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
            K0f, K1f = _embed(K0, q, n), _embed(K1, q, n)
            rho = K0f @ rho @ K0f.conj().T + K1f @ rho @ K1f.conj().T
        elif model == 'combined':
            p_dep, p_damp = p * 0.5, p * 0.333333
            X, Y, Z = (_embed(_PAULI_2X2[k], q, n) for k in 'XYZ')
            rho = (1 - p_dep) * rho + (p_dep / 3.0) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
            gamma = p_damp
            K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
            K0f, K1f = _embed(K0, q, n), _embed(K1, q, n)
            rho = K0f @ rho @ K0f.conj().T + K1f @ rho @ K1f.conj().T
    return rho


def ghz_state(n):
    sim = DenseSVSimulator(n)
    ops = [('h', 0)] + [('cx', 0, q) for q in range(1, n)]
    sim.run_circuit(ops)
    return sim.get_statevector()


FREE_QUBITS = [0, 1, 3]
DERIVED_QUBITS = {2: [0, 1], 4: [0, 3], 5: [1, 3], 6: [0, 1, 3]}


def steane_zero_state():
    sim = DenseSVSimulator(7)
    ops = [('h', q) for q in FREE_QUBITS]
    for derived_q, sources in DERIVED_QUBITS.items():
        for src in sources:
            ops.append(('cx', src, derived_q))
    sim.run_circuit(ops)
    return sim.get_statevector()


SCRATCH = {
    'depolarizing': scratch_depolarizing,
    'bitflip': scratch_bitflip,
    'phaseflip': scratch_phaseflip,
    'amplitude_damping': scratch_amplitude_damping,
}


def run(model, observable, p_values, n_trials, seed, sv0, n=N):
    ideal = pauli_expectation(sv0, observable).real
    for p in p_values:
        rng_a = np.random.default_rng(seed)
        vals_a = np.empty(n_trials)
        for i in range(n_trials):
            svn = NoiseModel.apply_to_sv(sv0.copy(), n, model, float(p), rng=rng_a)
            vals_a[i] = pauli_expectation(svn, observable).real

        vals_b = None
        if model in SCRATCH:
            rng_b = np.random.default_rng(seed + 1)
            vals_b = np.empty(n_trials)
            scratch_fn = SCRATCH[model]
            for i in range(n_trials):
                svn = scratch_fn(sv0.copy(), n, float(p), rng_b)
                vals_b[i] = pauli_expectation(svn, observable).real

        rho = exact_dm(sv0, n, model, float(p))
        G = _embed_string(observable)
        exact_val = np.trace(rho @ G).real

        mean_a, sem_a = vals_a.mean(), vals_a.std(ddof=1) / np.sqrt(n_trials)
        line = f"  [{model}] G={observable} p={p:.3f}  apply_to_sv={mean_a:+.4f}+-{sem_a:.4f}  exact_dm={exact_val:+.4f}  ideal={ideal:+.4f}"
        sigma_dm = abs(mean_a - exact_val) / (sem_a + 1e-15)
        if vals_b is not None:
            mean_b, sem_b = vals_b.mean(), vals_b.std(ddof=1) / np.sqrt(n_trials)
            sigma = abs(mean_a - mean_b) / np.sqrt(sem_a ** 2 + sem_b ** 2)
            line += f"  scratch={mean_b:+.4f}+-{sem_b:.4f}  sigma_vs_scratch={sigma:.1f}"
        line += f"  sigma_vs_exact_dm={sigma_dm:.1f}"
        print(line)


def main():
    sv0 = ghz_state(N)
    p_values = [0.05, 0.15]
    n_trials = 20000

    print("=" * 100)
    print("bitflip on XXXX (should be EXACTLY invariant under bit-flip channel per qubit)")
    print("=" * 100)
    run('bitflip', 'XXXX', p_values, n_trials, seed=100, sv0=sv0)

    print("=" * 100)
    print("phaseflip on XXXX (closed form (1-2p)^4)")
    print("=" * 100)
    run('phaseflip', 'XXXX', p_values, n_trials, seed=200, sv0=sv0)

    print("=" * 100)
    print("depolarizing on XXXX (closed form (1-4p/3)^4) -- known-buggy control")
    print("=" * 100)
    run('depolarizing', 'XXXX', p_values, n_trials, seed=300, sv0=sv0)

    print("=" * 100)
    print("amplitude_damping on XXXX")
    print("=" * 100)
    run('amplitude_damping', 'XXXX', p_values, n_trials, seed=400, sv0=sv0)

    print("=" * 100)
    print("amplitude_damping on ZZZZ (diagonal, control)")
    print("=" * 100)
    run('amplitude_damping', 'ZZZZ', p_values, n_trials, seed=500, sv0=sv0)

    print("=" * 100)
    print("combined on XXXX")
    print("=" * 100)
    run('combined', 'XXXX', p_values, n_trials, seed=600, sv0=sv0)

    print("\n" + "=" * 100)
    print("Steane |0>_L, IIIXXXX (weight-4, genuine multi-branch coherence, unlike GHZ)")
    print("=" * 100)
    sv_steane = steane_zero_state()
    for model, seed in [('phaseflip', 700), ('bitflip', 800), ('depolarizing', 900),
                         ('amplitude_damping', 1000), ('combined', 1100)]:
        run(model, 'IIIXXXX', p_values, n_trials, seed=seed, sv0=sv_steane, n=7)


if __name__ == '__main__':
    main()
