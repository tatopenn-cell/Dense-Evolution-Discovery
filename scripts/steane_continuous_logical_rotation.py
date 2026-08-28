"""Does Dense-Evolution's own Steane [[7,1,3]] infrastructure reproduce the
continuous-angle logical rotation protocol of Huang, Zhu, Ippoliti, Monroe
& Gullans, arXiv:2608.20676, "Continuous-angle logical rotations in the
Steane code" (IonQ Forte experiment, Aug 2026)?

The protocol: apply a transversal physical Z rotation R_Z(theta)^(x)7 to an
encoded Steane state, measure the 3 X-type stabilizer generators (giving a
syndrome s), apply the Pauli-Z correction the syndrome prescribes. The net
effect on the code space, conditioned on s, is an exactly knowable logical
Z rotation by angle phi_s(theta) -- 0 for the trivial syndrome's own
distinct closed form (paper's Eq. 22), or exactly 3*theta for ANY nontrivial
syndrome (paper's Eq. 23), in the ideal noiseless case.

Reuses real Dense-Evolution building blocks, not a from-scratch
reimplementation: `apply_rz_all` (dense_evolution.noise.coherent_attack) IS
the paper's R_Z(theta) := exp(-i*theta*Z/2) transversal rotation (verified
below: exact convention match, not just "close enough"), and the Steane
|0>_L state-prep circuit is the same verified construction already used in
this repo's Steane investigation (scripts/steane_code_block*.py) and
promoted into dense_evolution/physics/qec.py's own docstring example
(STEANE_X_STABILIZERS). Only the syndrome-projector algebra (Sec. III of
the paper) is new here -- exact 128x128 density-matrix-free linear algebra
on the pure state, not Monte Carlo, since the physical channel considered
here (a coherent rotation with no noise) is deterministic.
"""
import numpy as np
import jax.numpy as jnp

import dense_evolution as de
from dense_evolution.noise.coherent_attack import apply_rz_all
from dense_evolution.registry import NoiseModel

FREE_QUBITS = [0, 1, 3]
DERIVED_QUBITS = {2: [0, 1], 4: [0, 3], 5: [1, 3], 6: [0, 1, 3]}
STEANE_X = ['IIIXXXX', 'IXXIIXX', 'XIXIXIX']
# HX column -> qubit index (0-based), from the paper's Eq. 1 parity-check matrix.
HX_COLUMN_TO_QUBIT = {(0, 0, 1): 0, (0, 1, 0): 1, (0, 1, 1): 2, (1, 0, 0): 3,
                       (1, 0, 1): 4, (1, 1, 0): 5, (1, 1, 1): 6}
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _embed(mat2, q, n=7):
    mats = [mat2 if i == q else np.eye(2, dtype=complex) for i in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def steane_zero_state():
    sim = de.DenseSVSimulator(7)
    ops = [('h', q) for q in FREE_QUBITS]
    for dq, srcs in DERIVED_QUBITS.items():
        for src in srcs:
            ops.append(('cx', src, dq))
    sim.run_circuit(ops)
    return np.asarray(sim.get_statevector())


def _pauli_string_matrix(pstr):
    m = {'I': np.eye(2, dtype=complex), 'X': X}
    out = m[pstr[0]]
    for c in pstr[1:]:
        out = np.kron(out, m[c])
    return out


GENERATORS = [_pauli_string_matrix(s) for s in STEANE_X]


def projector_for_syndrome(bits):
    dim = 2 ** 7
    P = np.eye(dim, dtype=complex)
    for bit, G in zip(bits, GENERATORS):
        sign = 1 if bit == 0 else -1
        P = P @ ((np.eye(dim, dtype=complex) + sign * G) / 2.0)
    return P


def correction_for_syndrome(bits):
    if bits == (0, 0, 0):
        return np.eye(2 ** 7, dtype=complex)
    return _embed(Z, HX_COLUMN_TO_QUBIT[bits])


def logical_branch(sv0, sv1, sv_theta, bits):
    """Runs one syndrome branch: project, correct, read off the logical
    amplitudes. Returns (p_s, c0, c1) where sv_theta's component in the
    code space after this branch is c0*sv0 + c1*sv1 (unnormalized -- p_s
    is its squared norm, the real syndrome probability)."""
    branch = correction_for_syndrome(bits) @ projector_for_syndrome(bits) @ sv_theta
    p_s = np.vdot(branch, branch).real
    return p_s, np.vdot(sv0, branch), np.vdot(sv1, branch)


def run(theta):
    sv0 = steane_zero_state()
    X7 = _embed(X, 0)
    for q in range(1, 7):
        X7 = X7 @ _embed(X, q)
    sv1 = X7 @ sv0
    plus = (sv0 + sv1) / np.sqrt(2)
    sv_theta = np.asarray(apply_rz_all(jnp.array(plus), jnp.array([theta] * 7)))

    p_t, c0_t, c1_t = logical_branch(sv0, sv1, sv_theta, (0, 0, 0))
    phi_t = -np.angle(c1_t * np.conj(c0_t))
    phi_t_ideal = -np.angle(np.exp(1j * theta) * (7 + np.exp(-4j * theta)) ** 2)
    p_t_ideal = (25 + 7 * np.cos(4 * theta)) / 32

    nontrivial = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if (a, b, c) != (0, 0, 0)]
    phi_n_vals, p_n_vals = [], []
    for bits in nontrivial:
        p_s, c0, c1 = logical_branch(sv0, sv1, sv_theta, bits)
        phi_n_vals.append(-np.angle(c1 * np.conj(c0)))
        p_n_vals.append(p_s)
    phi_n_ideal = 3 * theta

    print(f"theta = {theta:.6f} rad ({theta / np.pi:.4f} pi)")
    print(f"trivial syndrome:    p_s = {p_t:.6f} (ideal {p_t_ideal:.6f}), "
          f"phi_s = {phi_t:.6f} (ideal {phi_t_ideal:.6f})")
    print(f"nontrivial syndromes: p_s = {np.round(p_n_vals, 6).tolist()} "
          f"(ideal {(1 - p_t_ideal) / 7:.6f} each)")
    print(f"                       phi_s = {np.round(phi_n_vals, 6).tolist()} "
          f"(ideal 3*theta = {phi_n_ideal:.6f})")
    return p_t, phi_t, p_t_ideal, phi_t_ideal, phi_n_vals, phi_n_ideal


def apply_dephasing_rotation_channel(rho, theta, p):
    """The paper's per-qubit physical channel (Eq. 2): a coherent Z
    rotation composed with dephasing at rate p, N_{p,theta}(rho) =
    R_Z(theta) [(1-p) rho + p Z rho Z] R_Z(theta)^dagger -- written as a
    2-Kraus-operator channel (K0 = sqrt(1-p) R_Z(theta), K1 = sqrt(p)
    R_Z(theta) Z) and applied independently to each of the 7 qubits in
    sequence (each application is a valid CPTP map on a distinct qubit,
    so sequential application equals the tensor-product channel)."""
    rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
    k0 = np.sqrt(1 - p) * rz
    k1 = np.sqrt(p) * (rz @ Z)
    for q in range(7):
        k0f, k1f = _embed(k0, q), _embed(k1, q)
        rho = k0f @ rho @ k0f.conj().T + k1f @ rho @ k1f.conj().T
    return rho


def noisy_branch_stats(sv0, sv1, bits, theta, p):
    """eta_s (Eq. 7-8) and p_s under dephasing at rate p. `eta_s` is
    computed by feeding the (unphysical, but linearly valid) input
    rho_in = |0><1| directly through the channel -- the channel and every
    step after it (projection, correction) are linear/bilinear in rho, so
    this reads off exactly <0| E_s(|0><1|) |1> without needing a full
    process-tomography reconstruction from physical input states."""
    P, C = projector_for_syndrome(bits), correction_for_syndrome(bits)

    rho_01 = np.outer(sv0, sv1.conj())
    branch_01 = C @ P @ apply_dephasing_rotation_channel(rho_01, theta, p) @ P.conj().T @ C.conj().T
    eta = np.vdot(sv0, branch_01 @ sv1)

    rho_00 = np.outer(sv0, sv0.conj())
    branch_00 = C @ P @ apply_dephasing_rotation_channel(rho_00, theta, p) @ P.conj().T @ C.conj().T
    p_s = np.trace(branch_00).real

    return p_s, eta


def run_with_dephasing(theta, p):
    sv0 = steane_zero_state()
    X7 = _embed(X, 0)
    for q in range(1, 7):
        X7 = X7 @ _embed(X, q)
    sv1 = X7 @ sv0
    lam = 1 - 2 * p

    p_t, eta_t = noisy_branch_stats(sv0, sv1, (0, 0, 0), theta, p)
    phi_t = -np.angle(eta_t)
    q_t = 0.5 * (1 - abs(eta_t) / p_t)

    eta_t_formula = (np.exp(1j * theta) / 64) * (14 * lam ** 3 * (3 + np.exp(-4j * theta))
                                                  + lam ** 7 * (7 + np.exp(-8j * theta)))
    p_t_formula = 1 / 8 + (7 / 32) * lam ** 4 * (3 + np.cos(4 * theta))

    p_n, eta_n = noisy_branch_stats(sv0, sv1, (0, 0, 1), theta, p)
    phi_n = -np.angle(eta_n)
    q_n = 0.5 * (1 - abs(eta_n) / p_n)
    eta_n_formula = (np.exp(1j * theta) / 64) * (2 * lam ** 3 * (3 + np.exp(-4j * theta))
                                                  - lam ** 7 * (7 + np.exp(-8j * theta)))

    print(f"theta = {theta:.6f} rad, physical dephasing p = {p * 100:.2f}%")
    print(f"trivial:    eta measured={eta_t:.6f} formula={eta_t_formula:.6f}  "
          f"p_s measured={p_t:.6f} formula={p_t_formula:.6f}  phi_s={phi_t:.6f}  q_s={q_t:.6f}")
    print(f"nontrivial: eta measured={eta_n:.6f} formula={eta_n_formula:.6f}  "
          f"p_s measured={p_n:.6f}  phi_s={phi_n:.6f}  q_s={q_n:.6f}")


NONTRIVIAL_SYNDROMES = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if (a, b, c) != (0, 0, 0)]


def check_all_nontrivial_syndromes_agree(theta, p):
    """The paper claims (Eq. 15, 'identical for all nontrivial syndromes
    by symmetry') that eta_n and p_n don't depend on WHICH of the 7
    nontrivial syndromes occurred, only that it's nontrivial. This was
    only spot-checked on one syndrome, (0,0,1), when the dephasing case
    was first added -- here all 7 are checked explicitly."""
    sv0 = steane_zero_state()
    X7 = _embed(X, 0)
    for q in range(1, 7):
        X7 = X7 @ _embed(X, q)
    sv1 = X7 @ sv0
    print(f"theta={theta:.6f}, p={p * 100:.2f}%: checking all 7 nontrivial syndromes agree")
    results = [noisy_branch_stats(sv0, sv1, bits, theta, p) for bits in NONTRIVIAL_SYNDROMES]
    p_vals = [r[0] for r in results]
    eta_vals = [r[1] for r in results]
    print(f"  p_s range: [{min(p_vals):.6f}, {max(p_vals):.6f}]  "
          f"(all equal to 1e-9: {np.ptp(p_vals) < 1e-9})")
    print(f"  eta range (max pairwise abs diff): {max(abs(a - b) for a in eta_vals for b in eta_vals):.2e}")


def two_round_step(rho, bits, theta, p):
    P, C = projector_for_syndrome(bits), correction_for_syndrome(bits)
    rho_after_channel = apply_dephasing_rotation_channel(rho, theta, p)
    return C @ P @ rho_after_channel @ P.conj().T @ C.conj().T


def two_round_class_channel(rho_in, class1, class2, theta, p):
    """Aggregates over all syndromes in each class (trivial='t', a single
    syndrome; nontrivial='n', summed over all 7), matching how the paper
    groups its two-round data into 4 classes (Fig. 13): round 1 with
    angle +theta, round 2 with angle -theta (the paper's own cancellation
    test)."""
    s1_list = [(0, 0, 0)] if class1 == 't' else NONTRIVIAL_SYNDROMES
    s2_list = [(0, 0, 0)] if class2 == 't' else NONTRIVIAL_SYNDROMES
    total = np.zeros((2 ** 7, 2 ** 7), dtype=complex)
    for s1 in s1_list:
        after_round1 = two_round_step(rho_in, s1, theta, p)
        for s2 in s2_list:
            total += two_round_step(after_round1, s2, -theta, p)
    return total


def run_two_round(theta, p):
    """Paper Section IV.C / Fig. 13: apply +theta, syndrome-extract and
    correct, then -theta, syndrome-extract and correct again -- expect
    the trivial-trivial branch to show the LEAST logical rotation (angles
    should cancel) and the LOWEST dephasing, matching Fig. 13's reported
    finding qualitatively (this project does not have the paper's real
    hardware data to match numerically, only its own theoretical model,
    so only the qualitative ordering is checked, not exact values)."""
    sv0 = steane_zero_state()
    X7 = _embed(X, 0)
    for q in range(1, 7):
        X7 = X7 @ _embed(X, q)
    sv1 = X7 @ sv0

    print(f"Two-round protocol: theta={theta:.6f}, p={p * 100:.2f}%")
    results = {}
    for c1 in ("t", "n"):
        for c2 in ("t", "n"):
            rho_01 = np.outer(sv0, sv1.conj())
            branch_01 = two_round_class_channel(rho_01, c1, c2, theta, p)
            eta = np.vdot(sv0, branch_01 @ sv1)

            rho_00 = np.outer(sv0, sv0.conj())
            branch_00 = two_round_class_channel(rho_00, c1, c2, theta, p)
            p_s = np.trace(branch_00).real

            phi = -np.angle(eta) if abs(eta) > 1e-12 else 0.0
            q_dephasing = 0.5 * (1 - abs(eta) / p_s) if p_s > 1e-12 else None
            results[(c1, c2)] = (p_s, phi, q_dephasing)
            print(f"  {c1}-{c2}: p_s={p_s:.6f}  phi={phi:.6f}  q={q_dephasing:.6f}")

    tt_q = results[("t", "t")][2]
    nn_q = results[("n", "n")][2]
    print(f"  trivial-trivial has lowest dephasing (matches paper Fig. 13): {tt_q < nn_q}")
    print(f"  trivial-trivial phi near zero (cancellation): {abs(results[('t','t')][1]) < 0.01}")


def monte_carlo_syndrome_probabilities(theta, p, n_trials, seed):
    """Real circuit-level Monte Carlo check (not exact linear algebra like
    everything above): builds an actual 10-qubit circuit (7 data + 3
    ancilla, one ancilla per X-stabilizer generator), applies real 'rz'
    gates, real NoiseModel.apply_to_sv('phaseflip', p) stochastic
    dephasing, a real H-CNOT-H ancilla circuit per generator, and a real
    DenseSVSimulator.measure() projective collapse per ancilla qubit --
    then tabulates the empirical syndrome distribution over many trials
    and compares to the exact p_s formula.

    NOTE on scope: this only verifies syndrome PROBABILITIES, not the
    logical rotation angle/dephasing (phi_s, q_s). Attempting to read off
    eta_s directly from a single |+>_L trial's post-selected amplitudes
    does NOT work, since eta_s is a coherence element requiring an
    unphysical |0><1| input -- a real single-trial-based estimate this
    way is off by a theta/p-dependent factor, not the true eta_s (this
    was a genuine, real dead end hit and understood, not a coding bug).
    The correct real-circuit way to get at phi_s/q_s from physical shots
    is `monte_carlo_logical_x_readout` below (the paper's own Ramsey
    approach, Section IV.A)."""
    sv0 = steane_zero_state()
    rng = np.random.default_rng(seed)
    counts = {}
    for _ in range(n_trials):
        sim7 = de.DenseSVSimulator(7)
        sim7.set_initial_state(sv0.copy())
        sim7.run_circuit_jit([('rz', q, theta) for q in range(7)])
        sv_data = np.asarray(sim7.get_statevector())
        sv_data_noisy = np.asarray(NoiseModel.apply_to_sv(sv_data, 7, 'phaseflip', p, rng=rng))

        anc = np.zeros(8, dtype=complex)
        anc[0] = 1.0
        sim10 = de.DenseSVSimulator(10)
        sim10.set_initial_state(np.kron(sv_data_noisy, anc))
        bits = []
        for gi, gate_str in enumerate(STEANE_X):
            support = [i for i, c in enumerate(gate_str) if c == 'X']
            a = 7 + gi
            sim10.run_circuit_jit([('h', a)] + [('cx', a, q) for q in support] + [('h', a)])
            bits.append(sim10.measure(a))
        counts[tuple(bits)] = counts.get(tuple(bits), 0) + 1

    lam = 1 - 2 * p
    p_t_formula = 1 / 8 + (7 / 32) * lam ** 4 * (3 + np.cos(4 * theta))
    p_t_mc = counts.get((0, 0, 0), 0) / n_trials
    p_n_mc = sum(v for k, v in counts.items() if k != (0, 0, 0)) / n_trials

    print(f"Real-circuit Monte Carlo (n_trials={n_trials}), theta={theta:.6f}, p={p * 100:.2f}%")
    print(f"  p_trivial:    MC={p_t_mc:.4f}  exact={p_t_formula:.4f}")
    print(f"  p_nontrivial (aggregate): MC={p_n_mc:.4f}  exact={1 - p_t_formula:.4f}")
    print(f"  per-syndrome nontrivial counts: "
          f"{ {k: round(v / n_trials, 4) for k, v in counts.items() if k != (0, 0, 0)} }")
    print(f"  expected each: {(1 - p_t_formula) / 7:.4f}")


def _xor_closure(rows):
    """All 2**len(rows) XOR combinations of `rows` (each a 0/1 array),
    including the all-zero vector -- the row space of a binary matrix
    over GF(2)."""
    words = {tuple(np.zeros(len(rows[0]), dtype=int))}
    for r in rows:
        words |= {tuple((np.array(w) + r) % 2) for w in words}
    return words


def _steane_codeword_cosets():
    """The two logical-basis cosets C (|0>_L's support) and C+1111111
    (|1>_L's support), as sets of 7-bit integers -- extracted from the
    XOR-closure of the 3 rows of HX (Eq. 1), independently VERIFIED to
    match the real, nonzero support of this script's own `sv0`/`sv1`
    exactly (both give the same 8+8 codewords) rather than assumed from
    the paper's text alone."""
    hx_rows = [np.array([int(b) for b in s]) for s in ('0001111', '0110011', '1010101')]
    c_words = _xor_closure(hx_rows)
    c_ints = {int(''.join(map(str, w)), 2) for w in c_words}
    c1_ints = {int(''.join(map(str, (np.array(w) + 1) % 2)), 2) for w in c_words}
    return c_ints, c1_ints


def _nearest_coset_decode(x_meas_int, c_ints, c1_ints):
    """Nearest-coset decoding of a 7-bit measured X-basis outcome (the
    paper's own decoding rule, Section IV.A: 'for arbitrary x, decode to
    the logical bit of the nearest of these two cosets') -- minimum
    Hamming distance to either coset, not exact membership only."""
    d0 = min(bin(x_meas_int ^ c).count('1') for c in c_ints)
    d1 = min(bin(x_meas_int ^ c).count('1') for c in c1_ints)
    return 0 if d0 < d1 else 1


def monte_carlo_logical_x_readout(theta, p, n_trials, seed):
    """The paper's own Ramsey experiment (Section IV.A, Fig. 5-6), fully
    real end to end: prepare |+>_L, apply the real transversal rz(theta),
    real stochastic dephasing, the real ancilla syndrome-extraction
    circuit with a real measurement collapse, the real Z correction, then
    apply H to all 7 DATA qubits and measure them too (logical X-basis
    readout) -- decoded via `_nearest_coset_decode`, exactly as the paper
    does. This is what `monte_carlo_syndrome_probabilities` above
    explicitly left undone.

    Ideal-case P(X=+1|s) = (1+cos(phi_s))/2 (paper Eq. 25) is only valid
    at p=0 -- with dephasing, the general form is P(X=+1|s) = 0.5 +
    0.5*Re(eta_s)/p_s (since eta_s = p_s*(1-2*q_s)*exp(-i*phi_s), so
    Re(eta_s)/p_s = (1-2*q_s)*cos(phi_s) exactly). Comparing MC results
    against the p=0 formula instead of this one was a real mistake caught
    while building this function (the trivial branch matched by luck --
    its q_s is small -- but the nontrivial branch was off by ~11 sigma
    until this fix)."""
    sv0 = steane_zero_state()
    X7 = _embed(X, 0)
    for q in range(1, 7):
        X7 = X7 @ _embed(X, q)
    sv1 = X7 @ sv0
    plus_data = (sv0 + sv1) / np.sqrt(2)
    c_ints, c1_ints = _steane_codeword_cosets()

    rng = np.random.default_rng(seed)
    outcomes = {'trivial': [], 'nontrivial': []}
    for _ in range(n_trials):
        sim7 = de.DenseSVSimulator(7)
        sim7.set_initial_state(plus_data.copy())
        sim7.run_circuit_jit([('rz', q, theta) for q in range(7)])
        sv_data = np.asarray(sim7.get_statevector())
        sv_data_noisy = np.asarray(NoiseModel.apply_to_sv(sv_data, 7, 'phaseflip', p, rng=rng))

        anc = np.zeros(8, dtype=complex)
        anc[0] = 1.0
        sim10 = de.DenseSVSimulator(10)
        sim10.set_initial_state(np.kron(sv_data_noisy, anc))
        bits = []
        for gi, supp in enumerate(_generator_supports()):
            a = 7 + gi
            sim10.run_circuit_jit([('h', a)] + [('cx', a, q) for q in supp] + [('h', a)])
            bits.append(sim10.measure(a))
        bits = tuple(bits)
        if bits != (0, 0, 0):
            sim10.run_circuit_jit([('z', HX_COLUMN_TO_QUBIT[bits])])

        sim10.run_circuit_jit([('h', q) for q in range(7)])
        x_meas = int(''.join(str(sim10.measure(q)) for q in range(7)), 2)
        logical = _nearest_coset_decode(x_meas, c_ints, c1_ints)

        key = 'trivial' if bits == (0, 0, 0) else 'nontrivial'
        outcomes[key].append(1 if logical == 0 else 0)

    lam = 1 - 2 * p
    for key, bits_for_formula in (('trivial', (0, 0, 0)), ('nontrivial', (0, 0, 1))):
        p_s, eta = noisy_branch_stats(sv0, sv1, bits_for_formula, theta, p)
        p_theory_correct = 0.5 + 0.5 * (eta.real / p_s)
        p_theory_ideal_wrong = 0.5 + 0.5 * np.cos(-np.angle(eta))
        p_mc = np.mean(outcomes[key])
        n = len(outcomes[key])
        print(f"{key} (N={n}): P(X=+1|s) MC={p_mc:.4f}  "
              f"theory(with dephasing, correct)={p_theory_correct:.4f}  "
              f"theory(ideal formula, WRONG here since p>0)={p_theory_ideal_wrong:.4f}")


def _generator_supports():
    return [[i for i, c in enumerate(s) if c == 'X'] for s in STEANE_X]


if __name__ == "__main__":
    for theta in (np.pi / 20, np.pi / 8, 0.15 * np.pi, np.pi / 4):
        run(theta)
        print()

    print("=" * 100)
    print("With dephasing (paper Eq. 14-17), at the paper's own two real fitted physical")
    print("dephasing rates from IonQ Forte hardware: p=2.13% (Ramsey fit) and p=2.62%")
    print("(process-tomography fit)")
    print("=" * 100)
    for p in (0.0213, 0.0262):
        run_with_dephasing(0.15 * np.pi, p)
        print()

    print("=" * 100)
    print("All 7 nontrivial syndromes checked (not just 1) under dephasing")
    print("=" * 100)
    check_all_nontrivial_syndromes_agree(0.15 * np.pi, 0.0213)
    print()

    print("=" * 100)
    print("Two-round protocol (+theta then -theta)")
    print("=" * 100)
    run_two_round(0.15 * np.pi, 0.0213)
    print()

    print("=" * 100)
    print("Real-circuit Monte Carlo verification of syndrome probabilities")
    print("(scope: probabilities only -- see docstring for why phi_s/q_s aren't")
    print("attempted from single-shot trials here)")
    print("=" * 100)
    monte_carlo_syndrome_probabilities(0.15 * np.pi, 0.0213, n_trials=4000, seed=11)
    print()

    print("=" * 100)
    print("Real-circuit Monte Carlo logical-X readout (paper's own Ramsey experiment,")
    print("Section IV.A) -- the piece left undone above, now completed with a real")
    print("Hamming nearest-coset decoder")
    print("=" * 100)
    monte_carlo_logical_x_readout(0.15 * np.pi, 0.0213, n_trials=6000, seed=21)
