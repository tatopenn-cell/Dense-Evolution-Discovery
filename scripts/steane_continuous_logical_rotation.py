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


if __name__ == "__main__":
    for theta in (np.pi / 20, np.pi / 8, 0.15 * np.pi, np.pi / 4):
        run(theta)
        print()
