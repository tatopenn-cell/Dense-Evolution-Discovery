"""Does the stabilizer Renyi entropy (SRE), tracked through every stage of
the wormhole-inspired teleportation protocol (WITP) in a dense SYK4 model,
reproduce the qualitative regime-dependent structure reported by Joshi &
Mishra, arXiv:2606.19180, "Quantum magic is necessary but not sufficient
for wormhole-inspired teleportation"?

The paper's own quantity (Eq. 14, Leone-Oliviero-Hamma stabilizer Renyi
entropy) is NOT the same as anything already in dense_evolution.mitigation:
`magic_entropy`/`magic_entropy_from_shadows` (Bu-Gu-Jaffe 3-fold
self-convolution, single-qubit only) and `sandwiched_renyi_divergence`
(Muller-Lennert et al., a divergence between two density matrices) are both
genuinely different mathematical objects. The multi-qubit, single-state SRE
used here is implemented fresh below (Walsh-Hadamard-accelerated, Eq. 16-18),
verified against known values: stabilizer states give exactly 0, and a
single T state gives 0.415037 -- derived by hand from Eq. 14 directly
(-log2[(1 + 2*(1/sqrt(2))**4)/2] = -log2(0.75)), matching the function's
output to 6 decimals. (An initially recalled "known value" of log2(9/7) for
this same case was simply a wrong memory and is not used anywhere here.)

The dense SYK4 Hamiltonian, TFD state, and this paper's specific WITP
protocol (Eqs. 1-11) are NOT the same construction as the existing
Dense-Evolution-Discovery wormhole script (scripts/wormhole_syk_teleportation.py,
dashboard_core.wormhole) -- that implements a DIFFERENT paper's protocol
(arXiv:2604.10090: sparse binary SYK, mutual-information readout, Trotterized
coupling). What IS reused here, verified working exactly as documented: the
Jordan-Wigner Majorana-to-Pauli mapping (`dense_evolution.fermions.
majorana_pauli_terms`), Pauli-term multiplication (`dashboard_core.wormhole.
_multiply_pauli_dicts`, `_embed`), and Hamiltonian assembly
(`dense_evolution.pauli_hamiltonian_to_matrix`) -- the same three building
blocks the existing sparse-binary script already uses, just assembled into
this paper's dense, Gaussian-coupled, finite-beta-TFD construction instead.

KLEIN-FACTOR FIX (resolved 2026-08-29, verified numerically below, not just
derived): `_embed`'s independent per-side Jordan-Wigner mapping makes L-side
and R-side Majoranas act on disjoint qubits and therefore COMMUTE (this is
also the convention `dashboard_core.wormhole`'s own V = sum(chi_L^j chi_R^j)
coupling already relies on, per its own docstring) -- but this paper's Eq. 1
Dirac fermion c_i = (psi_L^i + i*psi_R^i)/2 needs psi_L^i and psi_R^i to
ANTIcommute for n_i = c_i^dagger c_i to have proper 0/1 eigenvalues, not the
degenerate n_i = 0.5*I this module originally found. This is the standard
"Klein factor" problem from bosonization / fermionic-entanglement literature
(e.g. Fidkowski-Kitaev): combining two independently-Jordan-Wigner-mapped
Majorana registers into one joint fermionic algebra needs an explicit
parity-string correction. The fix, implemented in `_side_parity` /
`_joint_right_majorana`: dress every right-side Majorana used in a
cross-register product with P_L, the LEFT register's total fermion-parity
operator (the ordered product of ALL its Majoranas) -- P_L anticommutes with
each individual psi_L^i (an even number of mutually anticommuting,
squares-to-I operators multiplied together anticommutes with each factor:
moving one past the other n_majorana-1 picks up (-1)^(n_majorana-1) = -1),
which flips the L-R commutator to an anticommutator while leaving the
right-side operators' own mutual algebra untouched (P_L^2 = I factors out
trivially). Verified: n_i now has exact eigenvalues {0.0, 1.0} (was
degenerate 0.5 before), and e^{igV} is unitary to machine precision
(||U^dagger U - I|| ~ 1e-14, was ~55 before) under EITHER sign convention
for W_i = +-i*psi_L^i*psi_R^i_joint -- both are equally valid once W_i is
genuinely Hermitian, so `_sanity_check_coupling_unitary`'s empirical
either-sign check (kept below) no longer needs to pick a winner.
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\Desktop\Dense-Evolution-main\Dense-Evolution-main\tools\dashboard")
from core.wormhole import _multiply_pauli_dicts, _embed  # noqa: E402
from dense_evolution.fermions import majorana_pauli_terms  # noqa: E402
from dense_evolution import pauli_hamiltonian_to_matrix  # noqa: E402

J = 5.0


def stabilizer_renyi_entropy(psi):
    """Eq. 14/18: M2 = -log2[(1/d) sum_a sum_b |WHT[c_a](b)|^4], via an
    explicit d x d Hadamard sign matrix (fine for d up to a few thousand;
    the paper's own O(4^n * n) butterfly algorithm is asymptotically
    better but unnecessary at the system sizes used here)."""
    d = len(psi)
    idx = np.arange(d)
    popcount = np.array([bin(i).count("1") for i in range(d)])
    signmat = np.array([(-1.0) ** popcount[idx & b] for b in range(d)])
    total = 0.0
    for a in range(d):
        c_a = np.conj(psi) * psi[idx ^ a]
        wht = signmat @ c_a
        total += np.sum(np.abs(wht) ** 4)
    return -np.log2(total / d)


def build_dense_syk4_terms(n_majorana, seed):
    """Eq. 2-3: all C(n_majorana,4) four-Majorana products, coefficients
    drawn from the paper's own Gaussian variance 6*J^2/n_majorana^3
    (Eq. 3), sharing ONE coupling tensor between L and R (Sec. 2: 'The
    left and right Hamiltonians HL and HR share the same coupling
    tensor')."""
    n_qubits = n_majorana // 2
    quads = list(itertools.combinations(range(1, n_majorana + 1), 4))
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(6.0 * J ** 2 / n_majorana ** 3)
    terms = []
    for quad in quads:
        coupling = rng.normal(0.0, sigma)
        dicts = [majorana_pauli_terms(m, n_qubits)[1] for m in quad]
        phase, merged = _multiply_pauli_dicts(dicts)
        terms.append((coupling * phase, merged))
    return n_qubits, terms


def _logical_xyz(offset, n_full):
    """Eq. 7: XL=-i*psi1*psi3, YL=-i*psi3*psi2, ZL=-i*psi2*psi1 (or the
    R-side analogues at qubit offset `n_side`), returned as full
    2**n_full x 2**n_full matrices."""
    p1, p2, p3 = (_embed(k, n_full, offset) for k in (1, 2, 3))
    ops = {}
    for name, pair, sign in (("X", (p1, p3), -1j), ("Y", (p3, p2), -1j), ("Z", (p2, p1), -1j)):
        phase, merged = _multiply_pauli_dicts(list(pair))
        ops[name] = (sign * phase) * pauli_hamiltonian_to_matrix([(1.0, merged)], n_full)
    return ops["X"], ops["Y"], ops["Z"]


def _embed_1q(mat2, qubit, n_full):
    mats = [mat2 if i == qubit else np.eye(2, dtype=complex) for i in range(n_full)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _side_parity(n_majorana, n_full, offset):
    """Total fermion-parity (Klein) operator for one side's n_majorana
    Majorana modes: the ORDERED product of all of them, embedded at qubit
    offset `offset` -- returned as a (phase, pauli_dict) term, computed by
    exact symbolic Pauli algebra (_multiply_pauli_dicts tracks the i^k
    phase from same-qubit XY=iZ etc. exactly; no numerical fudge needed).
    An even number (n_majorana) of mutually anticommuting, squares-to-I
    Majorana operators multiplied together anticommutes with each
    individual factor -- moving one past the other n_majorana-1 picks up
    (-1)^(n_majorana-1) = -1. This is what needs to dress every operator
    on the OTHER side in a cross-register product to restore correct
    anticommutation -- see the Klein-factor fix in the module docstring."""
    dicts = [_embed(m, n_full, offset) for m in range(1, n_majorana + 1)]
    return _multiply_pauli_dicts(dicts)


def _joint_right_majorana(i, n_majorana, n_side, n_full):
    """psi_R^i dressed with the LEFT register's total parity (Klein
    factor): P_L * psi_R^i, as a (phase, pauli_dict) term. Unlike the bare
    psi_R^i -- which COMMUTES with every psi_L^i by construction, since
    _embed's independent per-side Jordan-Wigner mapping puts L and R on
    disjoint qubits -- this dressed operator ANTIcommutes with every
    psi_L^i (P_L anticommutes with each individual left Majorana; see
    _side_parity), while still anticommuting with the other psi_R^j
    (j != i) and squaring to I, since P_L^2 = I acts as a trivial factor
    in {psi_R^i_joint, psi_R^j_joint} = P_L^2 * {psi_R^i, psi_R^j}."""
    phase_pl, dict_pl = _side_parity(n_majorana, n_full, offset=0)
    pr = _embed(i, n_full, n_side)
    phase_mul, merged = _multiply_pauli_dicts([dict_pl, pr])
    return phase_pl * phase_mul, merged


def _dirac_number_operator(i, n_majorana, n_side, n_full, use_klein_fix):
    """Eq. 1's Dirac fermion c_i = (psi_L^i + i*psi_R^i)/2 and its number
    operator n_i = c_i^dagger c_i. With use_klein_fix=False, psi_R^i is
    the bare, independently-JW-mapped operator (the documented bug: n_i
    degenerates to exactly 0.5*I since psi_L^i and psi_R^i commute
    instead of anticommuting). With True, psi_R^i is replaced by its
    Klein-dressed joint version (_joint_right_majorana), which should
    restore n_i's proper 0/1 eigenvalues."""
    pl = _embed(i, n_full, 0)
    psi_l = pauli_hamiltonian_to_matrix([(1.0, pl)], n_full)
    if use_klein_fix:
        phase, merged = _joint_right_majorana(i, n_majorana, n_side, n_full)
    else:
        phase, merged = 1.0, _embed(i, n_full, n_side)
    psi_r = phase * pauli_hamiltonian_to_matrix([(1.0, merged)], n_full)
    c = 0.5 * (psi_l + 1j * psi_r)
    return c.conj().T @ c


def _coupling_unitary(n_majorana, n_side, n_full, g, anti_hermitian_w, use_klein_fix):
    """e^{igV} = prod_i [1 + (e^{ig}-1) n_i], n_i = (1+W_i)/2 (paper's own
    factorization). Term_i = 0.5*(1+phi)*I + 0.5*(phi-1)*W_i (derived by
    hand from n_i's definition -- NOT copied from the paper's own
    Appendix A, which only expands the phase perturbatively and never
    states this exact closed form). `anti_hermitian_w` selects which sign
    convention for W_i = +-i*psi_L*psi_R to test; `use_klein_fix` selects
    whether psi_R is the bare (buggy, commuting) or Klein-dressed
    (anticommuting) operator -- see _joint_right_majorana."""
    phi = np.exp(1j * g)
    dim = 2 ** n_full
    Uc = np.eye(dim, dtype=complex)
    sign = -1.0 if anti_hermitian_w else 1.0
    for i in range(3, n_majorana + 1):
        pL = _embed(i, n_full, 0)
        if use_klein_fix:
            phase_r, dict_r = _joint_right_majorana(i, n_majorana, n_side, n_full)
        else:
            phase_r, dict_r = 1.0, _embed(i, n_full, n_side)
        phase, merged = _multiply_pauli_dicts([pL, dict_r])
        W = (sign * 1j * phase * phase_r) * pauli_hamiltonian_to_matrix([(1.0, merged)], n_full)
        term = 0.5 * (1 + phi) * np.eye(dim, dtype=complex) + 0.5 * (phi - 1) * W
        Uc = term @ Uc
    return Uc


def _sanity_check_coupling_unitary(n_majorana, n_side, n_full, g=1.83 * np.pi, use_klein_fix=False):
    """Empirically resolves the Hermiticity-convention question in the
    module docstring: builds e^{igV} both ways and reports which one is
    actually unitary (||U^dagger U - I|| ~ 0)."""
    results = {}
    for label, flag in (("W = +i*psiL*psiR", False), ("W = -i*psiL*psiR", True)):
        Uc = _coupling_unitary(n_majorana, n_side, n_full, g, anti_hermitian_w=flag, use_klein_fix=use_klein_fix)
        dim = Uc.shape[0]
        err = np.linalg.norm(Uc.conj().T @ Uc - np.eye(dim))
        results[label] = err
        print(f"  {label}: ||U^dagger U - I|| = {err:.3e}")
    return results


if __name__ == "__main__":
    print("Step 0: verify the SRE function")
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = 1.0
    print(f"  |000> (stabilizer): M2 = {stabilizer_renyi_entropy(psi0):.6f} (expect 0)")
    theta = np.pi / 8
    t1 = np.array([np.cos(theta), np.sin(theta)], dtype=complex)
    print(f"  T state (1 qubit): M2 = {stabilizer_renyi_entropy(t1):.6f} "
          f"(expect {-np.log2(0.75):.6f})")

    print("\nStep 1: verify the dense SYK4 Hamiltonian is real Hermitian")
    n_majorana = 8
    n_side, terms = build_dense_syk4_terms(n_majorana, seed=61)
    H = pauli_hamiltonian_to_matrix(terms, n_side)
    print(f"  n_side={n_side} qubits, {len(terms)} terms "
          f"(expect C(8,4)={len(list(itertools.combinations(range(8), 4)))})")
    print(f"  Hermitian: {np.allclose(H, H.conj().T)}, "
          f"eigenvalue range [{np.linalg.eigvalsh(H).min():.3f}, {np.linalg.eigvalsh(H).max():.3f}]")

    print("\nStep 2: resolve the coupling-operator Hermiticity convention empirically")
    n_full = 2 * n_side + 2
    print("  without Klein fix (documented bug):")
    _sanity_check_coupling_unitary(n_majorana, n_side, n_full, use_klein_fix=False)

    print("\nStep 3: Klein-factor fix -- verify n_i and the coupling operator")
    print("  n_i = c_i^dagger c_i eigenvalues, WITHOUT Klein fix (expect degenerate 0.5):")
    n_op_bug = _dirac_number_operator(3, n_majorana, n_side, n_full, use_klein_fix=False)
    eigs_bug = np.linalg.eigvalsh(n_op_bug)
    print(f"    eigenvalues: min={eigs_bug.min():.6f} max={eigs_bug.max():.6f} "
          f"unique(round4)={sorted(set(np.round(eigs_bug, 4)))}")

    print("  n_i = c_i^dagger c_i eigenvalues, WITH Klein fix (expect proper 0/1):")
    n_op_fixed = _dirac_number_operator(3, n_majorana, n_side, n_full, use_klein_fix=True)
    eigs_fixed = np.linalg.eigvalsh(n_op_fixed)
    print(f"    eigenvalues: min={eigs_fixed.min():.6f} max={eigs_fixed.max():.6f} "
          f"unique(round4)={sorted(set(np.round(eigs_fixed, 4)))}")

    print("  coupling operator e^{igV}, WITH Klein fix:")
    _sanity_check_coupling_unitary(n_majorana, n_side, n_full, use_klein_fix=True)
