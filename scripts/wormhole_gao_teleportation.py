"""Fourth wormhole experiment: Gao's OWN traversable-wormhole teleportation
construction (arXiv:1911.07416, "A Traversable Wormhole Teleportation
Protocol in the SYK Model", Gao & Jafferis) -- not previously attempted on
Discovery. The two existing wormhole scripts both borrow the Gao-Jafferis-
Wall THEORETICAL FRAMEWORK but implement DIFFERENT, later papers:
scripts/wormhole_syk_teleportation.py and dashboard_core.wormhole's
run_wormhole_protocol* implement arXiv:2604.10090 (sparse binary SYK,
mutual-information readout, Trotterized coupling); scripts/
wormhole_magic_entropy.py implements arXiv:2606.19180 (Joshi & Mishra's
WITP, dense SYK4, fidelity readout). Neither is Gao's own construction.

WHAT'S DIFFERENT ABOUT GAO'S OWN COUPLING (verified by fetching the actual
paper text, arxiv.org/pdf/1911.07416, Eq. 50 and 57): the interaction is a
RAW BILINEAR V = (i^q/(q*N)) * sum_j psi_L^j * psi_R^j -- a product of ONE
Majorana from each side, q=4 for our SYK4 (i^4=1, so the prefactor is just
1/(4*N)). Unlike Joshi & Mishra's Eq. 1 Dirac fermion c_i=(psi_L^i+i*psi_R^i)/2
(which needs the Klein-factor fix in wormhole_magic_entropy.py to have
correct anticommutation), a bilinear product of two ALREADY-Hermitian,
commuting (independent-JW-registers convention) Majoranas is automatically
Hermitian with no cross-register algebra fix needed: (psi_L*psi_R)^dagger =
psi_R^dagger*psi_L^dagger = psi_R*psi_L = psi_L*psi_R when they commute and
are each individually Hermitian. This is also the exact convention
dashboard_core.wormhole's own V=sum(chi_L^j chi_R^j) coupling already uses
(noted in wormhole_magic_entropy.py's Klein-factor docstring) -- Gao's own
paper is the origin of that convention, just never wired up as its own
dense-SYK/fidelity-readout experiment until now.

WHAT'S REUSED, VERBATIM, FROM wormhole_magic_entropy.py: the dense SYK4
Hamiltonian builder, TFD state preparation (Eq. 5-equivalent), the message-
insertion Bell-projector (Eq. 7-equivalent, Joshi & Mishra's own version of
this step is compatible with and plausibly derived from Gao's own Eq. 57
same-side Dirac-fermion logical-qubit construction), the embedding helpers,
and the fidelity readout (Eq. 11-equivalent). Only the coupling operator
itself is Gao's own -- everything else is shared TFD-wormhole-teleportation
scaffolding, not something either paper claims as novel.

Sum in V starts at j=3 (excluding modes 1,2, which encode the message
qubit here too, same reasoning as Joshi & Mishra's own exclusion) -- not
explicitly confirmed in the fetched Eq. 50 text (which doesn't state an
exclusion), so this is a documented ASSUMPTION carried over from the other
paper's convention for consistency, not a verified detail of Gao's own
paper. Flagged here rather than silently assumed.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\Desktop\Dense-Evolution-main\Dense-Evolution-main\tools\dashboard")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from core.wormhole import _multiply_pauli_dicts, _embed  # noqa: E402
from dense_evolution import pauli_hamiltonian_to_matrix  # noqa: E402
from wormhole_magic_entropy import (  # noqa: E402
    J, build_dense_syk4_terms, _embed_matrix, build_tfd_state,
    _message_insertion_projector, _fidelity, stabilizer_renyi_entropy,
)


def gao_bilinear_coupling_unitary(n_majorana, n_side, n_full, mu):
    """Eq. 50: e^{i*mu*V}, V = (i^q/(q*N)) * sum_{j=3}^{N} psi_L^j*psi_R^j,
    q=4 (SYK4) so i^q=1 and the prefactor is just 1/(4*n_majorana). V is
    built as an explicit Hermitian matrix (small enough here, dim<=1024)
    and exponentiated via its own eigendecomposition -- no Klein-factor
    dressing needed, see the module docstring."""
    dim = 2 ** n_full
    V = np.zeros((dim, dim), dtype=complex)
    prefactor = 1.0 / (4.0 * n_majorana)
    for j in range(3, n_majorana + 1):
        pL = _embed(j, n_full, 0)
        pR = _embed(j, n_full, n_side)
        phase, merged = _multiply_pauli_dicts([pL, pR])
        V += prefactor * phase * pauli_hamiltonian_to_matrix([(1.0, merged)], n_full)
    eigvals, eigvecs = np.linalg.eigh(V)
    return (eigvecs * np.exp(1j * mu * eigvals)) @ eigvecs.conj().T, V


def prepare_gao_pre_extraction_state(n_majorana, seed, beta, mu, t_scr, use_klein_fix=None):
    """Same pipeline shape as wormhole_magic_entropy.prepare_pre_extraction_state
    (TFD -> backward scrambling -> message insertion -> forward re-scrambling
    -> coupling), with Gao's own bilinear coupling instead of the Klein-
    dressed number-operator one. `use_klein_fix` is accepted and ignored
    (kept only so callers can pass the same kwargs used for the other
    protocol without branching) -- Gao's coupling never needs it."""
    n_side = n_majorana // 2
    n_full = 2 * n_side + 2
    msg_qubit = 2 * n_side

    _, terms = build_dense_syk4_terms(n_majorana, seed)
    H = pauli_hamiltonian_to_matrix(terms, n_side)
    H_L = _embed_matrix(H, 0, n_full)
    H_R = _embed_matrix(H, n_side, n_full)

    tfd = build_tfd_state(H, beta)
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = np.kron(tfd, bell)

    E_L, V_L = np.linalg.eigh(H_L)
    psi = (V_L * np.exp(1j * t_scr * E_L)) @ (V_L.conj().T @ psi)  # Stage 1: backward

    psi_unnorm = _message_insertion_projector(n_full, msg_qubit) @ psi  # Stage 2
    p_success = float(np.linalg.norm(psi_unnorm) ** 2)
    psi = psi_unnorm / np.linalg.norm(psi_unnorm)

    psi = (V_L * np.exp(-1j * t_scr * E_L)) @ (V_L.conj().T @ psi)  # Stage 3: forward

    Uc, V_matrix = gao_bilinear_coupling_unitary(n_majorana, n_side, n_full, mu)
    psi = Uc @ psi  # Stage 4: Gao's own bilinear coupling

    return psi, H_R, p_success, V_matrix


def finish_gao_protocol(psi, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R):
    """Identical Stage 5 to wormhole_magic_entropy.finish_witp_protocol --
    right-side extraction evolution, then fidelity + SRE."""
    psi_t = (V_R * np.exp(-1j * t_R * E_R)) @ (V_R.conj().T @ psi)
    return _fidelity(psi_t, n_side, n_full, ref_qubit), stabilizer_renyi_entropy(psi_t)


if __name__ == "__main__":
    print("Step 0: verify Gao's bilinear V is exactly Hermitian and e^{i*mu*V} is unitary")
    n_majorana = 8
    n_side = n_majorana // 2
    n_full = 2 * n_side + 2
    mu_test = 12.0  # order of magnitude used elsewhere in this codebase for mu (see dashboard_core.wormhole docstrings)
    Uc_test, V_test = gao_bilinear_coupling_unitary(n_majorana, n_side, n_full, mu_test)
    print(f"  ||V - V^dagger|| = {np.linalg.norm(V_test - V_test.conj().T):.3e} (expect ~0, exactly Hermitian)")
    dim = Uc_test.shape[0]
    print(f"  ||U^dagger U - I|| = {np.linalg.norm(Uc_test.conj().T @ Uc_test - np.eye(dim)):.3e} (expect ~0)")

    print("\nStep 1: fidelity/SRE vs t_R scan, Gao's own coupling, both mu signs")
    beta_units = 1.0
    beta = beta_units / J
    t_scr_units = 1.0
    t_scr = t_scr_units / J
    t_max = t_scr + 6.0 / J
    t_R_values = np.linspace(0.0, t_max, 28)
    ref_qubit = 2 * n_side + 1

    for mu, label in ((-12.0, "mu=-12 (traversable, per this codebase's own sign convention)"),
                      (12.0, "mu=+12 (non-traversable)")):
        psi_pre, H_R, p_success, _ = prepare_gao_pre_extraction_state(
            n_majorana, seed=61, beta=beta, mu=mu, t_scr=t_scr)
        E_R, V_R = np.linalg.eigh(H_R)
        fidelities, sres = [], []
        for t_R in t_R_values:
            fid, sre = finish_gao_protocol(psi_pre, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R)
            fidelities.append(fid)
            sres.append(sre)
        fidelities = np.array(fidelities)
        sres = np.array(sres)
        print(f"  {label}: p_success={p_success:.4f}, "
              f"fidelity min={fidelities.min():.4f} max={fidelities.max():.4f}, "
              f"SRE min={sres.min():.4f} max={sres.max():.4f}")
        np.save(f"witp_gao_fidelity_mu{int(mu)}.npy", fidelities)
        np.save(f"witp_gao_sre_mu{int(mu)}.npy", sres)
    np.save("witp_gao_tR_scan.npy", t_R_values)
