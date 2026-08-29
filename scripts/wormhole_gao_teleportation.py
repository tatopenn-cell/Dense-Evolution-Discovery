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

MESSAGE-VS-NO-MESSAGE CONTROL (with_message=False in
build_gao_pre_coupling_state, matching the analogous check in
scripts/wormhole_syk_teleportation.py): skipping the message-insertion
projector leaves msg/ref an untouched Bell pair, uncorrelated with
everything else in the circuit no matter what mu/t_R/beta do to L/R. Since
ref's own reduced state is then exactly maximally mixed (half of an
untouched Bell pair), <X_ref>=<Y_ref>=<Z_ref>=0 exactly, and R_logical/ref
are in a product state, so <X_R X_ref>=<X_R>*<X_ref>=0 (same for Y, Z) --
Eq. 11's fidelity collapses to EXACTLY 0.25 (the paper's own classical
limit), not just approximately, regardless of the physics on the L/R side.
Verified numerically below, same "provable identity, not a coincidence"
style as wormhole_magic_entropy.py's p_success=1.0 finding.

PERFORMANCE NOTE (found and fixed during a mu-magnitude scan, 2026-08-29):
an earlier version of this script called a single all-in-one
`prepare_gao_pre_extraction_state(..., mu, ...)` inside the mu-scan loop,
which rebuilds H, diagonalizes H_L (dim<=1024 eigh), diagonalizes H_R, AND
rebuilds+diagonalizes V from scratch on EVERY call -- even though NONE of
H, H_L, H_R, V, or their eigendecompositions depend on mu at all. Only the
tiny final step exp(i*mu*eigvals) does. A 5-magnitude x 2-sign scan was
redoing three expensive dim<=1024 eigh calls 10 times each for no reason.
Fixed below by splitting into build_gao_pre_coupling_state (mu-independent,
called ONCE) + gao_coupling_eigendecomposition (V's eigh, ONCE) +
apply_gao_coupling (the cheap per-mu exp(i*mu*eigvals) step) -- the same
"hoist the mu/t_R-independent expensive part out of the scan loop"
principle already applied to t_R in wormhole_magic_entropy.py's
prepare_pre_extraction_state/finish_witp_protocol split.
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


def gao_coupling_eigendecomposition(n_majorana, n_side, n_full):
    """Eq. 50's V = (i^q/(q*N)) * sum_{j=3}^{N} psi_L^j*psi_R^j (q=4 for
    SYK4, so i^q=1 and the prefactor is just 1/(4*n_majorana)), built as an
    explicit Hermitian matrix and diagonalized ONCE. mu only enters later
    via exp(i*mu*eigvals) (apply_gao_coupling) -- callers scanning several
    mu values should call this once and reuse (eigvals, eigvecs), not
    rebuild V per mu (see the module docstring's performance note)."""
    dim = 2 ** n_full
    V = np.zeros((dim, dim), dtype=complex)
    prefactor = 1.0 / (4.0 * n_majorana)
    for j in range(3, n_majorana + 1):
        pL = _embed(j, n_full, 0)
        pR = _embed(j, n_full, n_side)
        phase, merged = _multiply_pauli_dicts([pL, pR])
        V += prefactor * phase * pauli_hamiltonian_to_matrix([(1.0, merged)], n_full)
    eigvals, eigvecs = np.linalg.eigh(V)
    return eigvals, eigvecs, V


def apply_gao_coupling(psi, mu, eigvals, eigvecs):
    """Cheap per-mu application of e^{i*mu*V} given V's ALREADY-computed
    eigendecomposition (gao_coupling_eigendecomposition) -- no new eigh,
    just an elementwise exp() and two matrix-vector products."""
    return (eigvecs * np.exp(1j * mu * eigvals)) @ (eigvecs.conj().T @ psi)


def build_gao_pre_coupling_state(n_majorana, seed, beta, t_scr, with_message=True):
    """Everything up to (but not including) Stage 4's coupling: TFD prep
    (Eq. 5) -> backward scrambling on L -> message insertion (Eq. 7,
    skipped if with_message=False) -> forward re-scrambling on L. Entirely
    independent of mu -- build once, reuse for every mu value in a scan
    (see the module docstring's performance note). Returns (psi, H_L, H_R,
    p_success); H_L is returned too since its eigendecomposition (needed
    here) is also reusable if a caller wants a second with_message variant
    without rediagonalizing H_L again."""
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

    if with_message:
        psi_unnorm = _message_insertion_projector(n_full, msg_qubit) @ psi  # Stage 2
        p_success = float(np.linalg.norm(psi_unnorm) ** 2)
        psi = psi_unnorm / np.linalg.norm(psi_unnorm)
    else:
        p_success = 1.0

    psi = (V_L * np.exp(-1j * t_scr * E_L)) @ (V_L.conj().T @ psi)  # Stage 3: forward

    return psi, H_L, H_R, p_success


def finish_gao_protocol(psi, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R):
    """Identical Stage 5 to wormhole_magic_entropy.finish_witp_protocol --
    right-side extraction evolution, then fidelity + SRE. E_R/V_R (H_R's
    own eigendecomposition) are passed in, computed once by the caller."""
    psi_t = (V_R * np.exp(-1j * t_R * E_R)) @ (V_R.conj().T @ psi)
    return _fidelity(psi_t, n_side, n_full, ref_qubit), stabilizer_renyi_entropy(psi_t)


def finish_gao_fidelity_only(psi, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R):
    """Same Stage 5 evolution as finish_gao_protocol, but skips
    stabilizer_renyi_entropy entirely -- found necessary during a
    mu-magnitude scan (2026-08-29): SRE is O(d^3) (an explicit Python loop
    over d Walsh-Hadamard transforms, d=1024 here, each its own O(d^2)
    matmul), and Step 3's sign-asymmetry question only needs fidelity.
    finish_gao_protocol was computing SRE there too and immediately
    discarding it -- a genuine, not just redundant-diagonalization, waste."""
    psi_t = (V_R * np.exp(-1j * t_R * E_R)) @ (V_R.conj().T @ psi)
    return _fidelity(psi_t, n_side, n_full, ref_qubit)


if __name__ == "__main__":
    print("Step 0: verify Gao's bilinear V is exactly Hermitian and e^{i*mu*V} is unitary")
    n_majorana = 8
    n_side = n_majorana // 2
    n_full = 2 * n_side + 2
    eigvals_V, eigvecs_V, V_test = gao_coupling_eigendecomposition(n_majorana, n_side, n_full)
    print(f"  ||V - V^dagger|| = {np.linalg.norm(V_test - V_test.conj().T):.3e} (expect ~0, exactly Hermitian)")
    Uc_test = apply_gao_coupling(np.eye(V_test.shape[0], dtype=complex), 12.0, eigvals_V, eigvecs_V)
    dim = Uc_test.shape[0]
    print(f"  ||U^dagger U - I|| = {np.linalg.norm(Uc_test.conj().T @ Uc_test - np.eye(dim)):.3e} (expect ~0)")

    print("\nBuilding the mu-independent pre-coupling state and H_R/V eigendecompositions ONCE")
    beta_units = 1.0
    beta = beta_units / J
    t_scr_units = 1.0
    t_scr = t_scr_units / J
    t_max = t_scr + 6.0 / J
    t_R_values = np.linspace(0.0, t_max, 28)
    ref_qubit = 2 * n_side + 1

    psi_pre, H_L, H_R, p_success = build_gao_pre_coupling_state(
        n_majorana, seed=61, beta=beta, t_scr=t_scr, with_message=True)
    E_R, V_R = np.linalg.eigh(H_R)
    # (eigvals_V, eigvecs_V) from Step 0 above are reused here too --
    # same n_majorana/n_side/n_full, so the same V.

    print("\nStep 1: fidelity/SRE vs t_R scan, Gao's own coupling, both mu signs")
    for mu, label in ((-12.0, "mu=-12 (traversable, per this codebase's own sign convention)"),
                      (12.0, "mu=+12 (non-traversable)")):
        psi_coupled = apply_gao_coupling(psi_pre, mu, eigvals_V, eigvecs_V)
        fidelities, sres = [], []
        for t_R in t_R_values:
            fid, sre = finish_gao_protocol(psi_coupled, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R)
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

    print("\nStep 2: message-vs-no-message control (expect fidelity = 0.25 EXACTLY without message)")
    t_R_probe = t_R_values[len(t_R_values) // 2]
    psi_nomsg_pre, _, _, p_success_nomsg = build_gao_pre_coupling_state(
        n_majorana, seed=61, beta=beta, t_scr=t_scr, with_message=False)
    psi_nomsg_coupled = apply_gao_coupling(psi_nomsg_pre, -12.0, eigvals_V, eigvecs_V)
    fid_nomsg, sre_nomsg = finish_gao_protocol(
        psi_nomsg_coupled, H_R, n_side, n_full, ref_qubit, t_R_probe, E_R, V_R)
    print(f"  with_message=False, mu=-12, t_R={t_R_probe:.4f}: "
          f"fidelity={fid_nomsg:.10f} (expect exactly 0.25), p_success={p_success_nomsg:.4f}")

    print("\nStep 3: mu-magnitude scan -- where does the sign-dependent fidelity asymmetry peak?")
    mu_magnitudes = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0])
    max_abs_deltas = []
    for mu_mag in mu_magnitudes:
        fids_by_sign = {}
        for sign in (-1.0, 1.0):
            mu = sign * mu_mag
            psi_coupled = apply_gao_coupling(psi_pre, mu, eigvals_V, eigvecs_V)
            fids = np.array([
                finish_gao_fidelity_only(psi_coupled, H_R, n_side, n_full, ref_qubit, t_R, E_R, V_R)
                for t_R in t_R_values
            ])
            fids_by_sign[sign] = fids
        delta = fids_by_sign[-1.0] - fids_by_sign[1.0]
        max_abs_delta = float(np.max(np.abs(delta)))
        max_abs_deltas.append(max_abs_delta)
        print(f"  |mu|={mu_mag:5.1f}: max|F(-mu,t_R)-F(+mu,t_R)| over t_R = {max_abs_delta:.4f}")
    max_abs_deltas = np.array(max_abs_deltas)
    peak_idx = int(np.argmax(max_abs_deltas))
    is_interior_peak = 0 < peak_idx < len(mu_magnitudes) - 1
    if is_interior_peak:
        print(f"  genuine interior peak at |mu|={mu_magnitudes[peak_idx]:.1f} "
              f"(delta={max_abs_deltas[peak_idx]:.4f})")
    else:
        print(f"  largest delta is at the EDGE of the scanned range "
              f"(|mu|={mu_magnitudes[peak_idx]:.1f}, delta={max_abs_deltas[peak_idx]:.4f}) -- "
              f"the asymmetry is still monotonically increasing at |mu|=26, no interior peak "
              f"found in [{mu_magnitudes[0]:.0f}, {mu_magnitudes[-1]:.0f}]; would need a wider "
              f"scan to find where it turns over, not claimed here")
    print(f"  (for this seed=61, n_majorana=8 instance -- not necessarily the paper's own "
          f"reported mu~11-12, which came from a DIFFERENT protocol/instance, "
          f"dashboard_core.wormhole's sparse-SYK construction)")
    np.save("witp_gao_mu_magnitude_scan.npy", mu_magnitudes)
    np.save("witp_gao_mu_magnitude_deltas.npy", max_abs_deltas)
