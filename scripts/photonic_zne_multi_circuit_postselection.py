"""
Multi-circuit, multi-size extension of photonic_predictive_zne.py, plus
the one comparison that script never made: against TRUE postselection --
the actual baseline Mills & Mezher (arXiv:2405.02278) find nothing beats
for discrete-variable photon loss. photonic_predictive_zne.py compared
against plain scalar Richardson ZNE (which the paper also criticizes,
and which this repo independently reproduced going unphysical), never
against postselection itself -- a real gap this script closes.

Two circuit families, at n_qubits in (2, 3, 4):
- GHZ states (dense_evolution.states.ghz_state) -- the family already
  used in photonic_predictive_zne.py, extended here beyond n_qubits=2.
- A hardware-efficient VQE-style ansatz (RY rotation layer + linear CX
  entangling layer, repeated n_layers times, fixed non-optimized random
  parameters) -- deeper and structurally different from GHZ, chosen
  because VQE circuits are the primary real-world application of
  near-term quantum simulation (molecular/materials simulation), so
  whether this photonic-noise correction generalizes to VQE-style
  circuits and not just entanglement-witness states like GHZ is a real,
  practically-relevant question, not just a theoretical curiosity.

TRUE postselection, not an approximation: `_apply_amplitude_damping_
tracked` independently reimplements NoiseModel.apply_to_sv's own
Born-rule-correct amplitude_damping formula (this repo's own fix
earlier this session), verified to match it exactly, but additionally
returns whether ANY qubit's decay (K1) Kraus branch fired anywhere in
that single-shot trajectory -- the same information a real photonic
experiment's heralding detectors would give per shot. Postselection
here means: average only the trajectories where every qubit's decay
indicator came back all-False (no loss detected anywhere in that shot),
exactly matching what "keep only heralded-successful shots" means
physically, not a rough proxy for it.

VERIFICATION NOTE (2026-08-09): a first version of
`_apply_amplitude_damping_tracked` was missing NoiseModel.apply_to_sv's
own final global renormalization step (`sv_out / (norm(sv_out) + eps)`,
applied once after the whole per-qubit loop) -- found via a direct
numerical comparison against the real library output, not assumed
identical from matching formulas alone (max diff was ~0.18, not
floating-point noise, before the fix). Now verified bit-exact
(diff=0.00e+00) against NoiseModel.apply_to_sv across n_qubits in
(2, 3, 4), 20 trials each, before being trusted for postselection
tracking.

REAL RESULTS (2026-08-09, 18 configurations: 2 circuit families x 3
qubit counts x 3 loss rates, K=200 trajectories each):

Postselection wins in 14/18 configurations. Mean gap: JSD ZNE trails
postselection by -0.0345, plain density-matrix ZNE by -0.0362 -- the
JSD nudge narrows the gap slightly but does not close it. This is a
direct, broader confirmation of Mills & Mezher's core finding
(postselection is hard to beat for discrete-variable photon loss),
extended here across both circuit families and all 3 tested qubit
counts, not just the single Bell-state case photonic_predictive_zne.py
tested. The 4/18 cases where ZNE wins are small margins (+0.002 to
+0.007, all at eta=0.8 or eta=0.9 -- lower loss); every loss at
eta=0.7 (the highest tested) favors postselection, several by a wide
margin (up to -0.15 for VQE-4q).

Conclusion: `jsd_predictive_zne_density_matrix` (promoted to
dense_evolution.mitigation) is a real, measured improvement over plain
`zne_density_matrix` for photon-loss noise -- but neither approach
generally beats postselection when postselection is a viable option
(i.e. when loss events are heralded/detectable, as they typically are
in real linear-optical photonic hardware). The practical case for
either ZNE variant over postselection is narrower than it might first
appear: scenarios where loss events are NOT independently heralded (so
postselection isn't available at all), or where the lost-shot fraction
is high enough that discarding them is unacceptably wasteful of
measurement budget -- neither tested here.
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

import dense_evolution as de
from dense_evolution.mitigation import uhlmann_fidelity, richardson_extrapolate, zne_density_matrix

try:
    # jsd_predictive_zne_density_matrix was promoted into dense_evolution
    # (see mitigation.py's own changelog entry) but hasn't shipped in a
    # PyPI release yet as of this script's own commit -- requirements-ci.txt
    # installs the latest PyPI release unpinned, so CI here would hard-fail
    # on an ImportError until that release ships, not because anything is
    # actually broken. HAS_JSD_ZNE lets run_multi_circuit_comparison and
    # its tests degrade gracefully (skip, not fail) in the meantime rather
    # than block on a release timeline this script doesn't control.
    from dense_evolution.mitigation import jsd_predictive_zne_density_matrix
    HAS_JSD_ZNE = True
except ImportError:
    HAS_JSD_ZNE = False


def photon_loss_kraus_probability(eta: float) -> float:
    """Same formula as photonic_predictive_zne.py's own function of the
    same name (gamma = 1 - eta) -- reimplemented here rather than
    imported across sibling scripts, since importlib-based script
    loading (this repo's own test-import pattern, and how this script
    is loaded when run directly) doesn't put scripts/ on sys.path for
    sibling imports to resolve; self-contained scripts avoid that
    fragility entirely, the same reason _js_divergence in
    photonic_predictive_zne.py is reimplemented rather than imported."""
    return float(np.clip(1.0 - eta, 0.0, 1.0))


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

SCALES = (1.0, 2.0, 3.0)


def _qubit_index_pairs(dim, q):
    """Same LSB-based (1<<q) bit convention as NoiseModel.apply_to_sv's
    own _qubit_index_pairs -- verified to match by construction (same
    formula, independently re-derived here rather than imported, since
    that helper is private to dense_evolution.registry)."""
    all_idx = np.arange(dim)
    mask = 1 << q
    idx_1 = all_idx[(all_idx & mask) != 0]
    idx_0 = idx_1 - mask
    return idx_0, idx_1


def _apply_amplitude_damping_tracked(sv, n_qubits, gamma, rng):
    """Independently reimplements NoiseModel.apply_to_sv's amplitude_
    damping branch (this repo's own Born-rule fix earlier this session)
    -- same formula exactly, additionally returns whether ANY qubit's
    decay (K1) branch fired anywhere in this single-shot trajectory, so
    postselection can be computed honestly (real heralding-detector
    information), not approximated after the fact from the output
    state alone (which loses which-branch-decayed information once the
    branches are coherently combined back into one statevector)."""
    dim = len(sv)
    sv_out = sv.copy()
    any_decay = False
    for q in range(n_qubits):
        idx_0, idx_1 = _qubit_index_pairs(dim, q)
        half = len(idx_0)
        r = rng.random(half)
        v0, v1 = sv_out[idx_0].copy(), sv_out[idx_1].copy()
        p_decay = gamma * np.abs(v1) ** 2
        decay = r < p_decay
        if np.any(decay):
            any_decay = True
        sq_1m_gamma = np.sqrt(1.0 - gamma)
        p_no_decay = np.maximum(1.0 - p_decay, 1e-15)
        norm_no_decay = np.sqrt(p_no_decay)
        phase_v1 = v1 / (np.abs(v1) + 1e-15)
        sv_out[idx_0] = np.where(decay, phase_v1, v0 / norm_no_decay)
        sv_out[idx_1] = np.where(decay, 0.0 + 0j, v1 * sq_1m_gamma / norm_no_decay)
    # NoiseModel.apply_to_sv's own final step: a global renormalization
    # after the full per-qubit loop -- missed in an earlier version of
    # this reimplementation, caught by a direct numerical comparison
    # against the library's real output (max diff was ~0.18, not
    # floating-point noise) before this function was trusted for
    # postselection tracking. Multi-qubit sequential per-branch Kraus
    # application does not automatically stay exactly unit-norm.
    norm = np.linalg.norm(sv_out)
    sv_out = sv_out / (norm + 1e-15)
    return sv_out, any_decay


def _noisy_density_matrices_with_postselection(ideal_sv, n_qubits, gamma, k, rng):
    dim = len(ideal_sv)
    rho_all = np.zeros((dim, dim), dtype=np.complex128)
    rho_kept = np.zeros((dim, dim), dtype=np.complex128)
    n_kept = 0
    for _ in range(k):
        sv_noisy, any_decay = _apply_amplitude_damping_tracked(ideal_sv.copy(), n_qubits, gamma, rng)
        rho_all += np.outer(sv_noisy, sv_noisy.conj())
        if not any_decay:
            rho_kept += np.outer(sv_noisy, sv_noisy.conj())
            n_kept += 1
    rho_all /= k
    rho_kept = (rho_kept / n_kept) if n_kept > 0 else rho_all.copy()
    return rho_all, rho_kept, n_kept / k


def vqe_ansatz_ops(n_qubits, params, n_layers=2):
    """Hardware-efficient VQE-style ansatz: RY layer + linear CX
    entangling layer, repeated n_layers times. Fixed, non-optimized
    parameters -- this tests noise robustness on a representative deep
    parametrized circuit shape, not a solved VQE instance."""
    ops = []
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            ops.append(('ry', q, float(params[idx])))
            idx += 1
        for q in range(n_qubits - 1):
            ops.append(('cx', q, q + 1))
    return ops


def _circuit_specs(n_layers=2):
    specs = []
    for n_qubits in (2, 3, 4):
        specs.append((f'GHZ-{n_qubits}q', n_qubits, lambda nq=n_qubits: de.ghz_state(nq)))
        rng = np.random.default_rng(100 + n_qubits)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * n_layers)
        specs.append((f'VQE-{n_qubits}q', n_qubits,
                      lambda nq=n_qubits, p=params: vqe_ansatz_ops(nq, p, n_layers=n_layers)))
    return specs


def run_multi_circuit_comparison(eta_sweep, k_trajectories=200, seed=0):
    """Real comparison across circuit family x n_qubits x eta, of 5
    correction paths on the SAME noisy trajectories: raw, scalar ZNE,
    density-matrix ZNE, JSD-predictive density-matrix ZNE, and TRUE
    postselection (see module docstring)."""
    rows = []
    for circuit_name, n_qubits, build_ops in _circuit_specs():
        sim = de.DenseSVSimulator(n_qubits)
        sim.run_circuit(build_ops())
        ideal_sv = np.asarray(sim.get_statevector())
        rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
        rng = np.random.default_rng(seed)

        for eta in eta_sweep:
            gamma_base = photon_loss_kraus_probability(eta)
            rho_at_scales_np = []
            postselect_base_rho = None
            postselect_keep_frac = None
            for i, scale in enumerate(SCALES):
                gamma = min(gamma_base * scale, 1.0)
                rho_all, rho_kept, keep_frac = _noisy_density_matrices_with_postselection(
                    ideal_sv, n_qubits, gamma, k_trajectories, rng)
                rho_at_scales_np.append(rho_all)
                if i == 0:
                    postselect_base_rho = rho_kept
                    postselect_keep_frac = keep_frac

            rho_at_scales = jnp.stack([jnp.asarray(r, dtype=jnp.complex128) for r in rho_at_scales_np])
            fidelities_at_scales = jnp.array([
                uhlmann_fidelity(rho_at_scales[i], rho_ideal) for i in range(len(SCALES))
            ])

            raw = float(fidelities_at_scales[0])
            scalar_zne = float(richardson_extrapolate(fidelities_at_scales, SCALES))
            dm_zne_rho = zne_density_matrix(rho_at_scales, SCALES)
            dm_zne = float(uhlmann_fidelity(dm_zne_rho, rho_ideal))
            postselection = float(uhlmann_fidelity(
                jnp.asarray(postselect_base_rho, dtype=jnp.complex128), rho_ideal))

            if HAS_JSD_ZNE:
                jsd_rho = jsd_predictive_zne_density_matrix(rho_at_scales, SCALES)
                jsd_dm_zne = float(uhlmann_fidelity(jsd_rho, rho_ideal))
                jsd_vs_postselection = jsd_dm_zne - postselection
            else:
                jsd_dm_zne = float('nan')
                jsd_vs_postselection = float('nan')

            rows.append({
                'circuit': circuit_name,
                'n_qubits': n_qubits,
                'eta': float(eta),
                'raw_fidelity': raw,
                'scalar_zne_fidelity': scalar_zne,
                'dm_zne_fidelity': dm_zne,
                'jsd_dm_zne_fidelity': jsd_dm_zne,
                'postselection_fidelity': postselection,
                'postselection_keep_fraction': postselect_keep_frac,
                'jsd_vs_postselection': jsd_vs_postselection,
                'dm_zne_vs_postselection': dm_zne - postselection,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    df = run_multi_circuit_comparison(eta_sweep=np.array([0.9, 0.8, 0.7]), k_trajectories=200, seed=0)
    df.to_csv(_DATA_DIR / "photonic_multi_circuit_postselection.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(f"jsd_dm_zne vs postselection: mean {df['jsd_vs_postselection'].mean():+.6f}, "
          f"{(df['jsd_vs_postselection'] > 0).sum()}/{len(df)} jsd wins")
    print(f"plain dm_zne vs postselection: mean {df['dm_zne_vs_postselection'].mean():+.6f}, "
          f"{(df['dm_zne_vs_postselection'] > 0).sum()}/{len(df)} dm_zne wins")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, (circuit_name, grp) in zip(axes.flat, df.groupby('circuit')):
        ax.plot(grp['eta'], grp['raw_fidelity'], 'o-', color='#7f8c8d', label='raw')
        ax.plot(grp['eta'], grp['dm_zne_fidelity'], 'o-', color='#2980b9', label='dm ZNE')
        ax.plot(grp['eta'], grp['jsd_dm_zne_fidelity'], 'o-', color='#27ae60', label='JSD ZNE')
        ax.plot(grp['eta'], grp['postselection_fidelity'], 'o-', color='#e67e22', label='postselection')
        ax.set_title(circuit_name)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('eta')
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "photonic_multi_circuit_postselection.png", dpi=300)
    print(f"\nSaved data/photonic_multi_circuit_postselection.csv, "
          f"images/photonic_multi_circuit_postselection.png")
