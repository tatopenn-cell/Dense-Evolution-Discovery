"""
Density-matrix ZNE validation: a real measured noise-coherence trajectory,
and a direct comparison against scalar ZNE on a non-Pauli noise channel.

Origin: an August 2025 personal notebook ("qualia") modeled subjective
experience as vectors in a Hilbert space -- linear/nonlinear operators, a
discrete entanglement operator, Gaussian noise (structured noise from
Riemann zeros planned as future work) -- then fed the resulting invented
state trajectory to an LLM ("Sophia") for reflection. This script closes
that loop with real data instead of invented states.

Part 1 (run_trajectory): a genuine noise-coherence-correction trajectory
from `dense_evolution.mitigation`, measured on an actual simulated 2-qubit
Bell state under depolarizing noise, not synthesized. Unlike
zne_mitigation.py in this same repo (a hand-rolled stochastic dephasing +
Richardson protocol), this uses the published
`zne_density_matrix`/`uhlmann_fidelity` pipeline directly -- the
density-matrix extension of ZNE, not the scalar/vector one.

Part 2 (run_scalar_vs_density_matrix_comparison): the question Part 1
alone doesn't answer -- does the density-matrix extension actually do
something scalar ZNE can't, or does it just improve fidelity the way any
reasonable ZNE variant would? `zne_density_matrix` extrapolates the full
density matrix, then projects the (possibly unphysical, small-negative-
eigenvalue) extrapolated result back onto the nearest true density matrix
via `project_to_physical`'s Smolin-Gambetta-Smith projection. Plain
scalar ZNE (`richardson_extrapolate` on raw fidelity numbers) has no such
constraint -- Richardson extrapolation can and does overshoot, so its
"fidelity" estimate can land outside [0, 1], a value with no physical
meaning. Depolarizing noise (a Pauli mixture) tends to keep density
matrices well-behaved even under naive extrapolation; amplitude damping
(a genuinely non-Pauli channel, `NoiseModel`'s `amplitude_damping`) is a
sharper test, since its asymmetric decay is more likely to push a
Richardson-extrapolated scalar past a physical bound.

Produces `data/sophia_reflection.csv` (Part 1) and
`data/scalar_vs_density_matrix_zne.csv` (Part 2), plotted in
`images/sophia_reflection.png` and
`images/scalar_vs_density_matrix_zne.png`, summarized in
`SOPHIA_REFLECTION.md`.

    python scripts/sophia_reflection.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix, richardson_extrapolate

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 2
SCALES = (1.0, 2.0, 3.0)


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    return np.asarray(sim.get_statevector())


def _noisy_density_matrix(ideal_sv, p, k, rng, channel='depolarizing'):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, channel, p, rng=rng)
        rho += np.outer(sv_noisy, sv_noisy.conj())
    rho /= k
    return jnp.asarray(rho, dtype=jnp.complex128)


def run_trajectory(base_p_sweep, k_trajectories, seed=0):
    """Real (not invented) noise->correction trajectory for a Bell state
    under depolarizing noise, one row per base_p in base_p_sweep."""
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rows = []
    for base_p in base_p_sweep:
        rho_at_scales = jnp.stack([
            _noisy_density_matrix(ideal_sv, base_p * scale, k_trajectories, rng)
            for scale in SCALES
        ])
        raw = float(uhlmann_fidelity(rho_at_scales[0], rho_ideal))
        corrected_rho = zne_density_matrix(rho_at_scales, SCALES)
        corrected = float(uhlmann_fidelity(corrected_rho, rho_ideal))
        rows.append({
            'base_p': float(base_p),
            'raw_fidelity': raw,
            'corrected_fidelity': corrected,
            'delta': corrected - raw,
        })
    return rows


def run_scalar_vs_density_matrix_comparison(base_p_sweep, k_trajectories, channel='amplitude_damping', seed=10):
    """Real, measured comparison of scalar ZNE against the density-matrix
    extension on the same noisy Bell state, same noise scales, same
    trajectories -- only the extrapolation target differs.

    Scalar path: raw Uhlmann fidelity computed independently at each of
    the 3 noise scales, then `richardson_extrapolate` applied directly to
    those 3 scalar numbers -- no constraint that the result stay in
    [0, 1], since a bare fidelity number carries no positive-semidefinite/
    trace-1 structure for any projection step to enforce.

    Density-matrix path: identical to run_trajectory -- extrapolate the
    density matrix itself, project back onto the physical set
    (`project_to_physical`, inside `zne_density_matrix`), then compute
    fidelity from the corrected (guaranteed-physical) density matrix.

    One row per base_p, reporting both estimates and whether the scalar
    path's raw extrapolated number left [0, 1] (a physically meaningless
    "fidelity" if so) at that noise level.
    """
    ideal_sv = bell_state_sv()
    rho_ideal = jnp.asarray(np.outer(ideal_sv, ideal_sv.conj()), dtype=jnp.complex128)
    rng = np.random.default_rng(seed)

    rows = []
    for base_p in base_p_sweep:
        rho_at_scales = jnp.stack([
            _noisy_density_matrix(ideal_sv, base_p * scale, k_trajectories, rng, channel=channel)
            for scale in SCALES
        ])
        fidelities_at_scales = jnp.array([
            uhlmann_fidelity(rho_at_scales[i], rho_ideal) for i in range(len(SCALES))
        ])

        scalar_extrapolated = float(richardson_extrapolate(fidelities_at_scales, SCALES))
        scalar_unphysical = bool(scalar_extrapolated < 0.0 or scalar_extrapolated > 1.0)

        corrected_rho = zne_density_matrix(rho_at_scales, SCALES)
        density_matrix_fidelity = float(uhlmann_fidelity(corrected_rho, rho_ideal))

        rows.append({
            'base_p': float(base_p),
            'raw_fidelity': float(fidelities_at_scales[0]),
            'scalar_zne_fidelity': scalar_extrapolated,
            'scalar_zne_unphysical': scalar_unphysical,
            'density_matrix_zne_fidelity': density_matrix_fidelity,
            'divergence': abs(scalar_extrapolated - density_matrix_fidelity),
        })
    return rows


if __name__ == "__main__":
    _DATA_DIR.mkdir(exist_ok=True)
    _IMAGES_DIR.mkdir(exist_ok=True)

    rows = run_trajectory(base_p_sweep=np.linspace(0.02, 0.5, 16), k_trajectories=200, seed=0)
    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "sophia_reflection.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(f"mean delta:  {df['delta'].mean():+.6f}")
    print(f"delta range: [{df['delta'].min():+.6f}, {df['delta'].max():+.6f}]")
    print(f"positive:    {(df['delta'] > 0).sum()}/{len(df)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(df['base_p'], df['raw_fidelity'], 'o-', color='#c0392b', label='raw (uncorrected)')
    ax1.plot(df['base_p'], df['corrected_fidelity'], 'o-', color='#2980b9', label='corrected (ZNE)')
    ax1.set_ylabel('Uhlmann fidelity')
    ax1.set_title('Density-matrix ZNE: real noise-coherence trajectory (Bell state, depolarizing noise)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(df['base_p'], df['delta'], width=0.02, color='#27ae60')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('base_p (depolarizing noise probability)')
    ax2.set_ylabel('delta (corrected - raw)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_IMAGES_DIR / "sophia_reflection.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'sophia_reflection.png'}")

    print("\n" + "=" * 70)
    print("Scalar vs. density-matrix ZNE, amplitude-damping noise")
    print("=" * 70)

    rows2 = run_scalar_vs_density_matrix_comparison(
        base_p_sweep=np.linspace(0.05, 0.6, 16), k_trajectories=200, channel='amplitude_damping', seed=10)
    df2 = pd.DataFrame(rows2)
    df2.to_csv(_DATA_DIR / "scalar_vs_density_matrix_zne.csv", index=False)

    print(df2.to_string(index=False))
    n_unphysical = int(df2['scalar_zne_unphysical'].sum())
    print()
    print(f"scalar ZNE unphysical (fidelity outside [0,1]): {n_unphysical}/{len(df2)}")
    print(f"density-matrix ZNE unphysical: 0/{len(df2)} (guaranteed by project_to_physical)")
    print(f"mean |divergence|: {df2['divergence'].mean():.4f}")

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df2['base_p'], df2['raw_fidelity'], 'o-', color='#888888', label='raw (uncorrected)', alpha=0.6)
    ax.plot(df2['base_p'], df2['scalar_zne_fidelity'], 's-', color='#c0392b', label='scalar ZNE (fidelity extrapolated directly)')
    ax.plot(df2['base_p'], df2['density_matrix_zne_fidelity'], 'o-', color='#2980b9', label='density-matrix ZNE (physical by construction)')
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='physical bound (F=1)')
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    unphysical_mask = df2['scalar_zne_unphysical']
    if unphysical_mask.any():
        ax.scatter(df2.loc[unphysical_mask, 'base_p'], df2.loc[unphysical_mask, 'scalar_zne_fidelity'],
                   marker='x', s=120, color='red', zorder=5, label='scalar ZNE unphysical result')
    ax.set_xlabel('base_p (amplitude-damping probability)')
    ax.set_ylabel('Uhlmann fidelity')
    ax.set_title(f'Scalar vs. density-matrix ZNE under amplitude damping (non-Pauli channel)\n'
                 f'{n_unphysical}/{len(df2)} scalar-ZNE points land outside [0,1] -- density-matrix ZNE never does')
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(_IMAGES_DIR / "scalar_vs_density_matrix_zne.png", dpi=150)
    print(f"\nsaved plot: {_IMAGES_DIR / 'scalar_vs_density_matrix_zne.png'}")
