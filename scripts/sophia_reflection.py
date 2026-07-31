"""
Sophia reflection: a real density-matrix ZNE noise-coherence trajectory.

Origin: an August 2025 personal notebook ("qualia") modeled subjective
experience as vectors in a Hilbert space -- linear/nonlinear operators, a
discrete entanglement operator, Gaussian noise (structured noise from
Riemann zeros planned as future work) -- then fed the resulting invented
state trajectory to an LLM ("Sophia") for reflection.

This script closes that loop with real data instead of invented states: a
genuine noise-coherence-correction trajectory from `dense_evolution.
mitigation`, measured on an actual simulated 2-qubit Bell state under
depolarizing noise, not synthesized. Unlike zne_mitigation.py in this same
repo (a hand-rolled stochastic dephasing + Richardson protocol), this uses
the published `zne_density_matrix`/`uhlmann_fidelity` pipeline directly --
the density-matrix extension of ZNE, not the scalar/vector one.

Produces `data/sophia_reflection.csv`. The reflection an LLM (Claude) wrote
after reading this script's real output -- not a simulated stand-in
"Sophia" persona, not fabricated in advance of seeing the numbers -- lives
alongside it in `SOPHIA_REFLECTION.md`.

    python scripts/sophia_reflection.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import dense_evolution as de
from dense_evolution.registry import NoiseModel
from dense_evolution.mitigation import uhlmann_fidelity, zne_density_matrix

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"

N_QUBITS = 2
SCALES = (1.0, 2.0, 3.0)


def bell_state_sv():
    sim = de.DenseSVSimulator(N_QUBITS)
    sim.run_circuit([("h", 0), ("cx", 0, 1)])
    return np.asarray(sim.get_statevector())


def _noisy_density_matrix(ideal_sv, p, k, rng):
    dim = len(ideal_sv)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for _ in range(k):
        sv_noisy = NoiseModel.apply_to_sv(ideal_sv.copy(), N_QUBITS, 'depolarizing', p, rng=rng)
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
    ax1.set_title('Sophia reflection: real ZNE noise-coherence trajectory (Bell state, depolarizing noise)')
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
