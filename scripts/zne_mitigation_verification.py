"""
Independent verification of scripts/zne_mitigation.py's headline claim
(README section "2. Quantum Error Mitigation via Real Stochastic Richardson
Extrapolation (ZNE)"): that the ZNE protocol recovers a "true analytic
target value of -4.2467 eV" at k=0, without introducing artifacts.

Two things are checked independently of zne_mitigation.py's own
calcola_aspettazione_hamiltoniana():
  1. What is the actual ideal (noise_scale=0) energy at a given k, via
     (a) an explicit sparse Pauli Hamiltonian built with scipy.sparse and
         a plain <psi|H|psi> matrix-vector product, and
     (b) a hand-derived closed form.
  2. Whether E(noise_scale) is actually linear near noise_scale=0..2, by
     sampling a finer grid and comparing a 2-point Richardson extrapolant
     against a quadratic fit and the independently-verified true E(0).

generate_bloch_state / apply_stochastic_dephasing / measure_energy_with_shots
/ calcola_aspettazione_hamiltoniana are imported directly from the real
zne_mitigation.py module (not hand-copied). calcola_aspettazione_hamiltoniana
is used only as a cross-check print, never as ground truth.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import zne_mitigation as zm

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

N_Q = zm.N_Q
T_HOPPING = zm.t_hopping
NUM_SHOTS = zm.NUM_SHOTS


def _pauli(op, q, n_qubits):
    I2 = sp.identity(2, format="csr", dtype=np.complex128)
    mats = [I2] * n_qubits
    mats[q] = sp.csr_matrix(op, dtype=np.complex128)
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


def build_hopping_hamiltonian_sparse(n_qubits, t_hopping):
    """-(t/2) * sum_q (X_q X_{q+1} + Y_q Y_{q+1}) over a periodic chain,
    built directly as a sparse Pauli-string sum -- independent of
    zne_mitigation.calcola_aspettazione_hamiltoniana's bit-twiddling trick."""
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    dim = 1 << n_qubits
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for q in range(n_qubits):
        q_next = (q + 1) % n_qubits
        XX = _pauli(X, q, n_qubits) @ _pauli(X, q_next, n_qubits)
        YY = _pauli(Y, q, n_qubits) @ _pauli(Y, q_next, n_qubits)
        H = H + (XX + YY)
    return -(t_hopping / 2.0) * H


def closed_form_energy(k, t_hopping, n_qubits=N_Q):
    """Hand derivation: within the single-excitation sector, each periodic
    bond term -(t/2)(X_qX_{q+1}+Y_qY_{q+1}) acts on the two-site basis
    {|10>,|01>} as -t*(swap) (since (XX+YY)/2 = 2*(sigma+ sigma- + h.c.)
    restricted to that subspace has off-diagonal magnitude 2, so the -(t/2)
    prefactor gives off-diagonal magnitude -t). Summed over all periodic
    bonds this is the standard nearest-neighbor hopping matrix H|q> =
    -t|(q+1) mod N> - t|(q-1) mod N>.

    generate_bloch_state(k) sets amplitude e^{ikq} at site q for q=0..N-1
    with NO wraparound consistency requirement on the phase itself (it is
    only a genuine eigenstate of H when e^{ikN}=1, i.e. k = 2*pi*n/N).
    Expanding <psi(k)|H|psi(k)> and separating the "bulk" hopping terms
    (which telescope into (N-1)*cos(k)) from the single wraparound bond
    that connects site N-1 back to site 0 (whose relative phase is
    e^{ik(N-1)}, NOT e^{-ik} as it would be for an infinite chain) gives
    the EXACT general closed form, valid for every real k:

        <psi(k)|H|psi(k)> = -(2t/N) * [(N-1)*cos(k) + cos((N-1)*k)]

    At an allowed eigenmomentum k = 2*pi*n/N, cos((N-1)*k) = cos(kN - k) =
    cos(k) (since kN is a multiple of 2*pi), and this collapses to the
    textbook single-particle tight-binding band -2*t*cos(k). Away from
    those N discrete points the extra cos((N-1)*k) boundary term does NOT
    vanish and the textbook band formula is simply wrong for this trial
    state -- it only coincides with -2*t*cos(k) at the N eigenmomenta."""
    N = n_qubits
    return -(2.0 * t_hopping / N) * ((N - 1) * np.cos(k) + np.cos((N - 1) * k))


def eigenmomentum_band_energy(k, t_hopping):
    """Textbook single-particle tight-binding band -2*t*cos(k), exact ONLY
    at the chain's actual allowed eigenmomenta k = 2*pi*n/N; kept here only
    to show explicitly where it agrees/disagrees with closed_form_energy."""
    return -2.0 * t_hopping * np.cos(k)


def verify_ideal_energy(k, n_qubits, t_hopping, H_sparse):
    psi = zm.generate_bloch_state(k)
    E_sparse = float(np.real(np.conj(psi) @ (H_sparse @ psi)))
    E_closed = closed_form_energy(k, t_hopping)
    E_script = zm.calcola_aspettazione_hamiltoniana(psi)
    return E_sparse, E_closed, E_script


def noise_scaling_curve(k, noise_scales, n_repeats, base_seed_root):
    rows = []
    for ns_idx, ns in enumerate(noise_scales):
        vals = []
        for r in range(n_repeats):
            seed = base_seed_root + ns_idx * 1_000_003 + r * 7919
            vals.append(zm.measure_energy_with_shots(k, noise_scale=ns, base_seed=seed))
        vals = np.array(vals)
        std = vals.std(ddof=1) if n_repeats > 1 else 0.0
        sem = std / np.sqrt(n_repeats) if n_repeats > 1 else 0.0
        rows.append((ns, vals.mean(), std, sem))
    return pd.DataFrame(rows, columns=["noise_scale", "E_mean", "E_std", "E_sem"])


def richardson_2pt(curve_df):
    e1 = curve_df.loc[np.isclose(curve_df["noise_scale"], 1.0), "E_mean"].iloc[0]
    e2 = curve_df.loc[np.isclose(curve_df["noise_scale"], 2.0), "E_mean"].iloc[0]
    sem1 = curve_df.loc[np.isclose(curve_df["noise_scale"], 1.0), "E_sem"].iloc[0]
    sem2 = curve_df.loc[np.isclose(curve_df["noise_scale"], 2.0), "E_sem"].iloc[0]
    E_2pt = 2.0 * e1 - e2
    sem_2pt = np.sqrt((2 * sem1) ** 2 + sem2 ** 2)
    return E_2pt, sem_2pt


def quadratic_fit_at_zero(curve_df, max_scale=2.0):
    sub = curve_df[curve_df["noise_scale"] <= max_scale + 1e-9]
    coeffs = np.polyfit(sub["noise_scale"], sub["E_mean"], deg=2)
    return float(np.polyval(coeffs, 0.0)), coeffs


def main():
    H_sparse = build_hopping_hamiltonian_sparse(N_Q, T_HOPPING)

    k_values = {"k=0 (eigenmomentum)": 0.0,
                "k=pi/3 (eigenmomentum)": np.pi / 3,
                "k=1.0 (generic, non-eigenmomentum)": 1.0}
    k_tags = {"k=0 (eigenmomentum)": "k0",
              "k=pi/3 (eigenmomentum)": "kpi3",
              "k=1.0 (generic, non-eigenmomentum)": "k1generic"}

    noise_scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_repeats = 30

    summary_rows = []
    all_curves = {}

    for label, k in k_values.items():
        E_sparse, E_closed, E_script = verify_ideal_energy(k, N_Q, T_HOPPING, H_sparse)
        E_band = eigenmomentum_band_energy(k, T_HOPPING)
        print(f"\n=== {label} (k={k:.6f}) ===")
        print(f"Sparse-Hamiltonian <psi|H|psi>:      {E_sparse:+.10f}")
        print(f"Closed-form (general, w/ boundary):  {E_closed:+.10f}")
        print(f"Textbook band -2t*cos(k) (only exact at eigenmomenta): {E_band:+.10f}")
        print(f"Script's own calcola_aspettazione_hamiltoniana (cross-check only): {E_script:+.10f}")
        print(f"sparse vs general closed-form agreement: {abs(E_sparse - E_closed):.2e}")
        print(f"sparse vs textbook band agreement:       {abs(E_sparse - E_band):.2e}")

        curve = noise_scaling_curve(k, noise_scales, n_repeats, base_seed_root=int(abs(k) * 1_000_000) + 1)
        all_curves[label] = curve
        print(curve.to_string(index=False))

        E_2pt, sem_2pt = richardson_2pt(curve)
        E_quad, quad_coeffs = quadratic_fit_at_zero(curve)
        true_E = E_closed
        residual_2pt = E_2pt - true_E
        residual_quad = E_quad - true_E

        print(f"True E(0) (closed form):      {true_E:+.6f}")
        print(f"2-point Richardson (lam=1,2, n_repeats={n_repeats}): {E_2pt:+.6f} +/- {sem_2pt:.6f} (SEM)  | residual = {residual_2pt:+.6f} ({abs(residual_2pt) / sem_2pt:.1f} sigma from true value)")
        print(f"Quadratic fit -> lam=0:       {E_quad:+.6f}  | residual = {residual_quad:+.6f}")
        print(f"Quadratic coefficient (curvature term): {quad_coeffs[0]:+.6f}")

        summary_rows.append({
            "k_label": label,
            "k": k,
            "E_true_closed_form": true_E,
            "E_true_sparse": E_sparse,
            "E_richardson_2pt": E_2pt,
            "sem_richardson_2pt": sem_2pt,
            "residual_2pt": residual_2pt,
            "residual_2pt_in_sem": abs(residual_2pt) / sem_2pt,
            "E_quadratic_fit_at_0": E_quad,
            "residual_quadratic": residual_quad,
            "quadratic_curvature_coeff": quad_coeffs[0],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(_DATA_DIR / "zne_mitigation_verification_summary.csv", index=False)

    for label, curve in all_curves.items():
        curve.to_csv(_DATA_DIR / f"zne_mitigation_verification_curve_{k_tags[label]}.csv", index=False)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    colors = ["#00FFFF", "#FF3333", "#FFFF00"]

    for ax, (label, curve), color in zip(axes, all_curves.items(), colors):
        k = k_values[label]
        true_E = closed_form_energy(k, T_HOPPING)

        ax.errorbar(curve["noise_scale"], curve["E_mean"], yerr=curve["E_std"],
                    fmt="o-", color=color, linewidth=1.5, markersize=5, label="Measured E(noise_scale)")

        e1 = curve.loc[np.isclose(curve["noise_scale"], 1.0), "E_mean"].iloc[0]
        e2 = curve.loc[np.isclose(curve["noise_scale"], 2.0), "E_mean"].iloc[0]
        lam_line = np.array([0.0, 1.0, 2.0])
        e_line = np.array([2 * e1 - e2, e1, e2])
        ax.plot(lam_line, e_line, "--", color="#888888", linewidth=1.5, label="2-point linear (lam=1,2) extrapolation")

        _, quad_coeffs = quadratic_fit_at_zero(curve)
        lam_dense = np.linspace(0, 2, 100)
        ax.plot(lam_dense, np.polyval(quad_coeffs, lam_dense), ":", color="#FFFFFF", linewidth=1.5, label="Quadratic fit (0..2)")

        ax.axhline(true_E, color="#00FF00", linewidth=2, label=f"True E(0) = {true_E:.4f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("noise_scale")
        ax.grid(True, linestyle="--", alpha=0.2, color="#444444")

    axes[0].set_ylabel("Energy (eV)")
    axes[0].legend(loc="best", fontsize=7)
    plt.suptitle("zne_mitigation.py: noise-scaling curve linearity check", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "zne_mitigation_verification.png", dpi=300)

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
