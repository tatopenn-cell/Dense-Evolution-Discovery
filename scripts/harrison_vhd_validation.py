"""
Validates dense_evolution.harrison_tb (universal tight-binding parameters)
and dense_evolution.vhd_tb (Vogl-Hjalmarson-Dow material-specific
parameters) against real experimental band gaps for GaAs, Si, and Ge.

GaAs is a direct-gap material (minimum at Gamma); Si and Ge are
indirect-gap (minimum off-Gamma, along Gamma->X for Si, Gamma->L for Ge)
-- band_extrema_along_path scans the relevant line to find the true
valence-band max / conduction-band min instead of reading only Gamma.

Produces data/harrison_vhd_gap_comparison.csv and
images/harrison_vhd_gap_comparison.png. See docs/harrison_tight_binding.md
for the full write-up of these results.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dense_evolution.harrison_tb import zincblende_hamiltonian
from dense_evolution.vhd_tb import direct_gap_at_gamma, band_extrema_along_path

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
_DATA_DIR.mkdir(exist_ok=True)
_IMAGES_DIR.mkdir(exist_ok=True)

# name -> (lattice constant [Angstrom], experimental gap [eV], VHD scan direction)
_MATERIALS = {
    "GaAs": dict(a=5.6533, experimental_gap=1.42, gap_type="direct (Gamma)",
                 vhd_k_end=None),
    "Si":   dict(a=5.431,  experimental_gap=1.12, gap_type="indirect (Gamma->X)",
                 vhd_k_end=(1.0, 0.0, 0.0)),
    "Ge":   dict(a=5.658,  experimental_gap=0.66, gap_type="indirect (Gamma->L)",
                 vhd_k_end=(0.5, 0.5, 0.5)),
}


def harrison_gap(name: str, a_lattice: float, k_end) -> float:
    """Harrison universal-parameter gap: scan Gamma->k_end (or just Gamma
    for direct-gap GaAs) for VBM/CBM, same convention as vhd_tb's scan."""
    twopi_a = 2 * np.pi / a_lattice
    if k_end is None:
        H = zincblende_hamiltonian([0., 0., 0.], "Ga", "As", a_lattice)
        eig = np.sort(np.linalg.eigvalsh(H).real)
        return eig[4] - eig[3]

    ts = np.linspace(0.0, 1.0, 501)
    vbm, cbm = -np.inf, np.inf
    for t in ts:
        k = twopi_a * np.array(k_end) * t
        H = zincblende_hamiltonian(k, name, name, a_lattice)
        eig = np.sort(np.linalg.eigvalsh(H).real)
        vbm = max(vbm, eig[3])
        cbm = min(cbm, eig[4])
    return cbm - vbm


def vhd_gap(name: str, k_end) -> float:
    if k_end is None:
        return direct_gap_at_gamma(name)
    _, _, _, _, gap = band_extrema_along_path(name, (0., 0., 0.), k_end)
    return gap


def run_validation():
    rows = []
    for name, spec in _MATERIALS.items():
        h_gap = harrison_gap(name, spec["a"], spec["vhd_k_end"])
        v_gap = vhd_gap(name, spec["vhd_k_end"])
        exp = spec["experimental_gap"]
        rows.append({
            "material": name,
            "gap_type": spec["gap_type"],
            "harrison_universal_eV": h_gap,
            "vhd_material_specific_eV": v_gap,
            "experimental_eV": exp,
            "harrison_error_pct": 100 * (h_gap - exp) / exp,
            "vhd_error_pct": 100 * (v_gap - exp) / exp,
        })
        print(f"{name}: Harrison={h_gap:.4f} eV ({100*(h_gap-exp)/exp:+.1f}%), "
              f"VHD={v_gap:.4f} eV ({100*(v_gap-exp)/exp:+.1f}%), "
              f"experimental={exp:.2f} eV")

    df = pd.DataFrame(rows)
    df.to_csv(_DATA_DIR / "harrison_vhd_gap_comparison.csv", index=False)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["harrison_universal_eV"], width, color='#FF007F',
           label='Harrison (universal)')
    ax.bar(x, df["vhd_material_specific_eV"], width, color='#00FFFF',
           label='VHD (material-specific)')
    ax.bar(x + width, df["experimental_eV"], width, color='#FFFF00',
           label='Experimental')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({t})" for m, t in zip(df["material"], df["gap_type"])])
    ax.set_ylabel("Band gap (eV)", color='#888888')
    ax.set_title("Harrison Universal vs. VHD Material-Specific Tight-Binding Gaps",
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.2, color='#444444', axis='y')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(_IMAGES_DIR / "harrison_vhd_gap_comparison.png", dpi=300)

    print("============================================================")
    print("Data saved to data/harrison_vhd_gap_comparison.csv")
    print("Plot saved to images/harrison_vhd_gap_comparison.png")
    print("============================================================")
    return df


if __name__ == "__main__":
    run_validation()
