# Harrison / VHD Tight-Binding Validation

!!! note
    Source of truth for this content lives at
    [`reference/harrison_tight_binding/README.md`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/reference/harrison_tight_binding/README.md)
    in the repo -- this page mirrors it for the docs site. The implementation
    itself lives in the main library:
    [`harrison_tb.py`](https://tatopenn-cell.github.io/Dense-Evolution/api/harrison_tb/) /
    [`vhd_tb.py`](https://tatopenn-cell.github.io/Dense-Evolution/api/vhd_tb/).

Knowledge base for Walter A. Harrison's empirical tight-binding method,
collected here as reference/retrieval material for work in this repo
(GaAs/Si electronic-structure scripts under `scripts/`).

## Source

**Walter A. Harrison**, *Electronic Structure and the Properties of
Solids: The Physics of the Chemical Bond*. Originally published by
W. H. Freeman, 1980; reprinted by Dover Publications (Dover Books on
Physics), 1989, ISBN 0-486-66021-4.

**What's actually here:** this repo does not have access to the book's
full text (it's a copyrighted, non-open work) and none was reproduced.
What follows is the small set of *facts* from the book that are (a)
public/standard enough to appear identically across independent
secondary sources and (b) directly checkable by running the code --
transcribed and cross-checked against
[`jarvist/HarrisonSolidStateTable.jl`](https://github.com/jarvist/HarrisonSolidStateTable.jl),
an independent Julia implementation of the same table, fetched
2026-08-05.

## The method, in brief

Harrison's tight-binding model builds a solid's electronic Hamiltonian
from two ingredients only, both universal (materials-independent
functional form):

1. **Atomic term values** -- the free-atom s and p orbital energies
   (on-site Hamiltonian diagonal), tabulated per element.
2. **A universal bond-scaling law** for the off-diagonal (hopping)
   matrix elements:

   $$V_{ll'm} = \eta_{ll'm} \cdot \frac{\hbar^2}{m_e d^2}$$

   where $d$ is the bond length and the four dimensionless $\eta$
   coefficients are the *same for every element pair* -- only $d$ and
   the atomic term values change between materials.

$\hbar^2/m_e = 7.62\ \text{eV·Å}^2$.

## Universal eta coefficients (Harrison, Dover reprint)

| coefficient | value |
|---|---|
| $\eta_{ss\sigma}$ | -1.40 |
| $\eta_{sp\sigma}$ | +1.84 |
| $\eta_{pp\sigma}$ | +3.24 |
| $\eta_{pp\pi}$ | -0.81 |

## Validated results vs. experiment (run 2026-08-05)

| Material | Gap type | Harrison universal | VHD material-specific | Experimental |
|---|---|---|---|---|
| GaAs | direct, at $\Gamma$ | 2.906 eV (104.7% error) | 1.55 eV (9.2% error) | 1.42 eV |
| Si | indirect, $\Gamma \to X$ | 3.66 eV (227% error), CBM misplaced at $\Gamma$ | 1.171 eV (4.6% error) | 1.12 eV |
| Ge | indirect, $\Gamma \to L$ | 1.831 eV (177.5% error), CBM misplaced at $\Gamma$ | 0.765 eV (15.9% error), correctly finds L below X | 0.66 eV |

**Pattern across all three:** Harrison's universal parameters are
qualitatively sane (Hermitian, correct bonding/antibonding structure)
but consistently ~2-3x off quantitatively, and for the indirect-gap
materials (Si, Ge) it structurally cannot place the conduction-band
minimum correctly -- it always lands at $\Gamma$, missing the real
off-axis valley entirely. Vogl-Hjalmarson-Dow (1983) material-specific
sp3s* parameters -- fitted per material, at the cost of needing a
separate parameter row per material instead of one universal table --
get within 5-16% of experiment and correctly identify which valley
(X for Si, L for Ge) is the true minimum.

## Implementation

- `dense_evolution/harrison_tb.py`: `ELEMENTS`, `ETA`,
  `sp3_dimer_hamiltonian`, `zincblende_hamiltonian`.
- `dense_evolution/vhd_tb.py`: `MATERIALS`, `sp3s_star_hamiltonian`,
  `direct_gap_at_gamma`, `band_extrema_along_path`.

Both are numpy-only, zero dependency on PySCF/OpenFermion or any
external quantum-chemistry package.

## What this does NOT replace

Neither model replaces the DFT/SCF path already in this repo for GaAs
(see `scripts/vqe_tmi_material_design.py`'s DFT-derived hopping) when
first-principles accuracy is needed. Their value is as fast,
dependency-free estimates or starting points -- VHD close enough to
experiment to be useful quantitatively, Harrison useful mainly for
qualitative/order-of-magnitude checks or when no per-material fit
exists yet.

Full validation detail (eigenvalue tables, k-point scan data,
Hermiticity checks) is in
[`reference/harrison_tight_binding/README.md`](https://github.com/tatopenn-cell/Dense-Evolution-Ising-Tests/blob/main/reference/harrison_tight_binding/README.md).
