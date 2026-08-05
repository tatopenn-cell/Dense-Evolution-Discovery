# Harrison Solid State Table -- reference material

Knowledge base for Walter A. Harrison's empirical tight-binding method,
collected here as reference/retrieval material for work in this repo
(GaAs/Si electronic-structure scripts under `scripts/`).

## Source

**Walter A. Harrison**, *Electronic Structure and the Properties of
Solids: The Physics of the Chemical Bond*. Originally published by
W. H. Freeman, 1980; reprinted by Dover Publications (Dover Books on
Physics), 1989, ISBN 0-486-66021-4.

Note: the title string the user pasted alongside this request
("Manuale per i pionieri diretti a ovest del 1859") does not match this
book or any known edition of it -- it reads like a garbled/mismatched
listing blurb, not a real subtitle. Treat it as noise; the citation
above is the real book Harrison's tight-binding method comes from.

**What's actually in this folder:** this repo does not have access to
the book's full text (it's a copyrighted, non-open work) and none was
reproduced here. What follows is the small set of *facts* from the
book that are (a) public/standard enough to appear identically across
independent secondary sources and (b) directly checkable by running
the code -- transcribed and cross-checked against
[`jarvist/HarrisonSolidStateTable.jl`](https://github.com/jarvist/HarrisonSolidStateTable.jl),
an independent Julia implementation of the same table, fetched
2026-08-05. Where that repo's own comments flag values as unverified
("not well checked", d-block TODOs), those values are excluded here.

A tested Python implementation of the data below lives in the main
Dense-Evolution library: `dense_evolution/harrison_tb.py`
(`sp3_dimer_hamiltonian`, `sp3_bond_block`, `hopping_integral`,
`ELEMENTS`, `ETA`).

## The method, in brief

Harrison's tight-binding model builds a solid's electronic Hamiltonian
from two ingredients only, both universal (materials-independent
functional form):

1. **Atomic term values** -- the free-atom s and p orbital energies
   (on-site Hamiltonian diagonal), tabulated per element from
   Herman-Skillman-type free-atom calculations.
2. **A universal bond-scaling law** for the off-diagonal (hopping)
   matrix elements between neighboring atoms' orbitals:

   V_ll'm = eta_ll'm * hbar^2 / (m_e * d^2)

   where `d` is the bond length and the four dimensionless `eta`
   coefficients below are the *same for every element pair* -- only
   `d` and the atomic term values change between materials. This is
   what makes the method "universal": no fitting per material.

`hbar^2/m_e = 7.62 eV*Angstrom^2` (d in Angstrom, V in eV).

## Universal eta coefficients (Harrison, Dover reprint)

| coefficient   | value |
|---------------|-------|
| eta_ss_sigma  | -1.40 |
| eta_sp_sigma  | +1.84 |
| eta_pp_sigma  | +3.24 |
| eta_pp_pi     | -0.81 |

(d-block eta_sd/pd/dd values also appear in Harrison's book and in the
Julia repo, but are not transcribed here since this repo's use case --
Si/GaAs/Ge sp3 semiconductors -- doesn't need them.)

Shi & Papaconstantopoulos (Phys. Rev. B 70, 205101, 2004) revisited
these and published somewhat different sp values (eta_ssσ=-1.32,
eta_spσ=+1.42, eta_ppσ=+2.22, eta_ppπ=-0.63) plus improved d-block
values -- noted here for awareness, not used in `harrison_tb.py`,
which sticks to Harrison's original numbers as the canonical citation.

## Atomic term values (sp "simple atom" entries only)

Sign convention: table below gives actual orbital energies in eV
(negative, bound states). Harrison's book prints these as positive
magnitudes; `harrison_tb.ELEMENTS` stores the negative (physical) form.

| element | Z  | eps_s (eV) | eps_p (eV) |
|---------|----|-----------|-----------|
| Be      | 4  | -8.17     | -4.14     |
| B       | 5  | -12.54    | -6.64     |
| C       | 6  | -17.52    | -8.97     |
| N       | 7  | -23.04    | -11.47    |
| O       | 8  | -29.14    | -14.13    |
| Mg      | 12 | -6.86     | -2.99     |
| Si      | 14 | -13.55    | -6.52     |
| P       | 15 | -17.10    | -8.33     |
| S       | 16 | -20.80    | -10.27    |
| Cu      | 29 | -6.92     | -1.83     |
| Zn      | 30 | -8.40     | -3.38     |
| Ga      | 31 | -11.37    | -4.90     |
| Ge      | 32 | -14.38    | -6.36     |
| As      | 33 | -17.33    | -7.91     |
| Se      | 34 | -20.32    | -9.53     |
| Sn      | 50 | -12.50    | -5.94     |
| I       | 53 | -19.42    | -9.97     |
| Pb      | 82 | -12.07    | -5.77     |

Elements in Harrison's table without both an s and p value (H, He,
Li, N.B. some others) are omitted -- they're not usable for the sp3
construction below.

## Slater-Koster sp3 matrix elements

Standard Slater-Koster (1954) table for an (s, px, py, pz) basis and a
bond of direction cosines (l, m, n):

```
E(s,s)   = Vssσ
E(s,x)   = l * Vspσ         E(x,s) = -l * Vspσ
E(s,y)   = m * Vspσ         E(y,s) = -m * Vspσ
E(s,z)   = n * Vspσ         E(z,s) = -n * Vspσ
E(x,x)   = l^2 Vppσ + (1-l^2) Vppπ
E(y,y)   = m^2 Vppσ + (1-m^2) Vppπ
E(z,z)   = n^2 Vppσ + (1-n^2) Vppπ
E(x,y)   = l*m (Vppσ - Vppπ)   [= E(y,x)]
E(y,z)   = m*n (Vppσ - Vppπ)   [= E(z,y)]
E(x,z)   = l*n (Vppσ - Vppπ)   [= E(z,x)]
```

Implemented as `sp3_bond_block(l, m, n, d_angstrom)` in
`harrison_tb.py`.

## Validated sanity check (run 2026-08-05)

Two-atom sp3 dimer, `sp3_dimer_hamiltonian(a, b, d)`:

- **Si-Si** at d=2.35 A (diamond-Si nearest-neighbor bond length):
  eigenvalues [-16.626, -12.25, -9.847, -7.638, -7.638, -5.402,
  -5.402, -1.418] eV. Hermitian; bonding states pushed well below the
  atomic eps_p=-6.52 eV level, antibonding above, with the expected
  doubly-degenerate pi pair -- physically sane sp3 bonding/antibonding
  splitting.
- **Ga-As** at d=2.45 A (zinc-blende bond length): eigenvalues
  [-18.386, -12.344, -9.239, -8.228, -8.228, -4.582, -4.582, -1.541]
  eV. Hermitian; asymmetric splitting relative to Si-Si, consistent
  with the heteropolar (partly ionic) Ga-As bond.

These are single-bond cluster sanity checks, not periodic band
structure -- they confirm the Hamiltonian construction is correct, not
that it reproduces literature GaAs/Si band gaps.

## Periodic GaAs band gap vs. experiment (run 2026-08-05)

`zincblende_hamiltonian(k, cation, anion, lattice_constant_angstrom)`
builds the real periodic Bloch Hamiltonian (two-atom zinc-blende
basis, 4 nearest-neighbor bonds per atom, Bloch phase per bond).
Diagonalized at Gamma (k=0) for real GaAs (lattice constant a=5.6533
Angstrom, the standard experimental zinc-blende value):

- Gamma eigenvalues (eV): -22.069, -9.537 (x3), -6.631, -3.273 (x3)
- valence band max = -9.537 eV, conduction band min = -6.631 eV
- **computed direct gap = 2.906 eV**
- **experimental GaAs direct gap = 1.42 eV**

Off by roughly a factor of 2. This is Hermitian and structurally sane
(same qualitative bonding/antibonding picture as the dimer check) --
the gap error is not a code bug, it's a known, documented limitation
of Harrison's *universal* (same 4 eta coefficients for every element
pair, no per-material fitting, no d-orbitals) parameter set on
polar/ionic compound semiconductors like GaAs. Harrison's own book
frames the method as giving the right physics/order of magnitude, not
spectroscopic accuracy -- more accurate empirical tight-binding models
(e.g. Vogl-Hjalmarson-Dow sp3s*, 1983) fit parameters per material to
recover gaps to meV accuracy, at the cost of the "one universal table"
property that makes Harrison's version cheap and dependency-free.

X and L point eigenvalues (eV), for context on band ordering away from
Gamma:
- X: -19.351, -15.314, -13.435 (x2), -3.966, -2.880, 0.625 (x2)
- L: -20.202, -15.582, -11.442 (x2), -6.194, -1.368 (x2), 0.467

## Improved precision: Vogl-Hjalmarson-Dow (1983) material-specific parameters

Harrison's gap error above (2.91 vs 1.42 eV, ~105% high) comes from
using *universal* parameters -- same eta coefficients for every
element pair, no per-material fitting. The fix used in practice by
the tight-binding literature is a *material-specific* fitted
parameter set: P. Vogl, H. P. Hjalmarson, J. D. Dow, "A Semi-empirical
tight-binding theory of the electronic structure of semiconductors",
J. Phys. Chem. Solids 44 (5), 365-378 (1983) -- an sp3s* basis (5
orbitals/atom, including an extra fitted s* orbital with no literal
physical meaning, added purely to get the lowest conduction band
right). Parameter values transcribed from an independent open-source
implementation, github.com/rpmuller/TightBinding (`TB.py`, fetched
2026-08-05), which cites the same paper.

Implemented as `dense_evolution/vhd_tb.py`
(`sp3s_star_hamiltonian`, `direct_gap_at_gamma`, `MATERIALS` for 16
zinc-blende/diamond semiconductors: C, Si, Ge, Sn, SiC, AlP, AlAs,
AlSb, GaP, GaAs, GaSb, InP, InAs, InSb, ZnSe, ZnTe).

**Result (run 2026-08-05):** GaAs direct gap at Gamma = **1.55 eV**
vs. experimental **1.42 eV** -- **9.2% error**, vs. Harrison
universal's 104.7% error for the same material. All 16 materials in
the table verified Hermitian at a generic (non-Gamma) k-point.

Note: while porting the reference implementation, its own Hamiltonian
assembly (`Hac.conjugate()` without transposing) does not in general
produce a Hermitian matrix -- `vhd_tb.py` fixes this by assembling
with the proper conjugate *transpose* (`.conj().T`), verified
Hermitian for all 16 materials.

Tradeoff vs. Harrison: this is *not* one universal table anymore --
each material needs its own fitted row. It is still zero-dependency
(no PySCF/OpenFermion, numpy only) and far cheaper than DFT/SCF, just
no longer "the same 4 numbers for everything."

## What this does NOT replace

Harrison's model gives a fast, dependency-free *approximation* to
electronic structure -- it does not replace the DFT/SCF path already
in this repo for GaAs (see the project's SCF fixes), and the Gamma-gap
test above confirms it is **not** more accurate than this project's
own DFT-derived GaAs parameters (`T1_GAAS_DFT_EV` in
`vqe_tmi_material_design.py`) -- it's off by ~2x on the one number
that's directly checkable against experiment. Its value here is a
fast, zero-dependency (no PySCF/OpenFermion) qualitative estimate or
starting point, not a precision replacement.
