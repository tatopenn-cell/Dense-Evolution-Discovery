# QM/MM region partitioning + MMFF94 mechanical correction

The isodesmic-scission-energy feature (see
[Isodesmic bond-scission energy](isodesmic_bond_scission_energy.md)) hits
a real scaling wall on competition-scale molecules: STO-3G on a
20-heavy-atom molecule needs ~140 basis functions (~5.7GB dense ERI
tensor, feasible but a weak signal); 6-31G* needs ~255 basis functions
(~67GB, infeasible on a standard CPU kernel). A real QM/MM lets the QM
Hamiltonian stay small -- only the reactive region -- regardless of total
molecule size.

Real precedent (already indexed in quantumrag, no new paper needed): Li
et al. 2024, "hybrid quantum pipeline for drug discovery," uses a
5-heavy-atom QM region around a reactive covalent bond
(Sotorasib-KRAS(G12C)), the rest treated classically.

## Steps 1-2: partitioning + link-atom capping

```python
qm_frag, qm_atoms, boundary_bonds = partition_qm_mm(smiles, (o_idx, c_idx), qm_radius=2)
```

`o_idx`/`c_idx` are the two atoms of the reactive bond. `partition_qm_mm`
walks outward from them in heavy-atom hops up to `qm_radius`, then caps
the single bond it crosses with a hydrogen, returning a smaller,
already-valid molecule (`qm_frag`) in place of the whole one.

BFS from the reactive bond in **heavy-atom hops only** -- hydrogens
always follow their own heavy atom's region, never independently
BFS-expanded. A real bug was found and fixed building this: letting
hydrogens expand independently spuriously cuts a QM boundary atom's own
terminal C-H bonds, leaving the MM fragment empty (just an orphaned
capped hydrogen). One boundary bond is capped with H per the same
dummy-atom-to-hydrogen patch already used for isodesmic-scission
fragments.

## Step 3: small QM Hamiltonian only, tested for convergence with radius

1-hexanol's O-C1 isodesmic scission energy, QM region only (radius in
heavy-atom hops), vs. the whole-molecule reference (-16.14 kcal/mol):

| radius | heavy atoms | QM-only energy | diff from whole |
|---|---|---|---|
| 1 | 3 | -14.27 kcal/mol | 1.87 kcal/mol |
| 2 | 4 | -14.07 kcal/mol | 2.07 kcal/mol |
| 3 | 5 | -14.30 kcal/mol | 1.84 kcal/mol |

Honest finding: pure truncation+capping plateaus around a 1.8-2.1
kcal/mol gap and does **not** converge further with radius in this
range -- a small QM region gets ~87% of the answer immediately, but the
residual isn't closed by simply adding more atoms.

## Step 4: ONIOM-style MMFF94 mechanical correction

`correction = dE_MMFF94(whole-molecule reaction) - dE_MMFF94(QM-region-only reaction)`,
added on top of the small-region QM energy:

| radius | corrected diff | uncorrected diff |
|---|---|---|
| 1 | 0.69 kcal/mol | 1.87 |
| 2 | 0.98 kcal/mol | 2.07 |
| 3 | **0.48 kcal/mol** | 1.84 |

The correction roughly halves the error at every radius, and -- unlike
the uncorrected version -- improves with region size (best at radius 3).

**Retracted, not a real fix.** This result did not survive a later,
more careful geometry treatment. The QM energies above come from
independently re-embedding each fragment (whole, region, o-frag, c-frag)
with its own fresh RDKit conformer -- a real geometric inconsistency the
follow-up electrostatic-embedding kernel fixed by slicing every piece
from ONE shared whole-molecule conformer instead. Re-running the exact
same correction arithmetic under that better geometry: 1-hexanol's
uncorrected error was already under 0.2 kcal/mol at every radius (not
1.8-2.1), and applying the MMFF94 correction made it **worse** every
time (down to -1.5 kcal/mol). The same held on a second, branched-
aromatic molecule. The correction was compensating for the independent-
re-embedding artifact above, not for truncation itself -- it does not
generalize and is not used in `scripts/qmmm/`.

## Details

Step 5 (electrostatic embedding) was tried separately and is documented
in [QM/MM bond order and electrostatic embedding](qmmm_bond_order_and_embedding.md)
-- five different fixes tried, all worse than no embedding, still open.

**Status**: partitioning + capping (steps 1-2) are the part that holds
up, consolidated into a real reusable module, `scripts/qmmm/`
(`partition_qm_mm_region`, `sliced_geometry`) -- ring-safe (a boundary
bond into an aromatic ring now pulls the whole ring in, fixing a real
`AtomKekulizeException` crash found on a branched-aromatic molecule) and
validated with a shared conformer, no MMFF94 correction. Plain truncation
alone is accurate to <0.2 kcal/mol on every radius tested on two
different molecules -- promotion to Dense-Evolution not yet proposed
(this is Discovery research code, RDKit is not a Dense-Evolution
dependency), but this is the validated building block for any future
QM/MM experiment in this repo.

**Scripts**: `scripts/qmmm_region_partitioning_mmff_correction.py`
(original, retracted correction), `scripts/qmmm/` (the reusable
utility that replaces it).
