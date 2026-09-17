# QM/MM: electrostatic embedding and bond-order-weighted partitioning

Two follow-up experiments to
[QM/MM region partitioning](qmmm_region_partitioning_mmff_correction.md)'s
steps 1-4, each with a real, honest result -- one still open, one
reversed from negative to positive on a second test.

## Step 5: electrostatic embedding (open, not solved)

```python
V_ext = external_point_charge_potential(atomic_numbers, geometry_bohr, basis, mm_charges, mm_positions)
result = run_scf(S, H_core + V_ext, repulsion, n_electrons, nuclear_charges, geometry_bohr)
```

`mm_charges`/`mm_positions` are the truncated-away atoms' real MMFF94
partial charges and positions. Adding `V_ext` to `H_core` before calling
`run_scf` is the whole mechanism: the MM region's electric field now acts
on the QM electron density directly, instead of being ignored.

**First test (1-hexanol)**: plain vs. embedded gave identical numbers at
every radius. Not a bug -- MMFF94 assigns real charge only to the polar
O-C1-H(O) group, always inside the QM region by construction, so the
truncated alkyl tail has exactly zero charge to embed.

**Second test (5-amino-1-pentanol, a second polar group far from the
reactive bond)**: now there's real charge to embed (N ≈ -0.99), and
embedding measurably changes the result -- but makes it **worse**:

| radius | plain | embedded |
|---|---|---|
| 1 | 0.11 kcal/mol | 0.27 kcal/mol |
| 2 | 0.08 kcal/mol | 0.26 kcal/mol |
| 3 | 0.02 kcal/mol | 0.09 kcal/mol |

Plausible cause: the textbook QM/MM near-boundary overpolarization
artifact -- the MM point charge immediately across the cut bond sits too
close to the QM density.

**Charge-shifting attempt**: redistributing that one boundary atom's
charge onto its own remaining MM neighbors (the standard textbook fix)
did **not** close the gap -- at radius 3 (the only radius where the
boundary atom had nonzero charge to shift), the corrected result (0.21
kcal/mol) was worse than both plain embedding (0.09) and plain truncation
(0.02). Likely cause: the redistribution target (the boundary atom's own
hydrogens) can be just as close to the boundary as the atom being zeroed.
**Left open** -- a correct fix needs a more careful redistribution scheme
than this first attempt.

![Embedding helps only with real MM charge, but overpolarizes; charge-shifting doesn't fix it](assets/qmmm_bond_order_and_embedding/qmmm_embedding_charge_shifting.png)

## Bond-order-weighted region partitioning: reversed by a second test

Instead of a fixed hop-count radius, propagate a relevance score
outward from the reactive bond, weighted at each step by the real
Mayer/Wiberg bond order between the two atoms:

```python
bond_order = real_mayer_bond_order(density_matrix, overlap_matrix, ao_atom_map)
relevance = propagate(bond_order, seeds=(o_idx, c_idx), decay=0.3)
qm_atoms = [i for i in range(n_atoms) if relevance[i] > threshold]
```

`bond_order` comes from an actual HF density matrix (`D = 2P`, from
`run_scf`'s own `density_matrix`) -- no diffusion model exists for
molecules in this codebase, so real chemistry substitutes for a learned
affinity (inspired by Diffuse2Seg, arXiv:2609.06491, Hümmer et al. 2026,
which propagates through a diffusion model's own attention weights
instead). A stronger bond lets relevance travel further before `decay`
shrinks it below `threshold`.

**First test (5-amino-1-pentanol, aliphatic chain): negative.** Every
backbone bond has similar bond order (~0.97-0.99) -- no heterogeneity to
exploit. The method's regions differ from fixed-radius BFS but aren't
demonstrably better, BFS is itself already isomorphism-invariant (so
that property isn't a differentiator), and the method costs strictly
more (a real whole-molecule HF calculation just to get a density matrix,
where BFS is free and instant).

**Second test (1-phenyl-pentan-1-ol-like branching, `OCC(c1ccccc1)CCC`):
positive.** A branching carbon with two paths at equal hop-distance --
one into an aromatic ring, one into a saturated propyl chain. Real bond
order confirms the aromatic ring's internal bond (1.412) is genuinely
stronger than the alkyl chain's (0.991). At the same hop distance (3
hops), the aromatic branch's propagated relevance is 40-70% higher than
the alkyl branch's (e.g. decay=0.7: 0.551 vs. 0.322 relevance) -- a real
difference fixed-radius BFS structurally cannot represent, since it
always includes or excludes both branches together at a given radius.

![Real bond order differentiates the aromatic and alkyl branches; fixed-radius BFS cannot](assets/qmmm_bond_order_and_embedding/qmmm_bond_order_aromatic_vs_alkyl.png)

**Conclusion**: the method has no advantage on uniform aliphatic chains,
but a real, demonstrated one on molecules with real bond-order
heterogeneity near the reactive site -- aromatic/conjugated groups are
common in real drug-like molecules, so this is the realistic case, not
an edge case. Worth a second look for promotion once it's validated on
an actual reaction-energy comparison (this test only validated the
partitioning signal itself, not yet the resulting isodesmic energy).

## Details

Bond orders were checked for chemical sanity throughout (O-H ~0.95, C-C
~1.0, non-bonded pairs ~0.01), and the propagation was verified
isomorphism-invariant -- not just assumed from its structure -- by
randomly relabeling every atom and checking the selected region maps
back to the same atoms regardless.

**Status**: neither promoted to Dense-Evolution yet. Step 5 remains
genuinely unsolved. The bond-order partitioning idea has a real positive
signal (above) but still needs a full reaction-energy comparison against
steps 1-4's MMFF94-corrected result, at a matched atom budget, before
promotion is warranted.

**Scripts**: `scripts/qmmm_electrostatic_embedding_charge_shifting.py`
(step 5), `scripts/qmmm_bond_order_partition_vs_radius.py` (first,
negative test), `scripts/qmmm_bond_order_aromatic_vs_alkyl_branch.py`
(second, positive test).
