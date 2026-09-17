# Isodesmic bond-scission energy discriminates real isomers

Chemistry-informed candidate re-ranking feature for the Kaggle
`enveda-CASMI26-molecule-id-mass-spectra` competition, via
`dense_evolution.native_hf`. For a candidate SMILES, break a bond
H-capped on both sides (`whole + H2 -> fragment_A_H + fragment_B_H`) and
sum the resulting energy differences across up to 4 acyclic single bonds
into a per-molecule "stability fingerprint" -- isomers that pack their
atoms differently should break down differently.

## Basis-set choice, isolated-variable test

Three real C3H8O isomers (1-propanol, 2-propanol, methyl ethyl ether),
same scission code, only the basis changed:

| basis | spread across 3 isomers |
|---|---|
| STO-3G | 2.5 kcal/mol (does not discriminate -- within the method's own noise) |
| 6-31G* | 10.8 kcal/mol (discriminates: the ether separates clearly from both alcohols, the two alcohol regioisomers separate by ~4 kcal/mol from each other) |

## Real scaling bottleneck found and fixed

Getting the 6-31G* result required fixing a real memory bottleneck:
`native_hf`'s own JAX-jitted one-electron integral assembly ran out of
memory during XLA JIT compilation on a real Kaggle CPU kernel, even with
the (much more expensive) 4-index ERI tensor already routed through the
libcint bridge (~50000x faster than native_hf's own recursions). Fixed by
extending the same bridge to the one-electron integrals (S, H_core) --
promoted to Dense-Evolution as `build_overlap_and_core_hamiltonian_libcint`
(PR #281 there).

## Even-electron rule

Bond scission here always H-caps both fragments (closed-shell to
closed-shell), never producing an open-shell radical pair. This matches
real chemistry for this dataset: positive-mode ESI-CID mass spectrometry
fragmentation predominantly proceeds via heterolytic (closed-shell)
pathways, not homolytic radical ones -- so `native_hf`'s RHF-only solver
(no UHF/open-shell support) is the right tool here, not a limitation
being worked around.

## Status

Real signal confirmed at 6-31G*, but scale-limited: STO-3G is feasible on
a real ~20-heavy-atom competition molecule (~140 basis functions, ~5.7GB
dense ERI) but gives a weak/noisy signal; 6-31G* would need ~255 basis
functions (~67GB dense ERI tensor), infeasible on a standard CPU kernel.
This scaling wall is what motivated the QM/MM region-partitioning
experiments (see [QM/MM region partitioning](qmmm_region_partitioning_mmff_correction.md)).

Script: `scripts/native_hf_isodesmic_scission_energy.py`.
