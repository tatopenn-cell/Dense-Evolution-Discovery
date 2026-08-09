# Density-Matrix ZNE Validation

Run: `python scripts/sophia_reflection.py`, seed=0, 2-qubit Bell state,
16-point depolarizing-noise sweep (base_p 0.02 → 0.5), K=200 trajectories
per noise scale, 3-point Richardson ZNE + Smolin-Gambetta-Smith projection
(`dense_evolution.mitigation.zne_density_matrix`). Raw data:
[`data/sophia_reflection.csv`](data/sophia_reflection.csv), plot:
[`images/sophia_reflection.png`](images/sophia_reflection.png).

## Part 1: the measured trajectory

| base_p | raw fidelity | corrected fidelity | delta |
|---:|---:|---:|---:|
| 0.020 | 0.933750 | 0.959037 | +0.0253 |
| 0.052 | 0.871250 | 0.998395 | +0.1271 |
| 0.084 | 0.796250 | 0.850810 | +0.0546 |
| 0.116 | 0.703750 | 0.860179 | +0.1564 |
| 0.148 | 0.683750 | 0.992279 | +0.3085 |
| 0.180 | 0.615000 | 0.846814 | +0.2318 |
| 0.212 | 0.575000 | 0.941188 | +0.3662 |
| 0.244 | 0.538750 | 0.811397 | +0.2726 |
| 0.276 | 0.470000 | 0.707962 | +0.2380 |
| 0.308 | 0.407500 | 0.647910 | +0.2404 |
| 0.340 | 0.428750 | 0.681777 | +0.2530 |
| 0.372 | 0.352500 | 0.531778 | +0.1793 |
| 0.404 | 0.368750 | 0.587965 | +0.2192 |
| 0.436 | 0.313750 | 0.402500 | +0.0888 |
| 0.468 | 0.325000 | 0.446260 | +0.1213 |
| 0.500 | 0.303750 | 0.331181 | +0.0274 |

16/16 positive. Mean delta +0.1819. Range [+0.0253, +0.3662].

## Reading

Correction value is not monotonic in noise: it grows through the low-to-mid
range (peak delta +0.3662 at base_p=0.212), then tapers off past roughly
base_p=0.3 (down to +0.0274 by base_p=0.5). This matches the general
Richardson-extrapolation behavior already documented elsewhere in this
project and in dense-evolution's own changelogs: correction quality is
bounded by how clean the underlying estimate already is. At very low noise
there is little to recover; at very high noise the extrapolation itself
gets less reliable. The effective window is the middle of the sweep, and
this run shows where that window sits for this circuit, this channel, and
this K.

## Part 2: why the density-matrix extension, specifically

Part 1 shows the density-matrix ZNE extension improves fidelity, but that
alone doesn't say why the density-matrix approach matters over plain
scalar ZNE (Richardson-extrapolating a raw fidelity number directly,
already used elsewhere in this project). A direct, real comparison
answers that: `scripts/sophia_reflection.py`'s
`run_scalar_vs_density_matrix_comparison`, same Bell state, same 3 noise
scales, same K=200 trajectories, but on amplitude damping (a genuinely
non-Pauli channel, `NoiseModel`'s `amplitude_damping`) instead of
depolarizing -- chosen because its asymmetric decay is more likely to
push a naive scalar extrapolation past a physical bound than a
Pauli-mixture channel would. Raw data:
[`data/scalar_vs_density_matrix_zne.csv`](data/scalar_vs_density_matrix_zne.csv),
plot:
[`images/scalar_vs_density_matrix_zne.png`](images/scalar_vs_density_matrix_zne.png).

Two paths computed from the identical noisy density matrices at each
noise level:

- **Scalar ZNE**: compute the Uhlmann fidelity independently at each of
  the 3 noise scales, then `richardson_extrapolate` those 3 numbers
  directly. Nothing constrains the result to stay in `[0, 1]` -- a bare
  fidelity number carries no positive-semidefinite/trace-1 structure for
  any correction step to enforce.
- **Density-matrix ZNE**: extrapolate the density matrix itself
  (`zne_density_matrix`), which internally projects the result back onto
  the nearest true density matrix (`project_to_physical`'s
  Smolin-Gambetta-Smith projection) before fidelity is ever computed.

**16-point sweep (base_p 0.05 → 0.6), seed=10: 7/16 (44%) of the scalar
ZNE points land outside `[0, 1]` -- physically meaningless fidelity
values (e.g. 1.10 at base_p=0.160, 1.04 at base_p=0.343). The
density-matrix ZNE path never does: 0/16, guaranteed by construction, not
by luck.** Mean absolute divergence between the two paths across the
sweep: 0.0397. This is the concrete answer to "why the density-matrix
extension": scalar ZNE is a reasonable point estimate when the noise
happens to keep things well-behaved, but it has no mechanism to notice or
correct an unphysical result, while the density-matrix extension is
physical by construction whenever it returns an answer at all.
