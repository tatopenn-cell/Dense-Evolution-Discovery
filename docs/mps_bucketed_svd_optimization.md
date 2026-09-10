# Bucketed-SVD MPS: Real Bond Dimension Instead of a Fixed Ceiling

**In plain terms**: `MPSSimulator.run_circuit_jit` always runs its SVD truncation at a fixed size (`max_bond`), even when the actual entangled bond dimension of the state is much smaller -- the shipped code pads every cut up to the ceiling before compiling, so every gate application costs as much as the worst case ever could, all the time. This experiment replaces that fixed-size SVD with a small set of candidate sizes ("buckets") selected at runtime via `jax.lax.switch`, using a provable upper bound on how fast entanglement can grow, so most gates run a genuinely smaller SVD instead of the padded worst case -- while staying inside a single JIT-compiled program, same as the shipped path.

## The inefficiency

`_build_mps_runner`'s 2-qubit branch builds `theta_mat = theta_new.reshape(max_bond*2, 2*max_bond)` -- since `max_bond*2 == 2*max_bond` literally, this is *always* a square `2·max_bond` matrix, and the full economy SVD always runs at that size regardless of the real bond dimension. The real/dynamic chi is used only to *mask* the result afterward, never to shrink the computation itself. The eager path (`apply_gate_2q`/`_svd_truncate`) already avoids this -- it uses the gamma tensors' own real shapes -- confirming this is a known one-compile-for-the-whole-circuit tradeoff, not an oversight.

## The fix: a provable bucket bound, dispatched via `jax.lax.switch`

A 2-qubit gate acting on a cut can increase that cut's Schmidt rank by at most a factor of `d²=4` (`d=2`, the local physical dimension) -- so `predicted_max_rank = min(chi_l_real*2, chi_r_real*2)` is a guaranteed upper bound, never an underestimate. Each gate picks the smallest power-of-two bucket (`2, 4, 8, ..., max_bond`) that covers this bound, and `jax.lax.switch` dispatches to a branch that runs the SVD at that bucket's *real* size -- the same JIT-compatible mechanism `dense_evolution` already uses elsewhere in `mps.py` for 1q/2q gate-ID dispatch, just applied to bucket size instead of gate identity.

## Correctness: verified before trusting a single result

A first pass showed apparent mismatches in complex64 -- root-caused to a bug in the *test*, not the method: the random key was folded cumulatively across loop iterations instead of freshly each time, so several "mismatches" were comparing different random tensors by accident. After fixing that, an expanded matrix of ~23 `(chi_l, chi_r)` combinations x 4 precision/budget configurations (complex64 and complex128) matched, with one genuine discovery along the way: the shipped padded approach picks up floating-point "ghost" singular values (~2.8e-8 in complex64) purely from zero-padding before the SVD, which its own sqrt-scaled JSD metric can amplify into a spurious budget-violation signal near the truncation threshold -- an existing numerical fragility in the shipped code that the bucketed approach, which never pads before the SVD, doesn't have.

Full multi-gate circuit correctness (not just single-gate) was then checked via `jax.lax.scan` against the real eager `MPSSimulator` path, using statevector fidelity:

<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px">
<thead><tr>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Config</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Fidelity (ref, bucketed)</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">max_bond_used</th>
</tr></thead>
<tbody>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">complex128, N=6, max_bond=16</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">0.999999999998</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">ref=4, bucketed=4</td></tr>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">complex128, N=8, max_bond=32</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">0.999999999998</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">ref=4, bucketed=4</td></tr>
<tr><td style="padding:9px 10px">complex64, N=6, max_bond=16</td><td style="padding:9px 10px">1.000000005974</td><td style="padding:9px 10px">ref=3, bucketed=3</td></tr>
</tbody>
</table>

## A second real bug, found before merging: the middle bond

The single-gate correctness matrix above slices an *already fully-contracted* `theta` tensor -- safe by construction, since the middle-bond sum already ran at full size before the slice. The actual circuit-level runner does something different: it slices `gamma`/`lambda` tensors to bucket size `B` *before* the einsum contracts the middle bond away. Those are not the same computation. Concretely: with outer bonds `chi_l=2, chi_r=2` and a middle bond `chi_m=16`, the outer-bonds-only bound formula picked `B=4`, silently discarding ~82% of the state's real norm from the contraction -- not an approximation, a correctness bug. Fixed by widening the bound to `max(chi_l_real, chi_r_real, chi_m_real, min(chi_l_real*2, chi_r_real*2))`, verified against the exact computation on 6 constructed asymmetric cases (norm preserved to machine precision in all of them) and re-confirmed on the full circuit-fidelity check (which went from 0.999999999998 to an exact 1.000000000000 on the complex128 configs once fixed -- the bug's impact on the specific TFIM circuit tested had been small but nonzero).

## Speed: real, verified on CPU and GPU, at very different margins

Same N=50, 5-step TFIM Trotter circuit used throughout this repo's MPS benchmarking, `max_bond=64`, warm (post-compile) timing:

<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px">
<thead><tr>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Platform</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Shipped (warm)</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Bucketed (warm)</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Speedup</th>
</tr></thead>
<tbody>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">CPU (Windows)</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">4.64-4.68s</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">0.06-0.07s</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">68.80x-73.96x</td></tr>
<tr><td style="padding:9px 10px">GPU (Colab T4)</td><td style="padding:9px 10px">6.60s</td><td style="padding:9px 10px">2.41s</td><td style="padding:9px 10px">2.74x</td></tr>
</tbody>
</table>

Both platforms agree with the shipped simulator physically: `z0` (⟨Z⟩ on qubit 0, from the single-qubit reduced density matrix) matches to `diff=2.88e-10` on both CPU and GPU, despite a small `max_bond_used` gap (8 vs 6) -- the same class of benign SVD-rounding-near-cutoff variance already documented in `dense_evolution_mps_benchmark.ipynb` (Windows vs. Linux giving different `max_bond_used` for the identical circuit).

## Why the GPU speedup is so much smaller than the CPU one

The obvious suspicion -- that XLA lowers `jax.lax.switch` into an unconditional "compute every branch, then select" on GPU, defeating the whole point -- was checked directly and **refuted**. Two independent checks on the exact same switch mechanism, with the branch index passed as a genuine traced runtime argument to a single compiled program (not a Python-level constant, which would let XLA dead-code-eliminate branches at compile time -- a mistake in the first version of this check that gave a meaningless result):

- Forcing the smallest bucket (B=2) vs. the largest (B=128) for a batch of independent SVDs: **1351.57x** timing ratio -- consistent with genuine per-branch dispatch, not uniform "compute everything" cost.
- The compiled, post-optimization HLO retains a real `conditional(...)` opcode (count: 1) -- direct structural evidence the branch survives as an actual runtime conditional.

So the switch dispatch itself works correctly on GPU. The much smaller real-circuit speedup is best explained by a more mundane cause: the runner executes 638 sequential steps inside one `jax.lax.scan`, each paying its own kernel-launch/dispatch overhead -- a cost GPUs pay proportionally more for than CPUs do, per small op, regardless of how well any single switch behaves. This wasn't profiled down to the last microsecond, so it's the best-supported explanation, not a proven one.

## Status

Both platforms clear the 2x average-speedup target (CPU 68.80x-73.96x, GPU 2.74x), correctness holds across the full tested matrix plus a genuine multi-gate circuit check, and the GPU speedup gap has a real, evidence-backed (not hand-waved) explanation. Promoted into `dense_evolution.backends.mps` directly (`run_circuit_jit`'s SVD path), see the main [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution) repo for the shipped version.

## Reproduce

```bash
python scripts/mps_bucketed_svd_correctness.py
python scripts/mps_bucketed_svd_circuit_integration.py
python scripts/mps_bucketed_svd_timing_cpu.py
```

GPU checks (`colab_bucketed_svd_gpu_benchmark.py`, `colab_bucketed_svd_hlo_check.py`) were run on Google Colab (T4), not from this repo directly, following the same Colab-verification pattern used for every other GPU claim in this repo.
