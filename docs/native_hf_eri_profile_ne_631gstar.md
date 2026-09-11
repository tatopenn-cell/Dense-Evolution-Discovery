# ERI cost profile: Ne/6-31G* (P10)

Measurement only, per prog.txt's Prompt 12 -- no optimization code written or proposed as a change in this pass. Script: `scripts/native_hf_eri_profile_ne_631gstar.py`, run against `dense_evolution.native_hf.assembly.build_repulsion_tensor`'s real code path (same functions, same loop structure, instrumented from the outside).

Machine/run note: this run measured **172.1s** total (54.4s Schwarz bounds + 117.7s main quartet loop), not the 623s figure from an earlier session -- different machine, same qualitative structure, which is what this profile is actually answering.

## Setup

Ne, 6-31G*: 6 shells, 15 AOs.

| shell | degree | primitives |
|---|---|---|
| 0 | s | 6 |
| 1 | s | 3 |
| 2 | p | 3 |
| 3 | s | 1 |
| 4 | p | 1 |
| 5 | d | 1 |

## 1. Quartet counts

- After 8-way shell-index symmetry: **231** quartets.
- After Schwarz screening (`screening_tol=1e-12`): **231** quartets -- screening removed **zero**. For a single atom with no spatial separation between shells, Schwarz bounds never drop low enough to trigger it at this tolerance.

## 2/3. Warm (post-compile) execution cost

Grouped by sorted degree-signature, quartet counts range from 1 (dddd) to 39 (ppss). Per-quartet warm cost scales steeply with degree: `ssss` is 1.5ms, `dpdp` is 363ms, `ddpp` is 720ms. Summed as (count x warm-average) per exact ordered tuple: **12.6s**, covering only **10.7%** of the 117.7s main loop.

## 4. Compilation cost

**35 distinct exact (a,b,c,d)-ordered degree-tuples** get compiled (the JIT cache key is the literal ordered tuple, not the sorted multiset -- `dpsp` and `spdp`, e.g., compile separately even though the underlying physics is the same integral under relabeling). Sum of all first-call (compile+first-exec) times: **107.9s = 91.7%** of the 117.7s main loop. Some individual compiles are very expensive: `dddp` 6.85s, `ddpp` 4.42s, `dppp` 4.07s, `dsdp` 3.61s; the single `dddd` quartet (only one in the whole molecule) took **29.3s** for its one and only call.

## 5. Everything else (dispatch, `np.array()`, sync)

Compile time (107.9s) + count x warm estimate (12.6s) = 120.5s against a measured 117.7s main-loop wall time -- the two overlap slightly (each first call is counted once at full compiled cost and once inside its tuple's per-call count), so the true dispatch/other overhead is within noise of **zero**, not a separate cost worth chasing.

## Answer to the audit's diagnostic question

**Compilation, not execution or dispatch, is the dominant cost** -- 92% of the main loop's wall time is XLA tracing/compiling 35 distinct programs, only 11% is the actual numerical work, and dispatch overhead is negligible.

This is consistent with, and explains, both previously-rejected attempts:
- **Padding every quartet to the global max degree** (1049.5s vs 623s, regression): collapses compilation to one signature, but forces every quartet -- including the 21 cheap `ssss` ones -- to pay the `dddd`-tier *execution* cost (29s for a single call at this degree). Fixing the dominant cost (compilation) by making the dominated one (execution) universal was never going to win.
- **Batching by shell signature** (Si2, 3.98s vs 2.26s, regression): adds Python/JAX bookkeeping to reduce *dispatch* overhead, which this profile shows was never the bottleneck to begin with.

Per prog.txt's own framing, the indicated next step is reducing the number of distinct compiled signatures -- e.g. canonicalizing each quartet's degree order before calling `_quartet_block` (the same 8-fold permutation trick `build_repulsion_tensor` already applies at the shell-index level for storing results could plausibly extend to the compile-signature level too) -- but that is a change to propose and test in its own session, not part of this measurement.
