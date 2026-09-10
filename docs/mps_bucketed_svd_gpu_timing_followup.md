# Bucketed-SVD MPS on GPU: Correcting a Wrong Claim

**Status: resolved.** This corrects the GPU claim in [mps_bucketed_svd_optimization.md](mps_bucketed_svd_optimization.md) (2.74x faster) -- that number came from a standalone reimplementation of the bucketed dispatch, never from the real `MPSSimulator.run_circuit_jit` API it was promoted into (Dense-Evolution PR #226). The real API was not benchmarked on GPU before promotion. When it finally was, afterward, it first measured **slower than the pre-optimization baseline**, not faster -- traced below to a benchmark methodology bug, not a real regression. Once measured correctly, bucketing alone gives a real ~1.41x on GPU; [gate blocking](mps_gate_blocking_experiment.md) closes the rest of the gap to a genuine 2.87x, measured through one consistent methodology end to end.

## The numbers, in order

1. Standalone `_bucketed_runner` reimplementation, GPU (T4), N=50/STEPS=5/max_bond=64, complex128: **2.74x faster** than a matching standalone "always max_bond" reimplementation. This is the number that was published and used to justify promoting the change.
2. The real, promoted `MPSSimulator.run_circuit_jit`, same circuit, GPU: **8.89s warm**. The pre-optimization baseline (measured earlier in this session on the same circuit): 6.60s warm. The real, shipped change is **slower**, not faster, on this one measurement.
3. cuQuantum's `NetworkState` MPS, same circuit, GPU: 1.14s warm -- making cuQuantum look ~7.8x faster than Dense-Evolution 8.1.76, worse than the pre-optimization comparison (~1.7-2.9x).

## What's been ruled out

- **Python bookkeeping loop** (`run_circuit_jit`'s post-scan diagnostics-to-history conversion): 62ms of the 8.89s. Not the cause.
- **The extra `trunc_err`/entanglement-entropy computation** the real per-branch SVD step does (and the standalone benchmark didn't): tested both with and without, as an actual scan output (not just computed-and-discarded, which the first attempt at this check got wrong and had to be redone) -- 2.451s vs 2.449s. Not the cause.
- **Source drift**: `inspect.getsource` on the installed 8.1.76 package's `_build_mps_runner` was compared line-by-line against a hand-copied reimplementation using the same real private helpers (`_mps_1q_matrix`, `_mps_2q_matrix`, `_vectorized_chi_search_jax`, `_pad_gamma`, `_pad_lambda`, `_bucket_sizes`) -- byte-identical. The hand-copied version still measured 2.45s under the same conditions the real `sim._mps_runner` measured 8.6s in. The discrepancy is not (yet) explained by any source difference found so far.

## Root cause, confirmed

Not GPU session noise (a factory-reset re-test still showed the same slowdown, ruling that out). The real cause: `run_circuit_jit`'s "warm" timing was always measured on a **fresh `MPSSimulator` instance**, but `self._mps_runner` (the `@jax.jit`-compiled closure) is built lazily per instance and never shared across instances -- a fresh instance means a fresh, uncompiled closure, so "warm" was paying a full recompile every time, same as "cold". A second attempt (calling `run_circuit_jit` twice on the *same* instance) also failed, for a different reason: the second call's circuit ran on top of the first call's already-evolved state, compounding real entanglement growth instead of measuring steady state (10.2s -> 12.2s -> 42.2s across repeated calls). The fix: a **fresh `|0...0>` instance for each timing, with the already-compiled closure manually shared onto it** -- fresh state and a pre-compiled kernel together. That gives the real, stable, honest number: bucketing alone is ~1.41x faster on GPU (not the earlier flawed 2.74x, and not a regression either).

## Scripts

- `scripts/colab_gpu_mps_benchmark_v2.py` -- real `run_circuit_jit`, pinned to 8.1.76, circuit built via real QASM (8.89s warm, later found to be a flawed "fresh instance" measurement)
- `scripts/colab_cuquantum_mps_benchmark_v2.py` -- cuQuantum comparison (1.14s warm)
- `scripts/colab_gpu_mps_timing_breakdown.py` -- splits `run_circuit_jit` into prep/kernel/bookkeeping (kernel=8.6s dominates)
- `scripts/colab_gpu_diagnostics_cost_check.py` -- A/B test on the trunc_err/entropy hypothesis (ruled out)
- `scripts/colab_print_installed_mps_source.py` -- dumps the installed `_build_mps_runner` source for direct comparison (confirmed identical)
- `scripts/colab_gpu_mps_benchmark_v3_same_instance.py` -- same-instance repeated calls (also flawed: compounds entanglement instead of measuring steady state)
- `scripts/colab_gpu_mps_benchmark_v4_shared_compiled_runner.py` -- the version that got it right: fresh `|0...0>` instance + manually shared, already-compiled `self._mps_runner` (2.730s/2.737s warm, stable)
- `scripts/colab_gpu_mps_fair_comparison_old_vs_new.py` -- 8.1.75 (old, shipped) measured with the same correct methodology (3.746s/3.820s warm) -- the real baseline this experiment's ~1.37-1.40x GPU speedup is measured against
- `scripts/colab_full_chain_apples_to_apples.py` -- all three variants (original, bucketed-only, bucketed+gate-blocked) measured through one consistent internal-scan methodology, giving the independently-confirmed 1.41x (bucketing alone) and 2.87x (combined with [gate blocking](mps_gate_blocking_experiment.md)) used for the final target-vs-result comparison

## Lesson

The CPU speedup claim (68.80x-73.96x) was measured via the real `run_circuit_jit` API before promotion and has held up. The GPU claim was not -- it was measured on a reimplementation and assumed to transfer to the real API. It didn't, and the gap wasn't caught until after promotion and a PyPI release. Future promotions need the real public API benchmarked on every platform a speedup is claimed for, before promotion, not just the platform where it's most convenient to test.
