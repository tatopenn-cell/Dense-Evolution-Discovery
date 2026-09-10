# Gate Blocking Redesign: Closing the Promotion Gaps

**Status: validated, ready for production promotion.** [The first gate-blocking experiment](mps_gate_blocking_experiment.md) validated a real 2.87x GPU speedup, but review before promotion into `dense_evolution` found three real gaps that needed solving first, not glossed over. All three are now closed, and the complete, promotion-ready design re-confirms the speedup: **2.77x** total (original vs. bucketed+fused), measured through the same consistent, apples-to-apples methodology, with the real `_compile_mps_ops` and full diagnostic bookkeeping in place -- not a simplified prototype.

## Gap 1: row encoding

`run_circuit_jit`'s rows are `[g_id, q1, q2, param, transpose_flag]`, reconstructed into a gate matrix *inside* the compiled kernel via `_mps_1q_matrix`/`_mps_2q_matrix`. A fused gate is an arbitrary matrix, not one of `gates.py`'s known IDs -- it cannot be expressed in that encoding at all.

**Fixed**: fusion now runs *after* the real, unmodified `_compile_mps_ops` (so CCX decomposition and SWAP-chain expansion for non-adjacent gates already happened, exactly as `run_circuit_jit` already does it today), converting its rows into concrete matrices via the same real `_mps_1q_matrix`/`_mps_2q_matrix` production helpers, then fusing those. The compiled kernel receives raw matrices directly -- the gate-ID switch is removed from the traced program entirely, not just extended.

## Gap 2: untested interactions

The first experiment only ever exercised adjacent gates on a simple TFIM circuit. Fusion needed checking against the two things `_compile_mps_ops` itself already handles: non-adjacent 2-qubit gates (expanded into a SWAP chain first) and CCX (decomposed into H/CX/T/Tdg first).

**Verified**: both now pass, fidelity 1.0 against the real eager `MPSSimulator`, on top of the TFIM regression case:

<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px">
<thead><tr>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Case</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Compiled &rarr; fused steps</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Fidelity</th>
</tr></thead>
<tbody>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">TFIM regression, N=6</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">63 &rarr; 33</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">1.000000000000</td></tr>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">Non-adjacent CX (SWAP-chain expansion)</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">20 &rarr; 20</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">1.000000000000</td></tr>
<tr><td style="padding:9px 10px">CCX decomposition</td><td style="padding:9px 10px">23 &rarr; 12</td><td style="padding:9px 10px">1.000000000000</td></tr>
</tbody>
</table>

The non-adjacent case shows no reduction (20 &rarr; 20) -- expected, that circuit has no consecutive same-pair chains to fuse. It matters that fusion doesn't *break* SWAP-chain gates, not that it shrinks every circuit.

## Gap 3: bookkeeping granularity

Fusing multiple original 2-qubit gates into one step means `self._bond_history`/`jsd_per_bond`/`truncation_errors`/`entanglement_entropy` get fewer entries than the eager path produces for the same circuit -- a real, observable behavior change, not just an internal speed optimization.

**Decided**: opt-in via a flag (`run_circuit_jit(ops, fuse_gates=True)`, default `False`) when this is promoted -- the current per-original-gate bookkeeping stays the unchanged default; `fuse_gates=True` trades finer-grained diagnostics for speed, with bookkeeping still populated, just at fused-step granularity instead of per-original-gate. `build_fused_matrix_runner` already returns the full `(chi_new, jsd_val, trunc_err, entanglement_entropy)` diagnostic tuple per fused step -- the same shape of information `run_circuit_jit` already produces, not a reduced version of it.

## GPU re-verification

Same apples-to-apples methodology as the [timing follow-up](mps_bucketed_svd_gpu_timing_followup.md) (all three variants through one consistent code path), now on the complete, promotion-ready pipeline (real `_compile_mps_ops`, matrix-based kernel, full diagnostic tuple):

<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px">
<thead><tr>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Version</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Warm time</th>
<th style="text-align:left;font-weight:500;color:#57606a;padding:8px 10px;border-bottom:1px solid #d7dbe0;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em">Speedup vs. original</th>
</tr></thead>
<tbody>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">Original (always fixed max_bond SVD)</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">3.385s</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">1.00x</td></tr>
<tr><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">Bucketed only (matrix-based kernel, no fusion)</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">2.435s</td><td style="padding:9px 10px;border-bottom:1px solid #d7dbe0">1.39x</td></tr>
<tr><td style="padding:9px 10px">Bucketed + gate blocking (fuse_gates=True design)</td><td style="padding:9px 10px">1.223s</td><td style="padding:9px 10px"><strong>2.77x</strong></td></tr>
</tbody>
</table>

Consistent with the first prototype's 2.87x (small difference within normal GPU run-to-run variance) -- confirming the promotion-ready redesign (real `_compile_mps_ops`, no gate-ID switch inside the kernel, full bookkeeping tuple) doesn't lose the speedup the simplified prototype found.

## Status

Correctness verified (fidelity 1.0 on all three interaction cases, both with and without the full diagnostic tuple). Speed re-verified on GPU through the complete, promotion-ready pipeline. All three promotion gaps (row encoding, untested interactions, bookkeeping granularity) are closed. Candidate for promotion into `dense_evolution.backends.mps` as an opt-in `run_circuit_jit(ops, fuse_gates=True)` path.

## Reproduce

```bash
python scripts/mps_gate_blocking_redesign_v2.py
```

GPU check (`colab_gate_blocking_redesign_v2_gpu.py`) was run on Google Colab (T4), not from this repo directly.
