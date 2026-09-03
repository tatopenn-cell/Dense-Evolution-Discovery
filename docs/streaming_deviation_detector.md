# A Zero-Latency Streaming Port of classify_segments' Causal Deviation Check

Real robot control loops run at 30-100Hz and can't wait for a batch array -- but reading
`arbiter.classify_segments`' own implementation line by line first (not assuming) found a
real constraint: its final spike-vs-regime label looks `radius` points AHEAD of where a
deviant run ends (the "persiste" check, arbiter.py lines 178-186) to decide whether the run
settles at a new level or reverts. That part cannot be zero-latency. Reconsidered what a
real robot safety loop actually needs: not "was that a spike or a regime" (a triage
question, answerable after the fact), but "is this point deviant right now" -- exactly
`classify_segments`' own per-point `deviante` computation (arbiter.py lines 113-132),
before the run-length logic. That half genuinely is a causal, zero-latency computation, and
this experiment ports only that half.

## Step 1. The port

```python
from streaming_deviation import StreamingDeviationDetector

det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
for x in stream:
    is_deviant = det.update(x)
```

Same causal window (`radius*ref_mult`), same robust median/MAD center-scale, same
degenerate-baseline rule as the batch function -- a plain `deque`-backed buffer
recomputing median/MAD each step (O(span)), not a two-heap O(log span) structure: for the
window sizes this project already uses everywhere (10-100 points), the simpler
implementation is both fast enough and much easier to verify bit-exact against the batch
version.

## Step 2. The real correctness bar: bit-exact match, not "looks similar"

Reimplemented the batch `deviante` computation directly (not imported from
`classify_segments`, so the check tests the exact per-point logic being ported, not the
function's later post-processing) and compared point-by-point against the streaming
version, fed one value at a time, on real data from two independent domains:

```
LeRobot episode 0 (joint 2 diff): n=303  exact_match=True  mismatches=0
LeRobot episode 5 (joint 2 diff): n=231  exact_match=True  mismatches=0
LeRobot episode 22 (joint 2 diff): n=237  exact_match=True  mismatches=0
LeRobot episode 37 (joint 2 diff): n=227  exact_match=True  mismatches=0
Agent telemetry A_normal: n=50  exact_match=True  mismatches=0
Agent telemetry B_transient: n=50  exact_match=True  mismatches=0
Agent telemetry C_persistent: n=50  exact_match=True  mismatches=0
Agent telemetry D_legit_switch: n=50  exact_match=True  mismatches=0
Synthetic noise + 3 injected outliers: n=200  exact_match=True  mismatches=0

ALL EXACT MATCH: True
```

Zero mismatches across 4 real LeRobot arm episodes, Dense-Armor's own real agent telemetry
(all 4 scenarios), and a synthetic edge case with injected outliers -- two independent real
domains, the same bar `velocity_gated_stable_mask` was promoted at.

## Step 3. Is it actually fast enough for a real control loop?

```python
# 100000-point stream, 100-point warmup, timed per-call
```

```
99900 updates in 5.3737s -> 53.79 microseconds/call -> max sustainable rate: 18591 Hz
```

~18.6 kHz sustainable, over 180x the 30-100Hz a real robot control loop runs at -- the
simple buffer-recomputation design (chosen over a more complex O(log span) structure) had
real headroom to spare, not a premature-optimization tradeoff that needed making.

## Honest scope

This ports the causal deviation flag only -- not the spike-vs-regime label, which stays a
batch/offline question by design, not an oversight. A real-time consumer gets "deviant now,
yes/no" immediately; deciding whether a deviant episode was a transient blip or a genuine
regime change still requires the batch function once enough of the series is available.

## Reproducing this

`scripts/dense_armor_streaming/streaming_deviation.py` (`StreamingDeviationDetector`) and
`scripts/dense_armor_streaming/validate_streaming_deviation.py` (the equivalence + timing
checks -- reuses already-cached LeRobot data and Dense-Armor's own frozen agent telemetry,
no new downloads).
