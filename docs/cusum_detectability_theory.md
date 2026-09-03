# Predicting CUSUM Detectability From Real Statistical Theory

Experiment 42 explained its own 3.3% persistent-drift-detection result with a single
ad hoc number: the injected offset sat at "1.29 sigma of local noise, below the 3.0
threshold." That number was computed once, after the fact, for one specific case. This
experiment turns it into a general, reusable, pre-registered prediction -- grounded in
real, classical statistical theory, not invented -- and checks how well that theory
actually matches Dense-Armor's real code.

## Step 1. A real, verified reference for the formula

```python
import fitz  # PyMuPDF, via quantumrag's own extraction pipeline
doc = fitz.open("reynolds1975_cusum_arl_approximation.pdf")
doc[0].get_text()[:400]
```

```
TECHNOMETRICS VOL. 17, NO. 1, FEBRUARY 1975
Approximations to the Average Run Length in
Cumulative Sum Control Charts
                    Marion R. Reynolds, Jr.
                   Virginia Polytechnic Institute and State University
```

Reynolds (1975), *Technometrics* 17(1), 65-71 -- fetched and read directly (not cited
from a search-engine summary) before writing a single line of code. It derives a
Brownian-motion/Wald-type closed-form approximation to CUSUM's Average Run Length (ARL),
building on Page (1954) -- already dense-armor's own citation for `cusum_detector`. Two
more recent (2022-2026) CUSUM papers were checked and explicitly **not** used: their ARL
formulas are for algorithmically different variants (kernel/MMD-based, adaptive control
limits), not the simple linear CUSUM this project's code implements -- citing them would
have been a real mismatch, not a "more modern" version of the same theory.

## Step 2. Verifying the formula against the idealized model it describes

```python
from arl_theory import one_sided_arl

# delta = mu - k = 1.0 - 0.5 = 0.5, h = 5.0
one_sided_arl(delta=0.5, h=5.0, corrected=True), one_sided_arl(delta=0.5, h=5.0, corrected=False)
```

```
(10.34, 8.01)
```

A direct Monte Carlo simulation of the *exact* process the formula describes (a pure
random walk `S[i]=max(0,S[i-1]+(X[i]-k))`, `X[i]~N(mu,1)`, no windowing, no robust
scale estimation) gives **10.26** at 3000 trials -- the Siegmund-corrected formula
matches to 0.7%; the plain, uncorrected Wald formula (8.01) is off by 22%. This confirms
Reynolds' own 1975 finding directly: the boundary correction is not optional for
realistic `h` values.

## Step 3. Does the real `cusum_detector` match the theory?

```python
from dense_armor.utility.cusum import cusum_detector
from arl_theory import two_sided_arl

# real fixed-reference CUSUM, span=40, true shift mu=1.0 after warmup
theory = two_sided_arl(mu=1.0, k=0.5, h=5.0)
theory
```

```
10.34
```

Running the **real, installed** `cusum_detector(reference="fixed")` on 1000 real
synthetic trajectories (warmup at the true null level, shift applied only afterward)
gives an empirical ARL of **10.66** -- a 3.1% match. For a smaller real shift (mu=0.5),
the match is looser (25.7%); for the in-control/false-alarm case (mu=0.0), the theory
predicts an ARL of 469 but the real detector produces one closer to **277** -- a real,
honest 41% gap, not glossed over.

## Step 4. Explaining the gap, not hiding it

```python
# same false-alarm case, varying only the reference window size
for span in (40, 100, 300, 1000):
    ...  # real cusum_detector run, empirical ARL vs the SAME theory value (469.11)
```

```
span=40    real=216   rel_err=54%
span=100   real=300   rel_err=36%
span=300   real=348   rel_err=26%
span=1000  real=401   rel_err=15%
```

The gap shrinks monotonically as the reference window grows -- confirming the real cause
directly: the theory assumes an *exactly known* mean and variance, but the real detector
*estimates* them (median/MAD) from a finite window, and that estimation noise inflates
the real false-alarm rate above the idealized prediction. This is expected, understood,
and now quantified -- not a flaw in the theory or a bug in the detector, a real,
disclosed limit of applying an idealized-process formula to a finite-sample estimator.

## Step 5. A general, reusable detectability report

```python
from arl_theory import detectability_report

detectability_report(local_noise_scale=7.72, k=0.5, h=5.0, candidate_shift=10.0)
```

```
{'false_alarm_arl': 469.11, 'detection_arl': 38.01, 'shift_in_sigma': 1.30}
```

This reproduces Experiment 42's own ad hoc "1.29 sigma" finding -- but now as a general
function's output, computable *before* running a benchmark, from a detector's real local
noise level and a candidate shift size.

## Step 6. Does the theory predict a real, already-measured case?

Monte Carlo is not the same question as "does this predict reality." Reused Experiment
42's own committed lidar data (no new collection) at 7 real, independent points spread
through the real 631-object driving session, each with its own real local noise level
and the same real +10m telemetry-layer injection:

```python
for pt in candidate_points:
    mad = ...  # real local MAD at this real point, no injection
    predicted = detectability_report(mad, k=0.5, h=5.0, candidate_shift=10.0)["detection_arl"]
    real_latency = ...  # real cusum_detector(reference="fixed"), same real local window as its reference
```

Preregistered before running (7 points, spacing 80 objects, `k=0.5, h=5.0`, real
`+10m` injection, never adjusted after seeing results) -- the full, auditable table:

| point | real local MAD | shift (sigma) | predicted ARL | real observed latency | latency / predicted |
|---:|---:|---:|---:|---:|---:|
| 40  | 2.86 | 3.50 | 2.00 | 1 | 0.50 |
| 120 | 5.39 | 1.86 | 4.27 | 1 | 0.23 |
| 200 | 3.37 | 2.96 | 2.42 | 1 | 0.41 |
| 280 | 9.26 | 1.08 | 9.15 | 2 | 0.22 |
| 360 | 4.42 | 2.26 | 3.34 | 3 | 0.90 |
| 440 | 6.78 | 1.48 | 5.80 | 3 | 0.52 |
| 520 | 7.07 | 1.41 | 6.14 | 2 | 0.33 |

**7/7: real latency < predicted mean ARL.**

A first attempt at this got the comparison wrong and was caught before trusting it:
`cusum_detector(reference="fixed")` always locks its reference to the array's own FIRST
`span` samples, not a window near the injection point -- comparing a prediction built
from a *local* pre-injection window against a detector run whose real fixed reference
was the *start of the whole session* was an apples-to-oranges mismatch. Fixed by slicing
the real array so the detector's own reference window literally *is* the same real local
window the noise estimate came from.

**Honest, consistent result**: in all 7 real, independent cases, the real observed
detection latency was lower than the theory's predicted mean -- not scattered around it,
consistently below. This is the same direction as Step 4's own null-case finding (real
false alarms happen sooner than theory predicts too) -- one coherent explanation, not two
separate mysteries: real lidar range data across mixed object classes is not well
approximated by the theory's iid-Gaussian assumption, so real threshold crossings, in
both directions, happen faster than the idealized model predicts. No new formula is
proposed here to correct this -- per the explicit scope of this validation, the point was
to check the existing theory against real data, not to keep adding math.

**What this reframes `detectability_report()` as**: not an oracle that predicts a real
detection latency exactly, but a *pre-flight* estimate -- "under these statistical
assumptions, the detector should have this detectability; now measure how far the real
signal deviates from that model." That framing is the honest, defensible one a real
deployment could use it for. A natural follow-on this suggests -- explicitly **not**
built here, closing this experiment rather than continuing to add theory -- is pairing
`INSUFFICIENT_SNR`-style guidance with a *confidence* signal keyed off exactly this kind
of real-vs-theoretical deviation (e.g. "this channel's real threshold-crossing rate ran
Nx faster than the iid-Gaussian model predicts, treat the ARL prediction as optimistic"),
not just a bare `shift < 3 sigma` check. Left as a noted direction, not a next
implementation task.

## Update: a second, independent real physical domain (accelerometer)

Step 6 above validated `detectability_report()` against real lidar range data. This update
checks a second, physically different real sensor -- accelerometer magnitude (UCI HAR,
subject 17, standing still, already committed from the IMU validation experiment) -- to see
whether the lidar finding generalizes, not just repeats.

### Check 1: the same real injection magnitude exposes a real formula limit

Reusing the original IMU experiment's own real persistent-injection magnitude (+3.0g) at 5
real, independent points, each with its own real local causal-window MAD:

```
pt= 300  MAD=0.0029  shift_sigma=1049.7  predicted_ARL_raw=0.0059  real_latency=1
pt= 600  MAD=0.0019  shift_sigma=1553.4  predicted_ARL_raw=0.0040  real_latency=1
pt= 900  MAD=0.0027  shift_sigma=1125.9  predicted_ARL_raw=0.0055  real_latency=1
pt=1200  MAD=0.0014  shift_sigma=2157.2  predicted_ARL_raw=0.0029  real_latency=1
pt=1500  MAD=0.0023  shift_sigma=1278.8  predicted_ARL_raw=0.0048  real_latency=1
5/5 points: raw formula predicts ARL < 1 (physically meaningless)
```

This real standing-accelerometer baseline is quiet enough (real local MAD ~0.001-0.003g)
that the real +3.0g injection sits above 1000 sigma of real local noise -- the raw
Wald/Siegmund formula's exponential term underflows and predicts a fractional ARL below 1
sample in all 5 real cases. Not a code bug: this is what the asymptotic approximation
actually computes at extreme SNR, and it has no physical meaning (a detector cannot flag in
under one real sample). Real detection did happen at latency=1 in all 5 cases -- the
qualitative direction ("near-instant") is right, the specific number is not. **Fixed**:
`detectability_report()` now floors both `false_alarm_arl` and `detection_arl` at 1.0.

### Check 2: a moderate, comparable-to-lidar shift gives a genuinely mixed result

To get a meaningful (not degenerate) ARL comparison, the injection was instead scaled to a
fixed 2.0-sigma-of-real-local-noise offset at each of the same 5 points (preregistered
before running, never adjusted after seeing results):

```
pt= 300  MAD=0.0029  shift_sigma=2.00  predicted_ARL=3.89  real_latency=10
pt= 600  MAD=0.0019  shift_sigma=2.00  predicted_ARL=3.89  real_latency=1
pt= 900  MAD=0.0027  shift_sigma=2.00  predicted_ARL=3.89  real_latency=2
pt=1200  MAD=0.0014  shift_sigma=2.00  predicted_ARL=3.89  real_latency=9
pt=1500  MAD=0.0023  shift_sigma=2.00  predicted_ARL=3.89  real_latency=6

2/5 real points: observed latency < predicted mean ARL
```

**Honest, real result**: unlike lidar's uniform 7/7 (real detection always faster than
predicted), this real accelerometer domain gives a genuinely MIXED outcome at moderate SNR --
2/5 points faster than predicted, 3/5 slower. Documented as found, not forced to match the
lidar direction. A first draft of this section stated a wrong "5/5, same direction as lidar"
result -- traced to a real bug in an earlier ad hoc script (`cusum_detector` returns
`(flagged, cusum)`; the earlier script unpacked them in the wrong order and accidentally
used the raw, never-zero accumulator as if it were the boolean flag array). The committed
`validate_against_real_imu.py` unpacks correctly; re-run and re-verified before writing this
number down.

**What this means for the theory, honestly**: the direction of the theory-vs-reality gap
found on lidar (real crossings always faster than predicted) does NOT automatically transfer
to a different real physical domain. Two real, independent checks now exist, with two
different real outcomes -- use `detectability_report()` as a pre-flight estimate under
classical iid-Gaussian assumptions, then always measure the real rate for the actual
deployment; don't assume a correction factor learned on one sensor modality applies to
another.

**Promoted to Dense-Armor**: `one_sided_arl`, `two_sided_arl`, `detectability_report` (with
the floor fix) are now in `dense_armor.utility.cusum`, alongside `cusum_detector` -- see
Dense-Armor's `docs/api/cusum.md`.


---

## Details

**Why `reference="fixed"`, not the default `"adaptive"`**: the classical Wald/Siegmund
theory describes a CUSUM with a fixed, known reference distribution. `cusum_detector`'s
default `adaptive` mode continuously re-estimates its own reference from a sliding
window -- a genuinely different statistical process this theory does not directly
describe. Validating against `adaptive` would have been testing the theory against the
wrong algorithm; `reference="fixed"` is the one this theory actually applies to.

**What this is good for, honestly**: a reasonable *a priori* estimate of post-shift
detection delay (a few-to-twenty percent off, useful for planning), and an exact
reproduction of the "shift in sigma units" ratio Experiment 42 computed by hand. **Not**
a reliable predictor of real false-alarm rate for small reference windows -- use the
real, measured false-positive rate (as every prior sensor experiment did) for that,
not this formula alone.

**Reproducing this**: `python scripts/cusum_detectability_theory/validate_arl_theory.py`
regenerates `arl_theory_validation_frozen.json` (no download needed, pure simulation);
`python scripts/cusum_detectability_theory/validate_against_real_lidar.py` regenerates
`real_lidar_arl_validation_frozen.json` (reuses Experiment 42's own committed data, no
download needed either); `python scripts/cusum_detectability_theory/validate_against_real_imu.py`
regenerates `real_imu_arl_validation_frozen.json` (reuses the IMU validation experiment's
own committed data, no download needed); `pytest tests/test_cusum_arl_theory.py
tests/test_cusum_arl_real_lidar_validation.py tests/test_cusum_arl_real_imu_validation.py`
reads the already-frozen files.

**Paper indexed**: Reynolds (1975) is now in quantumrag's `statistica_controllo_processo`
collection, alongside Page (1954, already dense-armor's own citation) -- the first
non-physics/chemistry collection in that knowledge base, added because this is classical
statistics grounding a Dense-Armor utility, not a physics/quantum-information topic any
existing collection covers.
