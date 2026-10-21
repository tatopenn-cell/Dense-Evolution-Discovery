# Contributing Streaming Drift Detectors to online-ml/river

Same GitHub-issue-research strategy as Experiment 43 (LeRobot): search real, currently-open
needs instead of guessing one, and only propose a fix after validating it -- but this time
the target project (`online-ml/river`, a mature streaming-ML library, thousands of stars)
had an explicit, maintainer-authored roadmap issue asking for exactly a capability this
project's own `dense_armor.utility.cusum` already implements. That overlap turned into a
real bug found in Dense-Armor's shipped code, and a genuine, honestly-profiled trade-off
across three classical detectors -- none of which cleanly "wins."

## Step 1. Finding a credible need, and rejecting the weaker candidates

```bash
gh search issues "change point detection" "telemetry" --state open --limit 15
```

Several candidates surfaced. Checked directly, not assumed:

- `online-ml/river` issue [#1914](https://github.com/online-ml/river/issues/1914) --
  "Drift module roadmap", written by the maintainer (Max Halford), citing a real external
  benchmark (Besbes et al. 2026, arXiv:2606.18377) that found river's own detectors weak
  against a purpose-built CUSUM baseline the paper's authors had to implement themselves.
  Lists **CUSUM (Page 1954)** as a "highest-value, lowest-risk" Family-1 gap.
- Three other candidates (a personal sensor-analyzer repo, a temperature-monitor repo with
  its own fix already in flight, a Java telemetry project) were checked and rejected: low
  community credibility, redundant with work already in progress, or no shared language
  runtime to contribute code in -- same evaluation discipline as Experiment 43's rejected
  candidates.

**Before writing any code**: checked whether `river.drift.PageHinkley` already covers this
-- its own docstring says it "implements the CUSUM control chart." Reading its actual
`update()` logic settled it: `PageHinkley` standardizes against a **fading**,
exponentially-forgetting mean (`self._x_mean`, a running `stats.Mean()` updated every
step) -- the same "adaptive" convention `dense_armor.utility.cusum.cusum_detector` already
distinguishes from Page's original scheme. A **fixed** reference (estimated once, never
updated) is a real, non-redundant gap.

**Also checked**: had anyone already proposed and been rejected? `gh pr list --search
CUSUM` found one old, closed PR (#1252, 2023) -- closed for process reasons (too large, a
student group bundling many methods at once), never discussing CUSUM specifically. Another
contributor (`jevwithwind`) had already claimed the evaluation harness (PR #1963, still
open/under review) and BOCPD from the same roadmap -- not CUSUM.

## Step 2. A real parameter bug, found before it could bite

```python
from cusum_river import CUSUM
import random

rng = random.Random(12345)
cusum = CUSUM(warmup_period=30, k=0.5, h=5.0)  # the "classical" textbook default
data_stream = [rng.gauss(0, 1) for _ in range(500)] + [rng.gauss(1.5, 1) for _ in range(500)]
for i, val in enumerate(data_stream):
    cusum.update(val)
    if cusum.drift_detected:
        print(f"Change detected at index {i}, input value: {val:.3f}")
        break
```

```
Change detected at index 225, input value: 1.134
```

A false alarm at index 225 -- 275 points before the real change at 500. Not a fluke:

```python
from benchmark_harness import evaluate
res = evaluate(name="h=5.0", detector_factory=lambda: CUSUM(warmup_period=30, k=0.5, h=5.0),
                rng_seed=42, n_trials=200, n_points=1000, change_at=500,
                mean_before=0.0, mean_after=1.0, std=1.0)
res.false_alarm_rate
```

```
0.88
```

k=0.5/h=5.0 -- cited casually as "the" classical CUSUM tuning -- gives an **88%
stream-level false-alarm rate** on a 1000-sample purely stable series (200 trials above;
1000 trials: 0.880 and 0.862 for two seeds). The reason is the average run length (ARL)
under no change: **about 469 samples** (closed-form theory, Reynolds 1975 with Siegmund's
correction, `arl_theory.two_sided_arl`: 469.1; simulation with the true mean and sigma
known: 466). That is shorter than the 1000-sample horizon, so the probability of at least
one alarm is 1 - exp(-1000/469) = 0.88 (theory 0.881; simulated 0.888 with known
parameters, 0.877 with the baseline estimated on 30 points). Swept `h` empirically (1000
trials, two seeds): `h=10.0` gives a 27% false-alarm rate; `h=20.0` gives **5-6%**
(0.057 and 0.051), 0.5-0.8% missed detections and F1=0.91 for a 1-sigma shift, every
spurious flag counted. The theory gives practically no false alarm at `h=20.0` (ARL
1.6e9): the 5% left is the effect of estimating the mean and the sigma on the first 30
points, the same gap between theory and a detector with an estimated reference that
Experiment 44 measured for Dense-Armor's own detector.

*Correction, 21 October 2026.* The first version of this page said 3.0% false alarms and
F1=0.975 at `h=20.0`, and an average run length of "~19-38 samples" at `h=5.0`. The 3.0%
does not reproduce: the same call with 200 trials gives 0.095 today (Python 3.12.10, river
0.26.1) and the frozen comparison file of the same day records 7.5%; with 1000 trials the
rate is 5-6%. The F1 of 0.975 does not match the harness after the fix of Step 3 (0.91).
The 19 and 38 are the ARL for a shift of 0.5 sigma (38.02 = (5 + 1.166)^2 for one branch,
19.01 for two), not the ARL under no change. All three are replaced by the numbers above,
produced by `arl_check.py`; the conclusion (`h=5.0` is a trap, `h=20.0` removes it)
stands.

**Generalized immediately, not left as a one-off**: `dense_armor.utility.cusum.
cusum_detector` ships the exact same `h=5.0` default.

```python
import numpy as np
from dense_armor.utility.drift.cusum import cusum_detector

rng = np.random.default_rng(42)
n_with_flag = sum(1 for _ in range(200)
                   if cusum_detector(rng.normal(0, 1, 1000), h=5.0, reference="adaptive")[0].any())
n_with_flag / 200
```

```
1.0
```

**100%** on Dense-Armor's own shipped code, in `adaptive` mode (its default). Fixed there
too, empirically recalibrated to `h=20.0` (3.5% `adaptive` / 15.5% `fixed`), shipped in
[Dense-Armor v1.1.13](https://github.com/tatopenn-cell/Dense-Armor/pull/12) -- a real
production fix, found as a direct side effect of this contribution, not the goal of it.

## Step 3. A harness bug, found by stress-testing the harness itself

The user asked a sharp question mid-review: *"if it's really imperceptible, use
resonance"* led nowhere (not enough samples per event for a real frequency estimate), but
the same instinct -- **don't trust a metric until it's been attacked** -- prompted a
direct stress test against trivial baselines, per a real methodological critique already
posted on river's own PR #1963 by another contributor (`mateenali66`): *"a real detector
can land below the data-ignoring one without the report saying so."*

```python
from dummy_baselines import AlwaysFire
from benchmark_harness import evaluate

evaluate(name="AlwaysFire", detector_factory=AlwaysFire, rng_seed=123,
          n_trials=200, n_points=1000, change_at=500, mean_before=0.0, mean_after=1.0).f1()
```

```
0.667
```

A detector that flags literally every point scored **F1=0.667** -- higher than CUSUM's own
0.512 at a 0.5-sigma shift, at the time. Root cause: `false_alarms` counted STREAMS with
at least one spurious flag, not the number of flags -- a detector firing 1000 times in one
stable stream was penalized identically to one firing once. Fixed (`false_alarm_flags`,
every spurious flag counted individually); re-run, `AlwaysFire` collapses to F1=0.0013, and
every real detector clears every dummy baseline (`AlwaysFire`, `NeverFire`,
`river.drift.DummyDriftDetector`) by a wide margin at every shift size tested.

## Step 4. Honest comparison, no tuning in anyone's favor

Each detector at its own library defaults (ADWIN, KSWIN, PageHinkley) or its own
empirically-calibrated default (CUSUM, EWMA, Shewhart -- see Step 6):

| shift | CUSUM | EWMA | Shewhart | ADWIN | PageHinkley | KSWIN |
|---|---|---|---|---|---|---|
| 0.5σ | 0.424 | 0.215 | 0.031 | **0.847** | 0.797 | 0.359 |
| 1.0σ | 0.835 | 0.618 | 0.199 | **1.000** | 0.799 | 0.527 |
| 1.5σ | 0.842 | 0.805 | 0.453 | **1.000** | 0.802 | 0.539 |
| 2.0σ | 0.842 | 0.815 | 0.661 | **1.000** | 0.802 | 0.539 |
| 3.0σ | 0.842 | 0.815 | **0.871** | 1.000 | 0.802 | 0.539 |

The table is the record `compare_to_river_baselines_frozen.json` of 3 September (200
trials per cell); it was not run again on 21 October.

**ADWIN wins on F1 everywhere.** No detector built here is proposed as a replacement for
it. What each real, honest number says instead:

- **CUSUM** beats PageHinkley on F1 for medium/large shifts, but is the weakest of the
  three real detectors (CUSUM/PageHinkley/ADWIN) at a small 0.5-sigma shift -- a genuine,
  disclosed limitation, not smoothed over. A second, small-k parallel channel was tried to
  fix this: recovers some small-shift sensitivity but adds false alarms everywhere,
  net negative -- tried and rejected, not silently dropped.
- **EWMA** is the fastest detector in the entire comparison for medium/large shifts
  (mean delay 3.4-9.0 samples at 2-3 sigma, faster than both CUSUM and ADWIN), at the cost
  of the weakest small-shift F1 after Shewhart.
- **KSWIN** at its own library default has a 73.5% false-alarm rate -- a real, surprising
  number about river's own shipped default, not an error in this analysis.

## Step 5. Verifying interface conformity against river's own test suite, not by eye

```python
import river.checks as checks
from ewma_river import EWMA

for check in checks.yield_checks(EWMA()):
    check(EWMA().clone())
```

13/14 pass; the sole failure (`check_tags`, `AttributeError: no attribute '_tags'`) also
fails identically on `river.drift.PageHinkley`, `ADWIN`, and `KSWIN` when the same check is
run against them directly -- a real bug in river 0.26.1's own `check_estimator` for the
whole `DriftDetector` family, not something specific to this contribution. Same result
(13/14, same single failure) for CUSUM and Shewhart.

## Step 6. Shewhart: built for completeness, not because it's competitive

River's own roadmap lists Shewhart as "optional, completes the control-chart set." Built
and calibrated the same way (`L=4.5`, checked against the same 88.5%-false-alarm trap at
the textbook `L=3.0`) -- but being memoryless (each point judged independently, no
accumulation), it is structurally weak for anything but large, abrupt shifts: F1=0.871 at
3-sigma, but F1 collapses to 0.20-0.45 for 1.0-1.5-sigma shifts, and it is the slowest
detector tested at every shift size (27-286 sample delay). Included in the comparison
table above for completeness; not currently proposed to river as its own contribution --
no distinctive advantage was found to justify it.

## The verification, in one place

What validates the detector is not the detector: it is the order of the checks and what
each one is allowed to conclude. The protocol, as it stands after the corrections of Steps
2 and 3:

| # | Check | What it decides | Where |
|---|---|---|---|
| 1 | False-alarm rate over the monitoring horizon, on a purely stable stream, swept over the threshold | A threshold is accepted only if the rate over the real horizon is low, not because the textbook says so | `benchmark_harness.evaluate`, `arl_check.py` |
| 2 | Count every spurious flag, never the streams that have one | A detector that fires 1000 times must not score like one that fires once | `false_alarm_flags` in `benchmark_harness.py` |
| 3 | Stress the harness with trivial detectors | A real detector must clear `AlwaysFire`, `NeverFire` and `DummyDriftDetector` at every shift size; `AlwaysFire` scores F1=0.0013 | `dummy_baselines.py` |
| 4 | Run the same test on the defaults of the sister library | A calibration found once is checked against what is already shipped (100% at `h=5.0`, 3.5% at the default `h=20.0`) | Step 2 |
| 5 | Compare the measured run length with the closed-form theory | Explains the result instead of only measuring it: 469.1 theory, 466 simulated, 0.877 and 0.05 with an estimated baseline | `arl_check.py` |
| 6 | Compare at library defaults with the detectors that exist, and state where it loses | ADWIN wins everywhere; the 0.5-sigma weakness is disclosed | `compare_to_river_baselines.py` |
| 7 | Run the project's own interface checks | 13 of 14 pass; the one failure is the same on the other detectors | Step 5 |

Checks 1 to 4, 6 and 7 were published on 3 September 2026 (commit
[3018908](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/commit/3018908810bbc9b527d6f7937b11c92fdb45f500),
06:49 UTC); check 5 and the 1000-trial sweep of check 1 on 21 October 2026.

Fingerprints (first 16 hexadecimal characters of the SHA-256) of the scripts that
implement the protocol, at the commit that adds this section:

| File | SHA-256 |
|---|---|
| `benchmark_harness.py` | `0ac494203ea00b91` |
| `dummy_baselines.py` | `9445da89f8a0daec` |
| `cusum_river.py` | `0678abfd2a5bd67f` |
| `arl_check.py` | `a738080dcdf23680` |
| `check_candidate_cusum.py` | `8dc86ae797d329b6` |

## Current status

- **3 September 2026.** Posted [a comment on issue #1914](https://github.com/online-ml/river/issues/1914)
  proposing CUSUM specifically, with the h=5.0 finding, the harness stress-test finding and
  the honest F1 comparison. The prototype and the scripts were pushed to this repository
  the same day (commit 3018908).
- **21 September 2026.** The maintainer wrote to all the contributors of the thread that he
  prefers to give the drift work to someone with a track record in drift learning. Between
  21 and 26 September another contributor opened pull requests for EWMA (#2014), D3
  (#2015), CUSUM (#2026), Shewhart (#2027) and RDDM (#2028). The CUSUM one keeps the
  textbook threshold 5 as its default and reports 42 false alarms over 10 x 2000 stable
  samples (ARL about 476), the trap measured in Step 2.
- **5 October 2026.** The maintainer wrote that he does not ask anyone to stop
  contributing, only to follow the contribution guidelines.
- No pull request was opened from this work. The prototypes (CUSUM, EWMA, Shewhart), the
  harness and the verification protocol above stay in this repository.

## Reproducing this

All code lives in `scripts/river_cusum_contribution/`: `cusum_river.py`, `ewma_river.py`,
`shewhart_river.py` (the three detectors), `benchmark_harness.py` (the evaluation harness,
with the false-alarm-counting fix), `dummy_baselines.py` (`AlwaysFire`/`NeverFire`),
`compare_to_river_baselines.py` (the full comparison, writes
`compare_to_river_baselines_frozen.json`), `arl_check.py` (the run length against the
theory and the 1000-trial false-alarm sweep, 15 seconds). `check_candidate_cusum.py` applies the
verification (check 1 and the flag count) to any streaming CUSUM loaded from a file, for
example a candidate pull request, without copying it into this repository. Requires `pip install river` (not a Dense-Armor
or Dense-Evolution dependency, kept isolated to this experiment).
