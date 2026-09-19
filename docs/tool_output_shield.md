# A Pre-Reasoning Numeric Shield Catches 2 of 3 Injected Corruptions in Real Attack Telemetry

An agent that calls an external tool and gets back a number usually treats
that number as fact the instant it arrives -- before any reasoning happens.
This experiment asks a narrower question than
[Indirect Prompt Injection vs. Dense-Armor](agent_indirect_prompt_injection.md):
not "can Dense-Armor detect a semantic attack," but "can a lightweight,
stateless numeric gate stop an untrusted tool's numbers from becoming a
false premise before an agent computes anything from them."

## Step 1. A real numeric stream, not a synthetic one

The same real telemetry used in the sibling security experiment: Sysmon
`EventID 10` (ProcessAccess) counts from the
[OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets) APT29
day-1 emulation, binned into 500 time windows.

```python
import urllib.request, zipfile, json
import numpy as np

url = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/compound/apt29/day1/apt29_evals_day1_manual.zip"
urllib.request.urlretrieve(url, "data/apt29_day1_manual.zip")
with zipfile.ZipFile("data/apt29_day1_manual.zip") as zf:
    lines = zf.open(zf.namelist()[0]).readlines()
```

Each line is one JSON Sysmon event with a real UTC `@timestamp` and
`EventID`. Sorting by time and histogramming `EventID == 10` into 500 equal
bins gives a 500-point series -- exactly the shape a monitoring tool would
hand back to an agent asking "how many process-access events per window."

## Step 2. Three corruptions a tool could plausibly return without erroring

```python
corrupted = real_series.copy()
injected = {50: np.nan, 200: 999999999999999999.0, 350: -1.0}
for idx, val in injected.items():
    corrupted[idx] = val
```

A silent `NaN` (a failed aggregation that didn't raise), an absurdly large
count, and a negative count. The huge value is not an arbitrary round
number: it is the exact `999999999999999999`-dollar bounty figure found
during real GitHub bounty-listing research (`HUNTER.txt`) -- a value a
spam/garbage API response can plausibly emit, reused here as a stand-in for
"a tool that returns something syntactically valid but semantically
absurd."

## Step 3. The gate, run before any mean is computed

```python
from dense_armor import Armatura

shield = Armatura(livello_ia=0.0)
pulito, K, anomalie = shield.analizza(corrupted)
```

`livello_ia=0.0` means "treat the agent as a newborn": the shield actively
clips what it flags instead of only marking it. `analizza` returns the
cleaned series, a per-point confidence `K`, and the list of anomalous
indices -- all before any downstream reasoning sees the numbers.

## Step 4. The real numbers

| | value |
|---|---|
| Points | 500 |
| Shield runtime | 6304.5 ms (JAX JIT warm-up dominates; one-shot cold call) |
| Total anomalies flagged | 48 |
| idx 50 (`NaN`) caught | yes -- cleaned to 39.50 |
| idx 200 (`1e18`) caught | yes -- cleaned to 0.00 |
| idx 350 (`-1.0`) caught | **no** |
| Naive mean (NaN-skipped only) | 2,004,008,016,032,141.75 |
| Mean after the shield | 54.07 |
| True mean (clean reference series) | 78.57 |

The huge value alone would have pushed a naive downstream average from a
real ~79 to over two quadrillion -- a corruption large enough that no agent
could miss it once computed, but only *after* it already contaminated the
result. The shield's post-gate mean (54.07) is much closer to reality, but
it is not a perfect recovery of the true mean (78.57): the missed `-1.0` and
the 45 other flagged points (real variance in genuine attack telemetry, not
injected) both pull it down.

## Step 5. The honest miss

`-1.0` is a plausible negative count for a field that should never be
negative, yet it was not flagged. `analizza`'s anomaly test compares each
point's shift against the series' own local deviation and a robust
median/MAD threshold -- a small, in-range-looking constant near a
low-activity window does not clear either threshold. The gate is a
statistical outlier detector, not a domain-rule checker: it has no notion
that "process-access counts are non-negative." A deterministic range check
(`count >= 0`) would have caught this one instantly and costs nothing to
add on top, but was deliberately not folded into the shield's own logic to
keep the comparison honest about what a stateless statistical gate does and
does not cover.

## Details

**What this experiment did not test**: semantic correctness of the tool's
answer (see [Indirect Prompt Injection vs. Dense-Armor](agent_indirect_prompt_injection.md)
for that boundary) -- only whether numerically implausible values survive
to reach the point where an agent would reason over them.

**Reproducing this**: `python scripts/tool_output_shield.py` from this
repository's root (requires `dense-armor` importable and network access to
fetch the OTRF dataset; downloads ~500 points' worth of a 33-minute real
attack emulation, a few MB).

**Provenance**: built on `dense_armor.Armatura` (this project's own
in-repo wearable wrapper around the `hybrid_shield` engine in
`dense_armor/core/hybrid_engine.py`), following the same real-data-first
pattern as the rest of this repository's Dense-Armor experiments.
