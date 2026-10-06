# Online Platt Scaling for online-ml/river

`online-ml/river` issue [#1421](https://github.com/online-ml/river/issues/1421) asks for
Platt scaling: a small model that takes the probabilities of another classifier and makes
them honest. River learns one sample at a time, so the calibrator has to learn online too.
This page builds it from Gupta and Ramdas, *Online Platt Scaling with Calibeating*
(ICML 2023, [arXiv:2305.00070](https://arxiv.org/abs/2305.00070)), then tests a personal
variant against it on seven data streams.

## Step 1. What Platt scaling does

A classifier gives a probability `p`. Platt scaling replaces it with

```
sigmoid(a * logit(p) + b)
```

With `a = 1, b = 0` nothing changes. A tree that says 0.99 when it is right 80% of the time
needs `a < 1`, which pulls its probabilities back toward 0.5. Online Platt Scaling (OPS)
learns `a` and `b` one sample at a time.

## Step 2. The paper's algorithm

```python
from river import datasets, evaluate, metrics, tree
from ops import OnlinePlattScaling

dataset = datasets.Phishing()
evaluate.progressive_val_score(dataset, tree.HoeffdingTreeClassifier(), metrics.LogLoss())
evaluate.progressive_val_score(dataset, OnlinePlattScaling(tree.HoeffdingTreeClassifier()), metrics.LogLoss())
```

```
LogLoss: 0.4535476064322544
LogLoss: 0.35021471255578124
```

`OnlinePlattScaling` (`scripts/river_platt_scaling/ops.py`) follows Algorithm 1 of the
paper's appendix exactly:

| Element | Paper | Code |
|---|---|---|
| Pseudo-features | `logit(f(x))` and a bias | `[logit(p), 1]` |
| Start | `(a, b) = (1, 0)`: the base probabilities pass through unchanged | `theta = [1, 0]` |
| Optimizer | Online Newton Step, `gamma = 0.1`, `A_0 = rho I`, `rho = 100` | 2x2 matrix `A`, Newton step `A^-1 g / gamma` |
| Projection | `‖(a, b)‖₂ ≤ 100` in the `A` metric | `radius = 100` |
| Clipping | `f(x)` in `[0.01, 0.99]` (Theorem 2.1) | `clip = 0.01` |

On Phishing the log-loss of a Hoeffding tree drops from 0.4535 to 0.3502. River's own
`check_estimator` passes, and the class runs at about 32,000 samples per second on top of
the base model (river asks for at least 5,000).

## Step 3. Tracking (TOPS)

The paper's answer to drift is calibeating by tracking (Section 3.1, eq. 11): split `[0, 1]`
into 10 bins, and when OPS predicts a value in a bin, predict instead the average outcome
seen so far in that bin (the bin mid-point if it is empty).
`TrackingOnlinePlattScaling` (`tops.py`) adds this on top of OPS.

## Step 4. A personal variant: ensemble of step sizes and a JSD spring

`jsd_ops.py` keeps three calibrators with different SGD step sizes (0.01, 0.1, 1.0) and
mixes them with BLUE weights from their recent errors, adds a Jensen-Shannon "spring" that
reacts to changes in the distribution of probabilities, and switches calibration off when
the base model's expected calibration error is already below 0.05.

## Step 5. Results on seven streams

Setup: a Hoeffding tree trained on 500 random samples from the first half of each stream,
then frozen; 5000 samples per stream, predict then learn. Each cell is
accuracy / log-loss / Brier score (log-loss with `p` clipped to `[0.001, 0.999]`, since
TOPS can predict exactly 0 or 1).

| Stream | Base | OPS | TOPS | JSD + BLUE + auto |
|---|---|---|---|---|
| Phishing | 0.8848 / 0.3626 / 0.1031 | 0.8848 / 0.3648 / 0.1035 | 0.8824 / 0.3802 / 0.1055 | 0.8848 / 0.3626 / 0.1031 |
| Elec2 | 0.8114 / 0.5095 / 0.1494 | 0.8202 / 0.4362 / 0.1359 | 0.8180 / 0.4437 / 0.1368 | **0.8644 / 0.3537 / 0.1042** |
| Bananas | 0.6372 / 0.6665 / 0.2363 | 0.6304 / 0.6676 / 0.2364 | 0.6278 / **0.6588 / 0.2308** | 0.5920 / 0.6753 / 0.2405 |
| SEA drift | 0.8748 / 0.2728 / 0.0863 | **0.9416 / 0.1435 / 0.0430** | 0.9408 / 0.1561 / 0.0442 | 0.9372 / 0.1561 / 0.0460 |
| Agrawal drift | 0.6370 / 1.0399 / 0.2820 | **0.8560** / 0.3513 / **0.1074** | 0.8552 / **0.3504** / 0.1084 | 0.8296 / 0.3826 / 0.1206 |
| Hyperplane | 0.7500 / 0.5256 / 0.1738 | 0.7488 / **0.5015 / 0.1652** | 0.7422 / 0.5155 / 0.1681 | 0.7398 / 0.5160 / 0.1712 |
| SINE drift | 0.8578 / 0.5345 / 0.1164 | **0.8766 / 0.3186** / 0.0984 | 0.8732 / 0.3316 / **0.0978** | 0.8578 / 0.4387 / 0.1156 |

Which part of the variant does the work on Elec2:

| Elec2 | Accuracy | Log-loss | Brier |
|---|---|---|---|
| SGD 0.1 only | 0.8348 | 0.4103 | 0.1241 |
| SGD 1.0 only | **0.8718** | 0.4657 | 0.1114 |
| ensemble only | 0.8626 | 0.3562 | 0.1050 |
| JSD only | 0.8360 | 0.4048 | 0.1217 |
| auto only | 0.8348 | 0.4103 | 0.1241 |
| ensemble + JSD | 0.8644 | **0.3537** | **0.1042** |

## What this shows

- OPS, as written in the paper, has the best or near-best calibration on five of seven
  streams. On Bananas nothing beats the base model: there is nothing to correct.
- TOPS stays within a few thousandths of OPS everywhere; tracking does not help on these
  streams.
- The variant wins clearly on Elec2 only. Almost all of its gain there comes from the
  ensemble of step sizes, which lets it adapt fast to a stream that changes quickly; the JSD
  spring adds a little on top. On the other five streams it sits between the base and OPS.
- OPS optimizes log-loss, not accuracy: its accuracy moved by less than 0.01 on the streams
  without a large drift, sometimes down.

The candidate for river is therefore `ops.py`, the paper's algorithm. The variant is a
research lead for fast-drifting streams, not yet a general improvement.

## Reproduce

```bash
pip install river
python scripts/river_platt_scaling/benchmark.py
pytest tests/dense_armor/test_river_platt_scaling.py
```

`benchmark.py` prints both tables and writes `data/river_platt_scaling.csv`.

## Details

- The ONS projection radius is 100 (`‖(a, b)‖₂ ≤ 100`). On Phishing, Elec2 and SEA,
  `‖(a, b)‖` never went above 3.6, so the projection never triggered.
- The first version of OPS (in the notebook) used SGD, started from `a = 0` and needed a
  200-sample warm-up to hide that start. Moving to the paper's ONS with `(a, b) = (1, 0)`
  removed the warm-up.
- River's contribution rules (`AGENTS.md`) ask that docstrings, commit messages and PR text
  be written by the contributor, and that agent-written code be disclosed in the PR.
