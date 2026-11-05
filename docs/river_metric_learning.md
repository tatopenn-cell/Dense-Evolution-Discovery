# Online Metric Learning: OASIS, LEGO and POLA

A k-nearest-neighbours classifier is only as good as its distance. Euclidean distance treats
every feature the same; metric learning learns a matrix `M` so that
`d(u, v) = sqrt((u - v)^T M (u - v))` pulls same-class points together and pushes
different-class points apart. This page implements three online metric learners from their
papers, one sample (pair or triplet) at a time, and compares them as kNN metrics.

## Step 1. Three algorithms, three kinds of supervision

| Algorithm | Paper | Learns from | Model | Update |
|---|---|---|---|---|
| OASIS | Chechik, Sharma, Shalit, Bengio, NIPS 2009 | triplets (query, positive, negative) | bilinear similarity `x^T W y`, no PSD or symmetry constraint | passive-aggressive: `W += tau * x (x+ - x-)^T`, `tau = min(C, loss / ‖V‖²)` |
| LEGO | Jain, Kulis, Dhillon, Grauman, NIPS 2008 | pairs with a target squared distance | Mahalanobis `A` | LogDet-regularized closed form: `A -= eta (y_bar - y) A z z^T A / (1 + eta (y_bar - y) y_hat)` |
| POLA | Shalev-Shwartz, Singer, Ng, ICML 2004 | pairs labelled similar (+1) / dissimilar (-1) | pseudo-metric `(A, b)` | `A -= y alpha v v^T`, `alpha = loss / (1 + ‖v‖⁴)`, then projection onto the PSD cone and `b >= 1` |

All three are in `scripts/river_metric_learning/metric_learning.py`.

## Step 2. Train one and use it as a kNN metric

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from metric_learning import mahalanobis, train_pola

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
A = train_pola(X_tr, y_tr, n_pairs=30000)
KNeighborsClassifier(n_neighbors=3, metric=mahalanobis(A)).fit(X_tr, y_tr).score(X_te, y_te)
```

```
0.952
```

LEGO needs a target distance for each pair, not a similar/dissimilar flag: same-class pairs
get the 5th percentile of same-class squared distances, different-class pairs the 95th
percentile of different-class ones (Section 4 of the paper). OASIS learns a similarity, not a
distance; to use it in kNN this page takes `M = W^T W`, which is always positive semi-definite
(see Step 4).

## Step 3. Dense data: same split for every method

Setup: 30% train / 70% test (stratified, `random_state=42`), features standardized on the
training set, k = 3. OASIS with `C = 0.01`; triplets/pairs: 20,000 (Iris), 30,000 (Wine),
50,000 (Digits; POLA 20,000 because of its eigendecomposition per similar pair).

| Dataset | Features | Euclidean | OASIS (`W^T W`) | LEGO | POLA |
|---|---|---|---|---|---|
| Iris | 4 | 0.9333 | 0.9524 (+1.9) | **0.9619 (+2.9)** | 0.9524 (+1.9) |
| Wine | 13 | 0.9200 | **0.9520 (+3.2)** | 0.9360 (+1.6) | **0.9520 (+3.2)** |
| Digits | 64 | 0.9459 | 0.9603 (+1.4) | 0.9372 (−0.9) | **0.9658 (+2.0)** |

POLA is best or tied-best on Wine and Digits and never below the baseline; LEGO wins on Iris
but loses on Digits. Differences are of a few test points on Iris and Wine (105 and 125 test
samples), so they are indicative, not decisive.

## Step 4. Sparse text: how OASIS's `W` becomes a distance

On 20 newsgroups (500 binary word features, L2-normalized, 70/30 split, k = 5, 20,000
triplets), the way `W` is turned into a kNN distance matters more than `C`:

| Pair | C | Euclidean | `(W + W^T)/2` | `W^T W` |
|---|---|---|---|---|
| sci.space vs comp.graphics | 0.01 | 0.5255 | 0.6071 | 0.6820 |
| sci.space vs comp.graphics | 0.1 | 0.5255 | 0.6922 | **0.8588** |
| comp.sys.ibm.pc.hardware vs sci.electronics | 0.01 | 0.5051 | 0.5356 | 0.5593 |
| comp.sys.ibm.pc.hardware vs sci.electronics | 0.1 | 0.5051 | 0.6085 | **0.7831** |

`W^T W` beats the symmetrized `W` by 17 points on both pairs at `C = 0.1`. This is our own
finding, not the paper's: Section 4.3.1 of the OASIS paper compares symmetric variants
(`sym(W)` and a dissimilarity form) against the asymmetric score in a ranking task and finds
them "slightly worse, or equal"; `W^T W` does not appear there. A plausible reason it works
here: `(W + W^T)/2` can have negative eigenvalues, so `(u - v)^T M (u - v)` stops being a
distance, while `W^T W` is always positive semi-definite.

The euclidean baseline on text is close to chance (0.51–0.53 for two classes): with 500
binary features and headers removed, raw word overlap is a weak signal, which is exactly
where a learned metric has the most room.

## What this shows

- On small dense data all three improve kNN on Iris and Wine; POLA is the most reliable
  (never below the baseline here), at the cost of an eigendecomposition per similar pair.
- On sparse high-dimensional text, OASIS (no PSD constraint, cheap updates) gives large gains,
  provided `W` is turned into a distance through `W^T W`.
- A metric learner does not fit river's existing estimator types: it learns from pairs or
  triplets and outputs a distance, not a prediction. Plugging one into river would need its
  own base class (for example `learn_triplet` / `learn_pair` and a `distance` method) that
  `neighbors` estimators can take as their metric.

## Reproduce

```bash
pip install scikit-learn
python scripts/river_metric_learning/benchmark.py
pytest tests/dense_armor/test_river_metric_learning.py
```

`benchmark.py` prints both tables and writes `data/river_metric_learning.csv` (about 11
minutes on a laptop CPU, mostly POLA's eigendecompositions and the 20 newsgroups download).

## Details

- LEGO input: the original notebook passed a boolean (same class or not) as LEGO's target;
  the paper's target is a squared distance. With the boolean, same-class pairs are pushed
  toward distance 1. The script uses the 5th/95th-percentile targets of Section 4.
- An earlier version of the dense table mixed two splits (OASIS on 70/30, LEGO and POLA on
  30/70), so the deltas did not match the baseline column; Step 3 uses one split for all.
- The different-class percentile excludes the zero self-distances on the diagonal.
- kNN distances are computed as a precomputed matrix (`d² = uMu + vMv − 2uMv`, clipped at
  `1e-12`), identical to a per-pair callable and much faster.
