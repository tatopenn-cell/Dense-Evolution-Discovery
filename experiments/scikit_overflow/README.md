# Task T1: fix the overflow in `StandardScaler.partial_fit` (scikit-learn issue #5602)

**Case study published on Dense-Evolution-Discovery, branch `case/scikit-overflow`.**
**Upstream PR: to be opened on `scikit-learn/scikit-learn` only after the checks here are green.**

## The bug

`StandardScaler.partial_fit` keeps the cumulative sum of the samples
(`last_sum = last_mean * last_sample_count`). On long streams, or on values
with a large magnitude, the sum overflows to `inf`. The mean becomes `inf`,
and the variance is wrong. The reproducer, from the issue (giorgiop, 2015):

~~~python
from sklearn.preprocessing import StandardScaler
import numpy as np

rng = np.random.RandomState(0)
max_f = np.finfo(np.float64).max / 1e5
min_f = max_f / 1e2
batch = rng.uniform(min_f, min_f, size=(500000, 1))

iscaler = StandardScaler(with_std=False)
for _ in range(100):
    iscaler = iscaler.partial_fit(batch)
print(iscaler.mean_)  # inf
~~~

## The fix

The cumulative sum is replaced by the delta-based update of Chan, Golub and
LeVeque (1979), which is the batch generalisation of Welford (1962):

~~~text
delta = new_mean - last_mean
updated_mean = last_mean + delta * new_count / total_count
updated_m2 = last_m2 + new_m2 + delta**2 * last_count * new_count / total_count
~~~

The formula is algebraically equivalent to the cumulative-sum formula, but it
never reconstructs the sum, so it does not overflow. The source is the
Stanford tech report STAN-CS-79-773:

- Eq. (2.1a), page 4 of the paper (page 8 of the PDF): the mean merge.
- Eq. (2.1b), page 4 of the paper (page 8 of the PDF): the variance merge.
- Eq. (1.4), page 2 of the paper (page 6 of the PDF): the corrected two-pass
  correction term.

The PDF is in `papers/chan_golub_leveque_1979_stanford.pdf`.

## The tests

`test_incremental.py` has six tests. All pass (`pytest test_incremental.py -v`
-> `6 passed in 2.07s`):

| Test | What it checks |
|---|---|
| `test_old_formula_overflows` | The old formula returns `inf` on the original reproducer. |
| `test_new_formula_no_overflow` | The new formula returns a finite value, and matches `batch.mean()` within `1e-10`. |
| `test_new_formula_matches_numpy_on_normal_data` | On normal data, the new formula matches `numpy.mean` and `numpy.var` within `1e-12`. |
| `test_new_formula_constant_data` | Constant data has variance 0, no NaN, no division by zero. |
| `test_new_formula_streaming_matches_batch` | 1000 batches of 10 samples match the batch within `1e-10`. |
| `test_new_formula_with_std_false` | The `with_std=False` case (`last_variance=None`) is handled, for both formulas. |

The mutation test (restoring the old formula and re-running the suite) shows
which tests fail: `test_old_formula_overflows` and
`test_new_formula_no_overflow`. The other four are regression checks on normal
inputs; the old formula passes them too. The distinction is in
`CASE_STUDY.md` under "Test relevance".

## The numbers

Measured in `incremental.py` (mutation test) and in scikit-learn from pip with
the fix applied to `sklearn/utils/extmath.py`:

| Scenario | Old formula | New formula (scikit-learn) | Batch reference |
|---|---|---|---|
| Mean on the reproducer | `inf` | `1.7976931348623154e+301` | `1.7976931348623154e+301` |
| Difference on the reproducer | `inf` | `0.0` | `0` |
| Mean on constant data | `[42.0, 42.0]` | `[42.0, 42.0]` | `[42.0, 42.0]` |
| Variance on constant data | `[0.0, 0.0]` | `[0.0, 0.0]` | `[0.0, 0.0]` |
| Streaming vs batch, mean | `2.7755575615628914e-17` (vs numpy) | `1.249000902703301e-16` (vs sklearn fit) | `0` |
| Streaming vs batch, variance | `3.197442310920451e-14` (vs numpy) | `4.973799150320701e-14` (vs sklearn fit) | `0` |

The streaming rows are two independent measurements on the same seed
(`np.random.RandomState(1)`, 10000 samples, 2 features). The old formula is
compared with `numpy.mean` / `numpy.var`; the new formula is compared with
`sklearn.preprocessing.StandardScaler.fit`.

## Files

- `incremental.py`: the old formula and the new formula, side by side.
  Coverage 100% (57 statements, 0 missed).
- `test_incremental.py`: the six tests.
- `CASE_STUDY.md`: the full analysis.
- `README.md`: the summary and the run instructions.
- `results.json`: the measured results and the pytest summary.
- `papers/chan_golub_leveque_1979_stanford.pdf`: the source.

## References

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
  pairwise algorithm for computing sample variances*. Stanford CS tech report
  STAN-CS-79-773. PDF in `papers/`. Eq. (2.1a) and (2.1b), page 4 of the paper
  (page 8 of the PDF); Eq. (1.4), page 2 of the paper (page 6 of the PDF).
- Welford, B. P. (1962). *Note on a method for calculating corrected sums of
  squares and products*. Technometrics 4(3), 419-420. Bibliographic only
  (paywalled).
- Issue #5602: https://github.com/scikit-learn/scikit-learn/issues/5602

## Status

The fix and the tests pass on this branch: `6 passed in 2.07s`,
`mkdocs build --strict` exit 0, `ruff` clean, `mypy` clean, coverage 100% on
`incremental.py`. The upstream PR on `scikit-learn/scikit-learn` will be
opened only after the checks on this repository are green, with a link back
to this folder in the PR description.

Branch `case/scikit-overflow`, last commit `b67aea0`.
