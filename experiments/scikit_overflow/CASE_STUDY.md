# Case study: scikit-learn issue #5602 (StandardScaler.partial_fit overflow)

## The bug

`StandardScaler.partial_fit` keeps the cumulative sum of the samples
(`last_sum = last_mean * last_sample_count`). On long streams, or on values
with a large magnitude, the sum overflows to `inf`. The mean becomes `inf`,
and the variance is wrong.

Reproducer from the issue (giorgiop, 2015):

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
never reconstructs the sum, so it does not overflow. The file
`chan_golub_leveque_1979_stanford.pdf` in `papers/` is the source (STAN-CS-79-773,
equations 1.1 and 1.2).

## The tests

`test_incremental.py` has five tests:

1. `test_old_formula_overflows`: the old formula returns `inf` on the original
   reproducer. This is the bug.
2. `test_new_formula_no_overflow`: the new formula returns a finite value, and
   matches `batch.mean()` within `1e-10`.
3. `test_new_formula_matches_numpy_on_normal_data`: on normal data, the new
   formula matches `numpy.mean` and `numpy.var` within `1e-12`.
4. `test_new_formula_constant_data`: constant data has variance 0, no NaN, no
   division by zero.
5. `test_new_formula_streaming_matches_batch`: 1000 batches of 10 samples
   match the batch over the same data.

## The numbers

| Test | Old formula | New formula | Batch reference |
|---|---|---|---|
| Mean on the reproducer | `inf` | finite, difference < 1e-10 | finite |
| Mean on constant data | 0 (but only because `inf`-adjacent) | 42.0 | 42.0 |
| Variance on constant data | wrong | 0.0 | 0.0 |
| Streaming vs batch | wrong | difference < 1e-10 | - |

## Status

The fix and the tests pass on this branch. The upstream PR on
`scikit-learn/scikit-learn` will be opened only after the checks on this
repository are green, with a link back to this folder in the PR description.

## References

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
  pairwise algorithm for computing sample variances*. Stanford CS tech report
  STAN-CS-79-773. PDF in `papers/`.
- Welford, B. P. (1962). *Note on a method for calculating corrected sums of
  squares and products*. Technometrics 4(3), 419-420. Bibliographic only
  (paywalled).
- Issue #5602: https://github.com/scikit-learn/scikit-learn/issues/5602
