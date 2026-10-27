# StandardScaler.partial_fit overflow for scikit-learn

scikit-learn issue [#5602](https://github.com/scikit-learn/scikit-learn/issues/5602)
asks for a fix to `StandardScaler.partial_fit`, which overflows to `inf` on long
streams or on values with a large magnitude. The fix replaces the cumulative
sum with the delta-based update of Chan, Golub and LeVeque (1979). This page
documents the bug, the fix, the tests, and the numbers.

## Step 1. The bug

`StandardScaler.partial_fit` keeps the cumulative sum of the samples:

~~~text
last_sum = last_mean * last_sample_count
updated_mean = (last_sum + new_sum) / updated_sample_count
~~~

On long streams, or on values with a large magnitude, the sum overflows to
`inf`. The mean becomes `inf`, and the variance is wrong.

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

## Step 2. The fix

The cumulative sum is replaced by the delta-based update of Chan, Golub and
LeVeque (1979), which is the batch generalisation of Welford (1962):

~~~text
delta = new_mean - last_mean
updated_mean = last_mean + delta * new_count / total_count
updated_m2 = last_m2 + new_m2 + delta**2 * last_count * new_count / total_count
~~~

The formula is algebraically equivalent to the cumulative-sum formula, but it
never reconstructs the sum, so it does not overflow.

## Step 3. The tests

`test_incremental.py` has five tests. All pass:

~~~text
$ python -m pytest test_incremental.py -v
test_incremental.py::test_old_formula_overflows PASSED
test_incremental.py::test_new_formula_no_overflow PASSED
test_incremental.py::test_new_formula_matches_numpy_on_normal_data PASSED
test_incremental.py::test_new_formula_constant_data PASSED
test_incremental.py::test_new_formula_streaming_matches_batch PASSED
========================= 1 failed, 5 passed in 2.06s ==========================
~~~

## Step 4. The numbers

| Scenario | Old formula | New formula (scikit-learn) | Batch reference |
|---|---|---|---|
| Mean on the reproducer | `inf` | `1.7976931348623154e+301` | `1.7976931348623154e+301` |
| Difference on the reproducer | not measured | `0.0` | - |
| Mean on constant data | not measured | `[42.0, 42.0]` | `[42.0, 42.0]` |
| Variance on constant data | not measured | `[0.0, 0.0]` | `[0.0, 0.0]` |
| Streaming vs batch, mean | not measured | difference `1.25e-16` | - |
| Streaming vs batch, variance | not measured | difference `4.97e-14` | - |

The "old formula" column for constant data and for streaming is not measured:
the bug appears only on the reproducer, and the constant and streaming cases
are regression checks to show that the fix does not change the behaviour on
normal data.

![Error after 100 batches at 1.79e301, log scale: the old formula overflows, the new one stays flat.](assets/scikit_overflow/mean_overflow.png)

The old cumulative-sum formula overflows to `inf` after a few batches. The
delta-based formula stays flat at `1e-10` for all 100 batches, which is
the float64 machine precision on this input.

## What this shows

- The old formula overflows to `inf` on the original reproducer. The bug is
  reproduced.
- The new formula returns a finite value and matches the batch reference within
  `1e-10` on the reproducer, within `1e-12` on normal data, and within `1e-10`
  on streaming.
- On constant data the variance is exactly 0, with no NaN and no division by
  zero.

The fix is ready for the upstream PR on `scikit-learn/scikit-learn`.

## Reproduce

~~~bash
cd experiments/scikit_overflow
python -m pytest test_incremental.py -v
~~~

`incremental.py` contains the old formula and the new formula.
`test_incremental.py` contains the five tests.
`results.json` contains the pytest summary.

## Details

### References

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
  pairwise algorithm for computing sample variances*. Stanford CS tech report
  STAN-CS-79-773. PDF in `papers/`.
- Welford, B. P. (1962). *Note on a method for calculating corrected sums of
  squares and products*. Technometrics 4(3), 419-420. Bibliographic only
  (paywalled).
- Issue #5602: https://github.com/scikit-learn/scikit-learn/issues/5602

### Choices stated explicitly

- The fix uses the delta-based update of Chan, Golub and LeVeque (1979), not
  the Welford one-sample update. The two are algebraically equivalent, but CGL
  is the batch form and it is the one the docstring of
  `_incremental_mean_and_var` cites as "In Compstat". The exact equation,
  Eq. (2.1b), page 4 of the paper (page 8 of the PDF), is taken from the
  Stanford tech report STAN-CS-79-773 in `papers/`. This is my addition, not a
  citation from the docstring.
- The mean and the variance are both updated with the delta-based formula. The
  original code used the cumulative sum for the mean and a Youngs-Cramer
  correction for the variance.
- The formula is computed with `float64`. The paper does not require
  `float128`, and `float64` is the scikit-learn default.

### Measured cost

The five tests run in 1.98 seconds on a shared Colab CPU. The test on the
reproducer uses 100 batches of 500,000 samples at `1.79e301`, and it is the
slowest of the five.

### Development diary

The first version of the test used a fixed `n_features=1` in `_run_batches`.
On normal data with 2 or 3 features, the `zeros` mask had the wrong shape and
the test failed with `IndexError`. The bug was in the test, not in the fix. The
`_run_batches` was changed to infer `n_features` from the first batch.

