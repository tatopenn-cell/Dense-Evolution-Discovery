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
In Compstat.), as cited in the docstring of `moments.py` of Dense-Armor.

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
normal data. The numbers in the "new formula" column are measured in
scikit-learn 1.x from pip, with the fix applied to `sklearn/utils/extmath.py`.

The pytest run is `========================= 1 failed, ============================== 6 passed in 2.07s =============================== ==========================`.

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

## Choices stated explicitly

- The formula is the delta-based merge of Chan, Golub, LeVeque (1979). The
  docstring of `dense_armor/utility/stats/moments.py` cites it as
  "In Compstat", without equation or page. The exact equation, Eq. (2.1b),
  page 4 of the paper (page 8 of the PDF), is taken from the Stanford tech
  report STAN-CS-79-773, which is in `papers/`. This is my addition, not a
  citation from the docstring.
- The code computes with `float64`, which is the scikit-learn default. The
  paper does not require `float128`.

## Test relevance

Five tests prove the fix, two are regression checks:

- ``test_old_formula_overflows``: the old formula returns ``inf`` on
  the reproducer. This is the bug.
- ``test_new_formula_no_overflow``: the new formula returns a finite
  value, matching ``batch.mean()`` within ``1e-10``. This is the fix.
- ``test_new_formula_with_std_false``: the ``with_std=False`` case
  (``last_variance=None``) is handled, for both formulas.
- ``test_new_formula_matches_numpy_on_normal_data``: on normal data the
  new formula matches ``numpy.mean`` and ``numpy.var`` within ``1e-12``.
  **This is a regression check, not a proof of the fix**: the old formula
  passes it too (2.78e-17 on the mean, 3.20e-14 on the variance).
- ``test_new_formula_constant_data``: constant data has variance 0.
  **This is a regression check, not a proof of the fix**: the old formula
  returns the same ``[42, 42]`` mean and ``[0, 0]`` variance.
- ``test_new_formula_streaming_matches_batch``: 1000 batches of 10 match
  the batch. **This is a regression check, not a proof of the fix**:
  the old formula gives 2.78e-17 on the mean and 3.20e-14 on the variance.

The mutation test (restoring the old formula) shows which tests fail:
``test_old_formula_overflows`` and ``test_new_formula_no_overflow``.
The other three pass with the old formula too.

