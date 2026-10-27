# Task T1: fix the overflow in StandardScaler.partial_fit

Fix the overflow in `_incremental_mean_and_var` (scikit-learn, issue #5602),
add a non-regression test that fails on main and passes with the fix, open the
PR with the AI usage disclosed.

Follow `SKILL.md`. Code: a fresh fork of `scikit-learn/scikit-learn`, branch
`main`, new branch `fix/standard-scaler-overflow`.
One branch, as many commits as you need, one commit per item.
At the end, push `fix/standard-scaler-overflow` to your fork and open the PR on
`scikit-learn/scikit-learn`. Do not merge. Do not push to `main`.

## State of the base

The current implementation of `_incremental_mean_and_var` (in
`sklearn/utils/extmath.py`) reconstructs the cumulative sum for the mean and the
variance. On long streams or large values, the cumulative sum overflows
(`inf`), and the variance is wrong. The reproduction is in `papers/INDEX.md`.

Do not touch: the public signatures of `StandardScaler`, `_incremental_mean_and_var`,
and every other public function. Do not add a new dependency. Do not change the
default of any parameter.

## A. Non-regression test

1. `sklearn/utils/tests/test_extmath.py`: add two tests.

   `test_incremental_mean_and_var_large_values_no_overflow`: 100 batches of
   500,000 samples at `1.79e301`. Asserts that `partial_fit` matches `fit()`
   within `1e-10` and that the output is finite.

   `test_incremental_mean_and_var_constant_values`: 1000 samples at `42.0`.
   Asserts that `partial_fit` matches `fit()` within `1e-12` on the mean and
   within `1e-15` on the variance.

2. Run both tests on `main`. They must fail (the first with `inf`). Paste the
   failure.

Criterion: on the branch, `pytest sklearn/utils/tests/test_extmath.py -q`
passes. On `main`, the new test fails.

## B. The fix

1. `sklearn/utils/extmath.py`: replace the variance formula that reconstructs
   the cumulative sum with the delta-based Chan-Golub-LeVeque formula:
   `delta = new_mean - last_mean`,
   `new_var = (last_var * last_count + new_var * new_count + delta**2 * last_count * new_count / total_count) / total_count`.
   Apply the same update to the mean. Maintain compatibility with
   `sample_weight` and `NaN`.

2. Do not break the existing tests in `sklearn/utils/tests/test_extmath.py`.

Criterion: `pytest sklearn/utils/tests/test_extmath.py -q` passes with the new
tests in green.

## C. Changelog and PR

1. `doc/whats_new/upcoming_changes/<PR_NUMBER>.fix.rst`: create the file with
   the scikit-learn format, referencing issue #5602.

2. Open the PR. In the description, declare the AI usage in the "AI usage
   disclosure" section. Tag `@lorentzenchr` and `@EdenRochmanSharabi`.

Criterion: the PR is open, the CI on the PR is green or in progress. The link
is in the delivery.

## Delivery

The table of the items (A1, B1, C1, C2), done / not done / stopped with the
reason. The files touched. The real output of pytest on the branch and the
failure on main. The link to the PR. The holes, one per line. The last line:
the name of the pushed branch and its last commit.
