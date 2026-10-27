# SKILL.md: the rules of the code for scikit-learn

## 1. Goal

scikit-learn is a machine learning library. The change is in the numerical
stability of `StandardScaler.partial_fit`, which is used by every stream
pipeline.

## 2. Interface

Do not change the public signature of `StandardScaler.partial_fit`,
`_incremental_mean_and_var`, or any other public function. The fix is internal.
The output must match `fit()` within `1e-10`.

## 3. Correctness

The Welford update (Chan, Golub, LeVeque 1979) is the reference. It replaces the
cumulative-sum formula with a delta-based formula. The delta-based formula is
algebraically equivalent but never reconstructs the cumulative sum.

- Welford 1962: bibliographic only, not in the folder.
- Chan-Golub-LeVeque 1979: bibliographic only, not in the folder.
- Issue #5602: the reproduction and the analysis, with the page numbers.

## 4. Style

- PEP 8, ruff format, ruff check.
- NumPy-style docstrings (scikit-learn uses this, not Google style).
- Type hints on every new function.
- No inline comments: explain in the docstring.
- No new dependency.

## 5. Tests

- Every new function or new branch has a test.
- The non-regression test must fail on `main` and pass on the branch. Prove it
  by running it on both.
- The existing test suite in `sklearn/utils/tests/test_extmath.py` must pass.
- Coverage of the touched file must not decrease.

## 6. Docs and PR

- Changelog entry in `doc/whats_new/upcoming_changes/<PR_NUMBER>.fix.rst`.
- PR template with the "AI usage disclosure" section.
- The AI usage is declared: the code was written with AI assistance, the author
  understands every line and can explain it.

## The PR checklist (from scikit-learn official docs)

1. Title summarises the change.
2. Code passes tests.
3. Code is documented.
4. Tests are necessary for enhancements.
5. No PEP 8 violations.
6. Follow coding guidelines.
7. Bug fix: non-regression test fails on main, passes with the PR.
8. Benchmarks or tests substantiate the change.
9. Changelog entry.
10. AI usage disclosure (mandatory since 2025).
