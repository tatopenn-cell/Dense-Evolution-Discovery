# Package: PR for scikit-learn issue #5602

This folder is the package for the scikit-learn contribution that fixes the
`StandardScaler.partial_fit` overflow.

## Reading order

1. `README.md` (this file) - what the package is.
2. `ISTRUZIONI.md` - the task, the items, the criteria.
3. `SKILL.md` - the rules of the code for scikit-learn.
4. `04_SELF_CHECK.md` - the checks before the delivery.
5. `05_DELIVERY_TEMPLATE.md` - the form of the delivery.
6. `papers/INDEX.md` - the references (CGL 1979 PDF, Welford/Pebay/West
   bibliographic only).

## The two-stage flow

The work is published in two stages:

- **Stage 1 (this repository):** the case study lives in
  `experiments/scikit_overflow/` on `Dense-Evolution-Discovery`, branch
  `case/scikit-overflow`. It contains the full documentation, the paper
  (CGL 1979), the debug, and the reasoning. The tests (pytest, ruff, mypy,
  mkdocs strict) run here first.
- **Stage 2 (upstream):** only after the checks on `Dense-Evolution-Discovery`
  pass, the branch `fix/standard-scaler-overflow` is pushed to the fork of
  scikit-learn and the PR is opened on `scikit-learn/scikit-learn`.

The PR on scikit-learn is small: the fix in `_incremental_mean_and_var`, the
non-regression test, the changelog entry. The deep explanation, the paper, the
debug, and the full task live here.

## The four kinds of file

| Kind | Files | What they hold |
|---|---|---|
| Harness | `README.md`, `01_FOLDER_ARCHITECTURE.md` | what this package is, the reading order |
| Protocols | `ISTRUZIONI.md`, `04_SELF_CHECK.md`, `05_DELIVERY_TEMPLATE.md` | the task, the checks, the delivery |
| Skills | `SKILL.md` | the rules of the code |
| Memory | `papers/INDEX.md` | the references |

## The one-sentence version

Fix the overflow in `_incremental_mean_and_var`, prove it with a non-regression
test, publish the full case study here, and only then open the PR upstream.
