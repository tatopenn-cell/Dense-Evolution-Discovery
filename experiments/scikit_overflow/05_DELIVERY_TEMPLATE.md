# 05_DELIVERY_TEMPLATE: the form of the delivery

## 1. Header

# Task T1: fix the overflow in StandardScaler.partial_fit — delivery

Branch `fix/standard-scaler-overflow`, N commits above `main`, last commit `<hash>`.
PR: <link>.
State: green / red.

## 2. Table of the items

| Item | State | How it is checked |
|---|---|---|
| A1 | done | `pytest sklearn/utils/tests/test_extmath.py -q` on branch -> N passed |
| A2 | done | the same test on main -> fail (mean is `inf`) |
| B1 | done | the new tests in green; the old tests still pass |
| C1 | done | `doc/whats_new/upcoming_changes/<N>.fix.rst` exists |
| C2 | done | PR opened, CI in progress |

## 3. Files

New: `doc/whats_new/upcoming_changes/<N>.fix.rst`.
Modified: `sklearn/utils/extmath.py` (the delta-based update),
`sklearn/utils/tests/test_extmath.py` (two new tests).
Out of scope, declared: none.

## 4. Output of the checks

~~~text
$ pytest sklearn/utils/tests/test_extmath.py -q
<N> passed

$ ruff check sklearn/utils/extmath.py sklearn/utils/tests/test_extmath.py
All checks passed!

$ mypy sklearn/utils/extmath.py --follow-imports=skip
Success: no issues found in 1 source file

$ the new test on main
<N> failed (mean is inf)
~~~

## 5. Self-check of `04_SELF_CHECK.md`

Each of the eleven checks, with its output.

## 6. What does not work, declared

One line per hole. If the CI of the PR is in progress, it is declared. If a
Windows test fails, it is declared with the exact error.

## Last line

Branch `fix/standard-scaler-overflow`, last commit `<hash>`, PR `<link>`.
