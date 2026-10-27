# 04_SELF_CHECK: what the agent checks before the PR

## 1. The non-regression test fails on main
Run the new test on a fresh clone of main. It must fail (the mean is `inf`).
Paste the failure.

## 2. The non-regression test passes on the branch
Run the same test on the branch. It must pass with the new fix.

## 3. The existing test suite passes
Run `pytest sklearn/utils/tests/test_extmath.py -q` on the branch. It must pass
in full, including the tests that existed before.

## 4. No public signature changed
Run `git diff main -- sklearn/`. Only the internal `_incremental_mean_and_var`
and the new test file are allowed. Any other change is out of scope.

## 5. The AI usage is declared
The PR description has the "AI usage disclosure" section. The author can explain
every line. If the section is missing, the PR will be closed without review.

## 6. The changelog entry exists
The file `doc/whats_new/upcoming_changes/<PR_NUMBER>.fix.rst` exists and
references issue #5602.

## 7. The ruff check passes
`ruff check sklearn/utils/extmath.py sklearn/utils/tests/test_extmath.py` must
return "All checks passed".

## 8. The mypy check passes
`mypy sklearn/utils/extmath.py --follow-imports=skip` must return "Success: no
issues found".

## 9. No path that exists on one machine
`grep -rn "/content/" sklearn/utils/extmath.py sklearn/utils/tests/test_extmath.py`
must return nothing.

## 10. No pragma added
`grep -rn "pragma: no cover" sklearn/utils/extmath.py sklearn/utils/tests/test_extmath.py`
must return nothing.

## 11. The PR template is filled
Sections: Reference Issues/PRs (Fixes #5602), What does this implement/fix,
Introduce yourself, AI usage disclosure, Any other comments.
