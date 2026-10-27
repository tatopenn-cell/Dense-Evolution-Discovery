# Package: PR for scikit-learn issue #5602

This folder is the package for the scikit-learn contribution that fixes the
`StandardScaler.partial_fit` overflow.

## Reading order

1. `README.md` (this file) - what the package is.
2. `ISTRUZIONI.md` - the task, the items, the criteria.
3. `SKILL.md` - the rules of the code for scikit-learn.
4. `04_SELF_CHECK.md` - the checks before the delivery.
5. `05_DELIVERY_TEMPLATE.md` - the form of the delivery.

## The four kinds of file

| Kind | Files | What they hold |
|---|---|---|
| Harness | `README.md`, `01_FOLDER_ARCHITECTURE.md` | what this package is, the reading order |
| Protocols | `ISTRUZIONI.md`, `04_SELF_CHECK.md`, `05_DELIVERY_TEMPLATE.md` | the task, the checks, the delivery |
| Skills | `SKILL.md` | the rules of the code |
| Memory | `papers/INDEX.md` | the references, bibliographic only |

## The one-sentence version

Fix the overflow in `_incremental_mean_and_var`, add a non-regression test that
fails on `main` and passes with the fix, open the PR, declare the AI usage.
