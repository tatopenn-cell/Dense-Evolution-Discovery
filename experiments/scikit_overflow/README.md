# scikit-learn issue #5602: StandardScaler.partial_fit overflow

This folder documents the fix of `StandardScaler.partial_fit`, which overflows
to `inf` on long streams or large values (issue #5602, opened in 2015).

## Files

- `incremental.py`: the old formula and the fixed formula. The fixed formula
  is the delta-based update of Chan, Golub, LeVeque (1979).
- `test_incremental.py`: five tests. The old formula overflows; the new one
  matches the batch reference, on normal data, on constant data, and on
  streaming.
- `CASE_STUDY.md`: the full analysis (bug, fix, tests, numbers, references).
- `papers/chan_golub_leveque_1979_stanford.pdf`: the source (STAN-CS-79-773).

## Run

~~~bash
python -m pytest test_incremental.py -v
~~~

## Status

The fix passes the tests on this branch. The upstream PR will be opened only
after the checks here are green, with a link back to this folder.
