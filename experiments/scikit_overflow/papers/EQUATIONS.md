# papers/EQUATIONS.md: the equations used in the fix

The fix implements the delta-based update of Chan, Golub, and LeVeque (1979),
as cited in the docstring of `_incremental_mean_and_var` (Dense-Armor,
`dense_armor/utility/stats/moments.py`, lines 22-25).

## The formula (CGL 1979, merge formula)

Given the state before the batch (`last_mean`, `last_var`, `last_count`) and the
batch (`new_mean`, `new_var`, `new_count`):

~~~text
total_count = last_count + new_count
delta       = new_mean - last_mean
new_mean_c  = last_mean + delta * new_count / total_count
new_var_c   = (last_var * last_count
               + new_var * new_count
               + delta**2 * last_count * new_count / total_count) / total_count
~~~

The formula never reconstructs `last_sum` and `new_sum`. This is the property
that prevents the overflow on long streams.

## The reproduction (from issue #5602)

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

## Sources (bibliographic only, no PDF in this folder)

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
  pairwise algorithm for computing sample variances*. In Compstat.
- Welford, B. P. (1962). *Note on a method for calculating corrected sums of
  squares and products*. Technometrics 4(3), 419-420.
