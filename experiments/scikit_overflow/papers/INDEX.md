# papers/INDEX.md: the references cited in Dense-Armor docstrings

The docstrings of Dense-Armor cite four papers. One is available as a PDF, the
other three are paywalled or unreachable from this session and are declared
"bibliographic only".

## In this folder

| File | Authors, title, venue | Where it is used |
|---|---|---|
| `chan_golub_leveque_1979_stanford.pdf` | Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a pairwise algorithm for computing sample variances*. Stanford CS tech report STAN-CS-79-773. | The parallel merge formula for variance and covariance. |

## Not in this folder (bibliographic only)

| Id | Authors, title, venue | Why not |
|---|---|---|
| Welford 1962 | Welford, B. P. *Note on a method for calculating corrected sums of squares and products*. Technometrics 4(3), 419-420. | Paywalled (JSTOR). Cited bibliographically in `moments.py`. |
| Pébay 2008 | Pébay, P. *Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments*. Sandia Report SAND2008-6212. | The OSTI link returns 404. Cited bibliographically in `moments.py`. |
| West 1979 | West, D. H. D. *Updating mean and variance estimates: an improved method*. Communications of the ACM 22(9), 532-535. | Paywalled (ACM DL). Cited bibliographically in `moments.py`. |

## The formula used in the fix

The fix implements the merge formula of Chan, Golub, and LeVeque (1979),
described in the PDF above:

~~~text
delta = new_mean - last_mean
new_var = (last_var * last_count + new_var * new_count
           + delta**2 * last_count * new_count / total_count) / total_count
~~~

The formula is algebraically equivalent to the cumulative-sum formula, but it
never reconstructs the sum, so it does not overflow.
