"""Incremental mean and variance: the old formula and the fixed one.

The old formula reconstructs the cumulative sum and overflows on long streams
with large values. The fixed formula uses the delta-based update of Chan, Golub
and LeVeque (1979), which never reconstructs the sum.
"""

import numpy as np


def incremental_mean_and_var_old(X, last_mean, last_variance, last_sample_count):
    """The cumulative-sum formula (the one that overflows).

    The mean is computed as ``(last_sum + new_sum) / updated_sample_count``
    with ``last_sum = last_mean * last_sample_count``. The variance uses
    the same Eq. (2.1b) of the paper, written with the cumulative sums.
    On long streams with large values, ``last_sum`` overflows to ``inf``.
    The PDF is in ``papers/chan_golub_leveque_1979_stanford.pdf``
    (STAN-CS-79-773).
    """
    X = np.asarray(X, dtype=np.float64)
    last_mean = np.asarray(last_mean, dtype=np.float64)
    last_sample_count = np.asarray(last_sample_count, dtype=np.float64)
    if last_variance is not None:
        last_variance = np.asarray(last_variance, dtype=np.float64)

    last_sum = last_mean * last_sample_count
    X_nan_mask = np.isnan(X)
    new_sum = np.nansum(X, axis=0)
    n_samples = X.shape[0]
    new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)

    updated_sample_count = last_sample_count + new_sample_count
    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        return updated_mean, None, updated_sample_count

    T = new_sum / new_sample_count
    temp = X - T
    correction = np.nansum(temp, axis=0)
    temp **= 2
    new_unnormalized_variance = np.nansum(temp, axis=0)
    new_unnormalized_variance -= correction**2 / new_sample_count

    last_unnormalized_variance = last_variance * last_sample_count

    last_over_new_count = last_sample_count / new_sample_count
    updated_unnormalized_variance = (
        last_unnormalized_variance
        + new_unnormalized_variance
        + last_over_new_count
        / updated_sample_count
        * (last_sum / last_over_new_count - new_sum) ** 2
    )

    zeros = last_sample_count == 0
    updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
    updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def incremental_mean_and_var_new(X, last_mean, last_variance, last_sample_count):
    """The delta-based formula (Chan, Golub, LeVeque 1979).

    Mean update (Eq. 2.1a, page 4 of the paper, PDF page 8):
        T_{1,m+n} = T_{1,m} + T_{m+1,m+n}

    Variance merge (Eq. 2.1b, page 4 of the paper, PDF page 8):
        S_{1,m+n} = S_{1,m} + S_{m+1,m+n}
                    + (m*n)/(m+n) * (T_{1,m}/m - T_{m+1,m+n}/n)**2

    Corrected two-pass term (Eq. 1.4, page 2 of the paper, PDF page 6):
        S = sum((x_i - x_bar)**2) - (sum(x_i - x_bar))**2 / N

    The PDF is in ``papers/chan_golub_leveque_1979_stanford.pdf``
    (STAN-CS-79-773).
    """
    X = np.asarray(X, dtype=np.float64)
    last_mean = np.asarray(last_mean, dtype=np.float64)
    last_sample_count = np.asarray(last_sample_count, dtype=np.float64)
    if last_variance is not None:
        last_variance = np.asarray(last_variance, dtype=np.float64)

    X_nan_mask = np.isnan(X)
    new_sum = np.nansum(X, axis=0)
    n_samples = X.shape[0]
    new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)

    updated_sample_count = last_sample_count + new_sample_count

    safe_new_count = np.where(new_sample_count > 0, new_sample_count, 1)
    T = new_sum / safe_new_count
    delta = T - last_mean
    updated_mean = last_mean + delta * new_sample_count / updated_sample_count

    if last_variance is None:
        return updated_mean, None, updated_sample_count

    temp = X - T
    correction = np.nansum(temp, axis=0)
    temp **= 2
    new_unnormalized_variance = np.nansum(temp, axis=0)
    new_unnormalized_variance -= correction**2 / new_sample_count

    last_unnormalized_variance = last_variance * last_sample_count

    updated_unnormalized_variance = (
        last_unnormalized_variance
        + new_unnormalized_variance
        + delta**2 * last_sample_count * new_sample_count / updated_sample_count
    )

    zeros = last_sample_count == 0
    updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
    updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count
