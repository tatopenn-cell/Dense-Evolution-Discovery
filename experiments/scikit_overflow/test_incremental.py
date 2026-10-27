"""Tests for the incremental mean and variance fix."""

import numpy as np

from incremental import incremental_mean_and_var_new, incremental_mean_and_var_old


def _run_batches(fn, batches, with_variance=True):
    first = batches[0]
    n_features = first.shape[1] if first.ndim > 1 else 1
    mean = np.zeros(n_features)
    var = np.zeros(n_features) if with_variance else None
    count = np.zeros(n_features)
    for batch in batches:
        mean, var, count = fn(batch, mean, var, count)
    return mean, var, count


def test_old_formula_overflows():
    """The old cumulative-sum formula returns inf on large values."""
    rng = np.random.RandomState(0)
    max_f = np.finfo(np.float64).max / 1e5
    min_f = max_f / 1e2
    batch = rng.uniform(min_f, min_f, size=(500000, 1))

    mean, _, _ = _run_batches(incremental_mean_and_var_old, [batch] * 100)
    assert not np.isfinite(mean).all()


def test_new_formula_no_overflow():
    """The delta-based formula returns a finite value and matches the batch mean."""
    rng = np.random.RandomState(0)
    max_f = np.finfo(np.float64).max / 1e5
    min_f = max_f / 1e2
    batch = rng.uniform(min_f, min_f, size=(500000, 1))

    mean, _, _ = _run_batches(incremental_mean_and_var_new, [batch] * 100)
    expected = batch.mean(axis=0)
    assert np.isfinite(mean).all()
    assert np.allclose(mean, expected, rtol=1e-10)


def test_new_formula_matches_numpy_on_normal_data():
    """On normal data, the delta-based formula matches numpy."""
    rng = np.random.RandomState(1)
    X = rng.randn(10000, 3) * 15 + 100

    mean, var, count = _run_batches(incremental_mean_and_var_new, [X])
    assert np.allclose(mean, X.mean(axis=0), rtol=1e-12)
    assert np.allclose(var, X.var(axis=0), rtol=1e-12)
    assert np.allclose(count, X.shape[0])


def test_new_formula_constant_data():
    """Constant data: variance 0, no NaN, no division by zero."""
    X = np.ones((1000, 2)) * 42.0
    batches = [X[i : i + 10] for i in range(0, 1000, 10)]

    mean, var, count = _run_batches(incremental_mean_and_var_new, batches)
    assert np.allclose(mean, 42.0)
    assert np.allclose(var, 0.0)
    assert np.allclose(count, 1000)


def test_new_formula_streaming_matches_batch():
    """Streaming 1000 batches of 10 matches the batch over the same data."""
    rng = np.random.RandomState(2)
    X = rng.randn(10000, 2) * 5
    batches = [X[i : i + 10] for i in range(0, 10000, 10)]

    mean, var, _ = _run_batches(incremental_mean_and_var_new, batches)
    assert np.allclose(mean, X.mean(axis=0), rtol=1e-10)
    assert np.allclose(var, X.var(axis=0), rtol=1e-10)


def test_new_formula_with_std_false():
    """The function handles ``last_variance=None`` (the ``with_std=False`` case).

    ``StandardScaler(with_std=False)`` calls ``_incremental_mean_and_var``
    with ``last_variance=None``. The test covers the ``if last_variance is
    None`` branch of both the old and the new formula.
    """
    X = np.ones((1000, 2)) * 42.0
    batches = [X[i : i + 10] for i in range(0, 1000, 10)]

    mean_old, var_old, count_old = _run_batches(
        incremental_mean_and_var_old, batches, with_variance=False
    )
    mean_new, var_new, count_new = _run_batches(
        incremental_mean_and_var_new, batches, with_variance=False
    )

    assert np.allclose(mean_old, 42.0)
    assert var_old is None
    assert np.allclose(mean_new, 42.0)
    assert var_new is None
    assert np.allclose(count_old, 1000)
    assert np.allclose(count_new, 1000)

