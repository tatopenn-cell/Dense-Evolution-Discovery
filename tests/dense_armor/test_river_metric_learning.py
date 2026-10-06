import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "river_metric_learning"))
from metric_learning import LEGO, OASIS, POLA, lego_targets, oasis_metric, train_lego, train_oasis, train_pola  # noqa: E402


def blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0, 0, 0], [0.3, 3.0, 3.0], (n, 3)), rng.normal([2, 0, 0], [0.3, 3.0, 3.0], (n, 3))])
    return X, np.array([0] * n + [1] * n)


def test_oasis_no_update_when_margin_satisfied():
    m = OASIS(C=1.0)
    x, xp, xn = np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 1.0])
    m.learn_triplet(x, xp, xn)
    assert np.allclose(m.W, np.identity(2))


def test_oasis_step_reduces_loss():
    m = OASIS(C=100.0)
    x, xp, xn = np.array([1.0, 0.0]), np.array([0.2, 0.0]), np.array([0.5, 0.0])
    m.W = np.identity(2)
    before = 1.0 - m.score(x, xp) + m.score(x, xn)
    m.learn_triplet(x, xp, xn)
    after = 1.0 - m.score(x, xp) + m.score(x, xn)
    assert before > 0 and after <= 1e-9


def test_pola_stays_psd_and_b_at_least_one():
    X, y = blobs()
    m = POLA()
    rng = np.random.default_rng(1)
    for _ in range(500):
        i, j = rng.integers(len(X)), rng.integers(len(X))
        if i != j:
            m.learn_pair(X[i], X[j], 1 if y[i] == y[j] else -1)
            assert np.linalg.eigvalsh(m.A)[0] >= -1e-9
            assert m.b >= 1.0


def test_lego_moves_distance_toward_target():
    m = LEGO(eta=0.1)
    u, v = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    m.A = np.identity(2)
    m.learn_pair(u, v, 4.0)
    assert 1.0 < m.distance(u, v) < 4.0
    m.learn_pair(u, v, 0.1)
    assert m.distance(u, v) < 4.0


def test_lego_targets_order():
    X, y = blobs()
    near, far = lego_targets(X, y)
    assert 0 < near < far


def test_learned_metrics_weight_the_informative_feature():
    X, y = blobs()
    for M in (train_pola(X, y, n_pairs=2000), train_lego(X, y, n_pairs=2000)):
        assert M[0, 0] > max(M[1, 1], M[2, 2])
    W = train_oasis(X, y, C=0.1, n_triplets=2000)
    G = W.T @ W
    assert np.linalg.eigvalsh(G)[0] >= -1e-9


def test_oasis_metric_modes():
    W = np.array([[1.0, 2.0], [0.0, 1.0]])
    u, v = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    assert oasis_metric(W, "WtW")(u, v) == np.sqrt(1.0)
    assert oasis_metric(W, "sym")(u, v) == np.sqrt(1.0)
