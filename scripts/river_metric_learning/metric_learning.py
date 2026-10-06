"""
Online metric learning: OASIS, LEGO and POLA.

OASIS  Chechik, Sharma, Shalit, Bengio, NIPS 2009 / JMLR 2010: bilinear similarity
       S(x, y) = x^T W y learned from triplets (query, positive, negative) with a
       passive-aggressive step. No PSD or symmetry constraint.
LEGO   Jain, Kulis, Dhillon, Grauman, NIPS 2008: Mahalanobis matrix A learned from
       pairs with a target squared distance, LogDet-regularized closed-form update.
POLA   Shalev-Shwartz, Singer, Ng, ICML 2004: pseudo-metric (A, b) learned from
       similar/dissimilar pairs, projected onto the PSD cone and onto b >= 1.
"""
import numpy as np


class OASIS:
    def __init__(self, C=0.01):
        self.C = C
        self.W = None

    def score(self, x, y):
        return float(x @ self.W @ y)

    def learn_triplet(self, x, x_pos, x_neg):
        if self.W is None:
            self.W = np.identity(len(x))
        loss = 1.0 - x @ self.W @ x_pos + x @ self.W @ x_neg
        if loss <= 0.0:
            return
        V = np.outer(x, x_pos - x_neg)
        tau = min(self.C, loss / (np.sum(V * V) + 1e-12))
        self.W = self.W + tau * V


class LEGO:
    def __init__(self, eta=0.1):
        self.eta = eta
        self.A = None

    def distance(self, u, v):
        z = u - v
        return float(z @ self.A @ z)

    def learn_pair(self, u, v, target):
        if self.A is None:
            self.A = np.identity(len(u))
        z = u - v
        y_hat = float(z @ self.A @ z)
        if y_hat <= 0.0:
            return
        eta = self.eta
        a = eta * target * y_hat - 1.0
        y_bar = (a + np.sqrt(a**2 + 4.0 * eta * y_hat**2)) / (2.0 * eta * y_hat)
        Az = self.A @ z
        self.A = self.A - eta * (y_bar - target) * np.outer(Az, Az) / (1.0 + eta * (y_bar - target) * y_hat)


class POLA:
    def __init__(self, b_init=1.0):
        self.A = None
        self.b = b_init

    def distance(self, x, xp):
        v = x - xp
        return float(v @ self.A @ v)

    def learn_pair(self, x, xp, y):
        if self.A is None:
            self.A = np.zeros((len(x), len(x)))
        v = x - xp
        loss = max(0.0, y * (float(v @ self.A @ v) - self.b) + 1.0)
        if loss <= 0.0:
            return
        alpha = loss / (1.0 + float(v @ v) ** 2)
        A = self.A - y * alpha * np.outer(v, v)
        b = self.b + y * alpha
        if y == 1:
            w, U = np.linalg.eigh(A)
            if w[0] < 0:
                A = A - w[0] * np.outer(U[:, 0], U[:, 0])
        self.A = A
        self.b = max(b, 1.0)


def train_oasis(X, y, C=0.01, n_triplets=20000, seed=0):
    model = OASIS(C=C)
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(n_triplets):
        i = rng.integers(n)
        pos = np.where((y == y[i]) & (np.arange(n) != i))[0]
        neg = np.where(y != y[i])[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        model.learn_triplet(X[i], X[rng.choice(pos)], X[rng.choice(neg)])
    return model.W


def lego_targets(X, y, seed=0):
    if len(X) <= 2000:
        D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        same = y[:, None] == y[None, :]
        np.fill_diagonal(same, False)
        off = ~np.eye(len(X), dtype=bool)
        return float(np.percentile(D2[same], 5)), float(np.percentile(D2[~same & off], 95))
    idx = np.random.default_rng(seed).integers(0, len(X), size=(5000, 2))
    D2 = np.sum((X[idx[:, 0]] - X[idx[:, 1]]) ** 2, axis=-1)
    same = y[idx[:, 0]] == y[idx[:, 1]]
    return float(np.percentile(D2[same], 5)), float(np.percentile(D2[~same], 95))


def train_lego(X, y, eta=0.1, n_pairs=20000, seed=0):
    near, far = lego_targets(X, y, seed)
    model = LEGO(eta=eta)
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(n_pairs):
        i, j = rng.integers(n), rng.integers(n)
        if i != j:
            model.learn_pair(X[i], X[j], near if y[i] == y[j] else far)
    return (model.A + model.A.T) / 2


def train_pola(X, y, n_pairs=20000, seed=0):
    model = POLA()
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(n_pairs):
        i, j = rng.integers(n), rng.integers(n)
        if i != j:
            model.learn_pair(X[i], X[j], 1 if y[i] == y[j] else -1)
    return model.A


def mahalanobis(M):
    def d(u, v):
        z = u - v
        return float(np.sqrt(max(z @ M @ z, 1e-12)))
    return d


def oasis_metric(W, mode="WtW"):
    if mode == "WtW":
        return mahalanobis(W.T @ W)
    if mode == "sym":
        return mahalanobis((W + W.T) / 2)
    raise ValueError(mode)
