import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_digits, load_iris, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from metric_learning import train_lego, train_oasis, train_pola


def knn_accuracy(M, X_tr, y_tr, X_te, y_te, k):
    def dist(A, B):
        d2 = np.einsum("ij,jk,ik->i", A, M, A)[:, None] + np.einsum("ij,jk,ik->i", B, M, B)[None, :] - 2 * A @ M @ B.T
        return np.sqrt(np.maximum(d2, 1e-12))
    knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed").fit(dist(X_tr, X_tr), y_tr)
    return knn.score(dist(X_te, X_tr), y_te)


def dense_table(rows):
    sets = [("Iris", load_iris, 20000, 20000), ("Wine", load_wine, 30000, 30000), ("Digits", load_digits, 50000, 20000)]
    for name, loader, n_main, n_pola in sets:
        X, y = loader(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
        k = 3
        I = np.identity(X.shape[1])
        W = train_oasis(X_tr, y_tr, C=0.01, n_triplets=n_main)
        res = {
            "euclidean": knn_accuracy(I, X_tr, y_tr, X_te, y_te, k),
            "OASIS (WtW)": knn_accuracy(W.T @ W, X_tr, y_tr, X_te, y_te, k),
            "LEGO": knn_accuracy(train_lego(X_tr, y_tr, eta=0.1, n_pairs=n_main), X_tr, y_tr, X_te, y_te, k),
            "POLA": knn_accuracy(train_pola(X_tr, y_tr, n_pairs=n_pola), X_tr, y_tr, X_te, y_te, k),
        }
        print(name, "  ".join(f"{m}: {a:.4f}" for m, a in res.items()))
        rows += [(name, m, "", a) for m, a in res.items()]


def text_table(rows):
    pairs = [("sci.space", "comp.graphics"), ("comp.sys.ibm.pc.hardware", "sci.electronics")]
    for cats in pairs:
        data = fetch_20newsgroups(subset="all", categories=list(cats), remove=("headers", "footers", "quotes"))
        X = CountVectorizer(max_features=500, min_df=5, binary=True).fit_transform(data.data).toarray().astype(float)
        X = normalize(X, axis=1, norm="l2")
        y = np.array(data.target)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        name = " vs ".join(cats)
        base = knn_accuracy(np.identity(X.shape[1]), X_tr, y_tr, X_te, y_te, 5)
        print(name, f"euclidean {base:.4f}")
        rows.append((name, "euclidean", "", base))
        for C in (0.01, 0.1):
            W = train_oasis(X_tr, y_tr, C=C, n_triplets=20000)
            sym = knn_accuracy((W + W.T) / 2, X_tr, y_tr, X_te, y_te, 5)
            wtw = knn_accuracy(W.T @ W, X_tr, y_tr, X_te, y_te, 5)
            print(f"  C={C}: sym {sym:.4f}  WtW {wtw:.4f}")
            rows += [(name, "OASIS sym", C, sym), (name, "OASIS WtW", C, wtw)]


def main():
    rows = []
    dense_table(rows)
    text_table(rows)
    out = Path("data/river_metric_learning.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        csv.writer(f).writerows([("dataset", "method", "C", "accuracy"), *rows])


if __name__ == "__main__":
    main()
