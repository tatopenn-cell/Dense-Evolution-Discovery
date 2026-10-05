import copy
import csv
import random
import sys
from pathlib import Path

from river import base, datasets, metrics, optim, tree
from river.datasets import synth

sys.path.insert(0, str(Path(__file__).parent))
import jsd_ops
import ops
import tops


class Frozen(base.Wrapper, base.Classifier):
    def __init__(self, c):
        self.c = c

    @property
    def _wrapped_model(self):
        return self.c

    def predict_proba_one(self, x, **k):
        return self.c.predict_proba_one(x)

    def learn_one(self, x, y, **k):
        pass


DATASETS = {
    "Phishing": datasets.Phishing,
    "Elec2": datasets.Elec2,
    "Bananas": datasets.Bananas,
    "SEA drift": lambda: synth.ConceptDriftStream(synth.SEA(variant=0, seed=1), synth.SEA(variant=3, seed=2), seed=3, position=2500, width=500),
    "Agrawal drift": lambda: synth.ConceptDriftStream(synth.Agrawal(classification_function=0, seed=1), synth.Agrawal(classification_function=4, seed=2), seed=3, position=2500, width=500),
    "Hyperplane": lambda: synth.Hyperplane(seed=1, n_features=10, n_drift_features=5, mag_change=0.01),
    "SINE drift": lambda: synth.ConceptDriftStream(synth.Sine(classification_function=0, seed=1), synth.Sine(classification_function=2, seed=2), seed=3, position=2500, width=500),
}

E = dict(use_ensemble=True, ensemble_lrs=(0.01, 0.1, 1.0), ensemble_window=100)
J = dict(use_jsd_spring=True, jsd_k=3.0, jsd_window=30)
A = dict(auto_calibrate=True, ece_threshold=0.05)


def run(m, data):
    acc, ll, brier = metrics.Accuracy(), metrics.LogLoss(), 0.0
    for x, y in data:
        p = m.predict_proba_one(x).get(True, 0.5)
        acc.update(y, p > 0.5)
        ll.update(y, min(max(p, 1e-3), 1 - 1e-3))
        brier += (y - p) ** 2
        m.learn_one(x, y)
    return acc.get(), ll.get(), brier / len(data)


def frozen_tree(data):
    random.seed(42)
    fb = tree.HoeffdingTreeClassifier()
    for x, y in random.sample(data[:2500], 500):
        fb.learn_one(x, y)
    return fb


def yours(fb, **kw):
    return jsd_ops.OnlinePlattScaling(copy.deepcopy(fb), warmup=500, freeze_base=True, **kw)


def main():
    rows = []
    for name, mk in DATASETS.items():
        data = [(x, bool(y)) for x, y in mk().take(5000)]
        fb = frozen_tree(data)
        models = [
            ("base", Frozen(copy.deepcopy(fb))),
            ("OPS", ops.OnlinePlattScaling(Frozen(copy.deepcopy(fb)))),
            ("TOPS", tops.TrackingOnlinePlattScaling(Frozen(copy.deepcopy(fb)))),
            ("JSD+BLUE+auto", yours(fb, **E, **J, **A)),
        ]
        if name == "Elec2":
            models += [
                ("SGD 0.1 only", yours(fb)),
                ("SGD 1.0 only", yours(fb, optimizer=optim.SGD(1.0))),
                ("ensemble only", yours(fb, **E)),
                ("JSD only", yours(fb, **J)),
                ("auto only", yours(fb, **A)),
                ("ensemble+JSD", yours(fb, **E, **J)),
            ]
        for label, m in models:
            a, l, b = run(m, data)
            rows.append((name, label, a, l, b))
            print(f"{name:14} {label:15} acc {a:.4f}  logloss {l:.4f}  brier {b:.4f}")
    out = Path("data/river_platt_scaling.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        csv.writer(f).writerows([("dataset", "model", "accuracy", "logloss", "brier"), *rows])


if __name__ == "__main__":
    main()
