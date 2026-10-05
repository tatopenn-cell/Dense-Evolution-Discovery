
from __future__ import annotations

import copy
import math
import random
from collections import deque

import numpy as np

from river import base, datasets, linear_model, metrics, optim, preprocessing, tree


def _jensen_shannon(a, b, n_bins=None, pseudo=1.0):
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi - lo < 1e-12:
        return 0.0
    if n_bins is None:
        n_bins = max(3, min(10, (a.size + b.size) // 6))
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)
    p = p.astype(float) + pseudo
    q = q.astype(float) + pseudo
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))


def _get_lr(optimizer):
    sched = optimizer.lr
    for attr in ("learning_rate", "rate", "_lr"):
        if hasattr(sched, attr):
            return float(getattr(sched, attr))
    return float(sched)


def _set_lr(optimizer, new_lr):
    sched = optimizer.lr
    for attr in ("learning_rate", "rate", "_lr"):
        if hasattr(sched, attr):
            setattr(sched, attr, new_lr)
            return


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE = somma pesata |avg(p) - avg(y)| sui bin."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_p = float(probs[mask].mean())
        avg_y = float(labels[mask].mean())
        ece += (count / n) * abs(avg_p - avg_y)
    return ece


class OnlinePlattScaling(base.Wrapper[base.Classifier], base.Classifier):
    def __init__(
        self,
        classifier,
        optimizer=None,
        clip=0.01,
        warmup=200,
        freeze_base=False,
        use_jsd_spring=False,
        jsd_k=3.0,
        jsd_window=30,
        use_ensemble=False,
        ensemble_lrs=(0.01, 0.1, 1.0),
        ensemble_window=100,
        auto_calibrate=False,
        ece_threshold=0.05,
        ece_n_bins=10,
    ):
        self.classifier = classifier
        self.optimizer = optimizer if optimizer is not None else optim.SGD(0.1)
        self.clip = clip
        self.warmup = warmup
        self.freeze_base = freeze_base
        self.use_jsd_spring = use_jsd_spring
        self.jsd_k = jsd_k
        self.jsd_window = jsd_window
        self.use_ensemble = use_ensemble
        self.ensemble_lrs = ensemble_lrs
        self.ensemble_window = ensemble_window
        self.auto_calibrate = auto_calibrate
        self.ece_threshold = ece_threshold
        self.ece_n_bins = ece_n_bins

        if use_ensemble:
            self._calibrators = [
                linear_model.LogisticRegression(
                    optimizer=optim.SGD(lr),
                    intercept_init=0.0,
                    intercept_lr=0.0,
                )
                for lr in ensemble_lrs
            ]
            self._base_lrs = list(ensemble_lrs)
            self._errors = [deque(maxlen=ensemble_window) for _ in ensemble_lrs]
            self._calibrator = None
        else:
            self._calibrators = None
            self._calibrator = linear_model.LogisticRegression(
                optimizer=self.optimizer,
                intercept_init=0.0,
                intercept_lr=0.0,
            )
            self._base_lr = _get_lr(self.optimizer)

        self._n = 0
        self._p1_buffer = deque(maxlen=2 * jsd_window)

        self._probe_p1 = []
        self._probe_y = []
        self._calibrate_active = not auto_calibrate
        self._decision_made = not auto_calibrate
        self._ece_at_decision = None
        self._debug_prints = False

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return False

    @property
    def is_calibrating(self):
        return self._calibrate_active

    @property
    def ece_at_decision(self):
        return self._ece_at_decision

    def _feats(self, x, **kwargs):
        p = self.classifier.predict_proba_one(x, **kwargs)
        p1 = p.get(True, 0.5)
        p1 = min(max(p1, self.clip), 1.0 - self.clip)
        z = math.log(p1 / (1.0 - p1))
        return {"logit": z, "bias": 1.0}, p1

    def _compute_jsd(self):
        if len(self._p1_buffer) < 2 * self.jsd_window:
            return 0.0
        arr = np.array(self._p1_buffer)
        recent = arr[-self.jsd_window:]
        old = arr[: self.jsd_window]
        return _jensen_shannon(recent, old)

    def _blue_weights(self):
        scales = []
        for k in range(len(self._calibrators)):
            if self._errors[k]:
                rmse = math.sqrt(sum(self._errors[k]) / len(self._errors[k]))
            else:
                rmse = 1.0
            scales.append(max(rmse, 1e-6))
        weights = [1.0 / (s * s) for s in scales]
        total = sum(weights)
        return [w / total for w in weights]

    def _predict_ensemble(self, feats, p1):
        probas = []
        for cal in self._calibrators:
            proba = cal.predict_proba_one(feats)
            probas.append(p1 if not proba else proba.get(True, p1))
        weights = self._blue_weights()
        p1_out = sum(w * p for w, p in zip(weights, probas))
        return 1.0 - p1_out, p1_out

    def _make_decision(self):
        ece = expected_calibration_error(
            self._probe_p1, self._probe_y, self.ece_n_bins
        )
        self._ece_at_decision = ece
        self._calibrate_active = ece > self.ece_threshold
        self._decision_made = True
        if self._debug_prints:
            verdict = "ATTIVO" if self._calibrate_active else "DISATTIVO"
            print(f"    [auto] ECE={ece:.4f} soglia={self.ece_threshold:.4f} -> OPS {verdict}")

    def predict_proba_one(self, x, **kwargs):
        feats, p1 = self._feats(x, **kwargs)

        if self._n < self.warmup:
            return {False: 1.0 - p1, True: p1}

        if not self._calibrate_active:
            return {False: 1.0 - p1, True: p1}

        if self.use_ensemble:
            p0, p1_out = self._predict_ensemble(feats, p1)
        else:
            proba = self._calibrator.predict_proba_one(feats)
            p1_out = p1 if not proba else proba.get(True, p1)
            p0 = 1.0 - p1_out
        return {False: p0, True: p1_out}

    def learn_one(self, x, y, **kwargs):
        feats, p1 = self._feats(x, **kwargs)
        self._p1_buffer.append(p1)

        if self.auto_calibrate and not self._decision_made:
            self._probe_p1.append(p1)
            self._probe_y.append(float(bool(y)))

        lr_mult = 1.0
        if self.use_jsd_spring:
            jsd = self._compute_jsd()
            lr_mult = 1.0 + self.jsd_k * jsd

        y_float = float(bool(y))

        if self.use_ensemble:
            for k, cal in enumerate(self._calibrators):
                if self.use_jsd_spring:
                    _set_lr(cal.optimizer, self._base_lrs[k] * lr_mult)
                proba = cal.predict_proba_one(feats)
                p1_k = proba.get(True, 0.5) if proba else 0.5
                self._errors[k].append((y_float - p1_k) ** 2)
                cal.learn_one(feats, y)
        else:
            if self.use_jsd_spring:
                _set_lr(self._calibrator.optimizer, self._base_lr * lr_mult)
            self._calibrator.learn_one(feats, y)

        self._n += 1

        if self.auto_calibrate and not self._decision_made and self._n >= self.warmup:
            self._make_decision()

        if not self.freeze_base:
            self.classifier.learn_one(x, y, **kwargs)

import copy
from river import datasets, metrics, tree


