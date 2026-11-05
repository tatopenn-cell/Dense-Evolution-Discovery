from ops import OnlinePlattScaling


class TrackingOnlinePlattScaling(OnlinePlattScaling):
    def __init__(self, classifier, n_bins: int = 10, **kwargs):
        super().__init__(classifier, **kwargs)
        self.n_bins = n_bins
        self._sum = [0.0] * n_bins
        self._count = [0] * n_bins

    def _bin(self, x, **kwargs):
        p = super().predict_proba_one(x, **kwargs)[True]
        return min(int(p * self.n_bins), self.n_bins - 1)

    def predict_proba_one(self, x, **kwargs):
        b = self._bin(x, **kwargs)
        p = self._sum[b] / self._count[b] if self._count[b] else (b + 0.5) / self.n_bins
        return {False: 1.0 - p, True: p}

    def learn_one(self, x, y, **kwargs):
        b = self._bin(x, **kwargs)
        self._sum[b] += float(y)
        self._count[b] += 1
        super().learn_one(x, y, **kwargs)
