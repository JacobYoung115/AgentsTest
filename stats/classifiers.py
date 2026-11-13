"""Classifiers implemented from scratch (standard library only)."""
import math
import random
from collections import Counter


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_one(self, x):
        d = sorted((sum((a - b) ** 2 for a, b in zip(x, p)), l) for p, l in zip(self.X, self.y))
        return Counter(l for _, l in d[: self.k]).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]


class GaussianNB:
    def fit(self, X, y):
        self.stats, self.prior = {}, {}
        n = len(X)
        for c in set(y):
            rows = [x for x, l in zip(X, y) if l == c]
            m = [sum(col) / len(rows) for col in zip(*rows)]
            v = [sum((a - mu) ** 2 for a in col) / len(rows) + 1e-9 for col, mu in zip(zip(*rows), m)]
            self.stats[c] = (m, v)
            self.prior[c] = len(rows) / n
        return self

    def predict_one(self, x):
        best, bs = None, -1e300
        for c, (m, v) in self.stats.items():
            s = math.log(self.prior[c]) + sum(
                -0.5 * math.log(2 * math.pi * vi) - (a - mi) ** 2 / (2 * vi) for a, mi, vi in zip(x, m, v))
            if s > bs:
                best, bs = c, s
        return best

    def predict(self, X):
        return [self.predict_one(x) for x in X]


class LogisticRegression:
    """Binary (labels 0/1), batch gradient descent with L2 penalty."""
    def __init__(self, lr=0.1, epochs=200, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sig(z):
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    def fit(self, X, y):
        d, n = len(X[0]), len(X)
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for x, t in zip(X, y):
                e = self._sig(sum(w * a for w, a in zip(self.w, x)) + self.b) - t
                for j in range(d):
                    gw[j] += e * x[j]
                gb += e
            self.w = [w - self.lr * (g / n + self.l2 * w) for w, g in zip(self.w, gw)]
            self.b -= self.lr * gb / n
        return self

    def predict(self, X):
        return [int(sum(w * a for w, a in zip(self.w, x)) + self.b > 0) for x in X]


class Perceptron:
    def __init__(self, epochs=50, seed=0):
        self.epochs, self.seed = epochs, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        self.w, self.b = [0.0] * len(X[0]), 0.0
        idx = list(range(len(X)))
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:
                t = 1 if y[i] else -1
                if t * (sum(w * a for w, a in zip(self.w, X[i])) + self.b) <= 0:
                    self.w = [w + t * a for w, a in zip(self.w, X[i])]
                    self.b += t
        return self

    def predict(self, X):
        return [int(sum(w * a for w, a in zip(self.w, x)) + self.b > 0) for x in X]


class DecisionTree:
    """CART with Gini impurity."""
    def __init__(self, max_depth=6, min_samples=5):
        self.max_depth, self.min_samples = max_depth, min_samples

    @staticmethod
    def _gini(counts, n):
        return 1 - sum((c / n) ** 2 for c in counts.values()) if n else 0

    def _build(self, X, y, depth):
        counts = Counter(y)
        if depth >= self.max_depth or len(counts) == 1 or len(y) < self.min_samples:
            return counts.most_common(1)[0][0]
        n, best = len(y), (self._gini(counts, len(y)), None)
        for j in range(len(X[0])):
            order = sorted(range(n), key=lambda i: X[i][j])
            left, right = Counter(), Counter(counts)
            for k in range(n - 1):
                l = y[order[k]]
                left[l] += 1
                right[l] -= 1
                a, b = X[order[k]][j], X[order[k + 1]][j]
                if a == b:
                    continue
                g = (self._gini(left, k + 1) * (k + 1) + self._gini(right, n - k - 1) * (n - k - 1)) / n
                if g < best[0] - 1e-12:
                    best = (g, (j, (a + b) / 2))
        if best[1] is None:
            return counts.most_common(1)[0][0]
        j, thr = best[1]
        L = [i for i in range(n) if X[i][j] <= thr]
        R = [i for i in range(n) if X[i][j] > thr]
        return (j, thr, self._build([X[i] for i in L], [y[i] for i in L], depth + 1),
                self._build([X[i] for i in R], [y[i] for i in R], depth + 1))

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def predict_one(self, x):
        t = self.tree
        while isinstance(t, tuple):
            t = t[2] if x[t[0]] <= t[1] else t[3]
        return t

    def predict(self, X):
        return [self.predict_one(x) for x in X]
