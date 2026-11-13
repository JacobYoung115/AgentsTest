"""Generate mock datasets, train all classifiers, report accuracy and timing.

Run: python3 stats/run_experiments.py
"""
import random
import time

from classifiers import KNN, GaussianNB, LogisticRegression, Perceptron, DecisionTree


def make_blobs(n, d=4, sep=1.5, seed=0):
    """Two Gaussian clusters; `sep` controls class separation."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        c = i % 2
        X.append([rng.gauss(sep * c, 1.0) for _ in range(d)])
        y.append(c)
    return X, y


def make_xor(n, seed=0):
    """Non-linearly-separable: label = sign(x0) XOR sign(x1), plus 2 noise features."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x = [rng.uniform(-1, 1) for _ in range(4)]
        X.append(x)
        y.append(int((x[0] > 0) != (x[1] > 0)))
    return X, y


def flip_labels(y, p, seed=1):
    rng = random.Random(seed)
    return [1 - l if rng.random() < p else l for l in y]


def split(X, y, frac=0.7):
    k = int(len(X) * frac)
    return X[:k], y[:k], X[k:], y[k:]


def models():
    return {
        "kNN(k=5)": KNN(5),
        "GaussianNB": GaussianNB(),
        "LogReg": LogisticRegression(),
        "Perceptron": Perceptron(),
        "Tree(d=6)": DecisionTree(),
    }


def acc(p, y):
    return sum(a == b for a, b in zip(p, y)) / len(y)


def evaluate(title, Xtr, ytr, Xte, yte):
    print(f"\n== {title} (train={len(Xtr)}, test={len(Xte)}) ==")
    print(f"{'model':<12}{'train acc':>10}{'test acc':>10}{'fit s':>8}{'pred s':>8}")
    for name, m in models().items():
        t0 = time.time(); m.fit(Xtr, ytr); t1 = time.time()
        pte = m.predict(Xte); t2 = time.time()
        print(f"{name:<12}{acc(m.predict(Xtr), ytr):>10.3f}{acc(pte, yte):>10.3f}{t1 - t0:>8.2f}{t2 - t1:>8.2f}")


def main():
    X, y = make_blobs(1000, sep=2.0, seed=1)
    evaluate("Linearly separable-ish blobs, clean", *split(X, y))

    Xtr, ytr, Xte, yte = split(*make_xor(1000, seed=2))
    evaluate("XOR (non-linear), clean", Xtr, ytr, Xte, yte)

    X, y = make_blobs(1000, sep=2.0, seed=1)
    Xtr, ytr, Xte, yte = split(X, y)
    evaluate("Blobs, 30% training labels flipped", Xtr, flip_labels(ytr, 0.3), Xte, yte)

    # Pure random: labels independent of features -> ideal accuracy is 50%.
    rng = random.Random(3)
    Xr = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(1000)]
    yr = [rng.randint(0, 1) for _ in range(1000)]
    evaluate("Pure random labels (no signal)", *split(Xr, yr))

    # Scaling of cost with n
    print("\n== Scalability: fit+predict time (s) vs n ==")
    ns = [250, 500, 1000, 2000]
    print(f"{'model':<12}" + "".join(f"{n:>8}" for n in ns))
    times = {k: [] for k in models()}
    for n in ns:
        Xtr, ytr, Xte, yte = split(*make_blobs(n, seed=5))
        for name, m in models().items():
            t0 = time.time(); m.fit(Xtr, ytr); m.predict(Xte)
            times[name].append(time.time() - t0)
    for name, ts in times.items():
        print(f"{name:<12}" + "".join(f"{t:>8.2f}" for t in ts))


if __name__ == "__main__":
    main()
