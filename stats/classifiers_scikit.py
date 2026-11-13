"""Compare several scikit-learn classifiers on clean, noisy and random data."""
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=0),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "SVM (RBF)": lambda: SVC(),
    "GaussianNB": lambda: GaussianNB(),
}


def make_data(flip_y=0.0, seed=0, n=2000):
    """Mock dataset; flip_y = fraction of labels randomly reassigned (noise)."""
    return make_classification(n_samples=n, n_features=20, n_informative=5,
                               n_redundant=5, flip_y=flip_y, random_state=seed)


def run(name, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"\n== {name} ==")
    for m, factory in MODELS.items():
        clf = factory()
        t = time.perf_counter()
        clf.fit(Xtr, ytr)
        fit = time.perf_counter() - t
        print(f"{m:20s} train={accuracy_score(ytr, clf.predict(Xtr)):.3f} "
              f"test={accuracy_score(yte, clf.predict(Xte)):.3f} fit={fit:.3f}s")


if __name__ == "__main__":
    run("clean data", *make_data(0.0))
    run("noisy labels (flip_y=0.3)", *make_data(0.3))
    X, _ = make_data()
    rng = np.random.RandomState(0)
    run("pure random labels (expect ~0.5 test acc)", X, rng.randint(0, 2, len(X)))
