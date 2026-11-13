# Classifier comparison

`classifiers.py` builds a mock dataset with `sklearn.datasets.make_classification`
(2000 samples, 20 features, 5 informative, 5 redundant) and trains six classifiers on
three variants: clean, 30% label noise, and purely random labels.
Run: `pip install scikit-learn && python stats/classifiers.py`.

> Note: the script was written in an environment without scikit-learn, so it has
> **not been run**; no measured numbers are included. The notes below are general
> knowledge of these methods, not results from this script.

| Model | Pros | Cons | On noisy / random data | Scalability (n samples, d features) |
|---|---|---|---|---|
| Logistic regression | Fast, interpretable, well calibrated | Linear boundary only | Low variance; train≈test, degrades gracefully; random labels -> ~50% on both | Training roughly O(n·d) per iteration; scales to very large data (SGD variants for more) |
| k-NN | No training, simple, nonlinear | Slow prediction, hurt by irrelevant features / high d, needs scaling | With small k memorises noise; larger k smooths. Random labels -> ~50% test | Fit O(1) but memory O(n·d); predict O(n·d) per query without index; poor at large n |
| Decision tree | Interpretable, handles mixed features, no scaling | High variance, unstable | Grows until it fits noise: train≈100%, test drops sharply; random labels -> 100% train, ~50% test. Limit with max_depth / min_samples_leaf | ~O(n·d·log n) to build; fast predict; fine for large n |
| Random forest | Strong default, robust, less overfit than one tree | Less interpretable, bigger/slower | Still memorises noisy labels in train (deep trees) but averaging keeps test accuracy better than a single tree; random labels -> ~50% test | Same as trees × n_estimators; parallelisable; memory heavy |
| SVM (RBF) | Powerful nonlinear, good in high d with modest n | Needs scaling, tuning of C/gamma, no native probabilities | Can overfit noise with large C/gamma; soft margin (small C) helps | Training between O(n²) and O(n³); impractical beyond ~100k samples (use LinearSVC/SGD) |
| Gaussian NB | Very fast, works with little data | Independence/Gaussian assumptions | High bias so barely overfits; stable but may underfit; random labels -> ~50% | O(n·d); scales very well, supports partial_fit |

**Key point:** on pure random labels no model can beat chance (~50%) on held-out data;
high-capacity models (tree, forest, kNN with k=1, RBF SVM) show high train accuracy
with chance test accuracy (overfitting), while low-capacity ones (logistic regression,
NB) show both near chance. A large train/test gap signals memorisation.
