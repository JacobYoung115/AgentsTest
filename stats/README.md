# From-scratch classifiers (`stats/`)

Pure-Python (stdlib only) implementations in `classifiers.py`: k-NN, Gaussian Naive Bayes,
Logistic Regression (batch GD), Perceptron, and a CART Decision Tree.
`run_experiments.py` generates mock datasets (2-Gaussian blobs, XOR, label-noise blobs, pure random),
trains every model and prints accuracy/timing. Output of one run (seeded) is in `results.txt`.

Run: `python3 stats/run_experiments.py`

## Results summary (test accuracy)
| Dataset | kNN | GNB | LogReg | Perceptron | Tree |
|---|---|---|---|---|---|
| Blobs, clean | .967 | .967 | .957 | .967 | .943 |
| XOR | .927 | .493 | .497 | .500 | .853 |
| Blobs, 30% label noise | .843 | .970 | .970 | .290 | .777 |
| Random labels | .550 | .520 | .537 | .473 | .470 |

## Pros / cons and behavior on noisy / random data
- **k-NN**: no training, handles non-linear boundaries (XOR). But memorizes: with noisy labels it
  fits noise (train .75 vs. clean-ish test .84) and on random labels it shows a train/test gap
  (.68 vs .55). Larger k smooths noise. Sensitive to feature scale and irrelevant features.
- **Gaussian NB**: very fast, few parameters, so it is robust to label noise (.97 despite 30% flips)
  and on random data stays near 50% (no overfitting). Fails when the Gaussian/independence
  assumption is wrong (XOR ≈ chance).
- **Logistic regression**: linear, regularized, convex; averages over noise so it is robust
  (.97 with noise, ~chance on random data). Cannot model XOR without feature engineering.
- **Perceptron**: simplest and cheapest, but only converges on separable data; with label noise it
  never converges and the final weights are arbitrary (0.29 here — an unlucky last update; an averaged
  perceptron or decaying learning rate would fix this). Least stable on noise.
- **Decision tree**: captures non-linearity and interactions, interpretable, no scaling needed.
  High variance: with noise or random labels it carves up the space to fit noise (train .65 vs test
  .47 on random). Depth limits / pruning / ensembling mitigate it.

General rule: high-capacity models (kNN, deep trees) overfit noise; low-capacity/high-bias models
(NB, logistic regression) degrade gracefully. On pure random data no model beats ~50% on test;
a train-test gap is the symptom of memorization.
Caveat: single seeded runs on 1000 points — differences of a few points are within noise.

## Scalability (n samples, d features, k classes)
| Method | Train | Predict (per sample) | Notes |
|---|---|---|---|
| k-NN | O(1) (stores data) | O(n·d) (+sort; here O(n log n)) | Memory O(n·d); measured time ~quadratic in n for a full test set. Needs KD-tree/ANN for large n, degrades in high d. |
| Gaussian NB | O(n·d) single pass | O(k·d) | Scales best; trivially parallel/streamable. |
| Logistic regression | O(epochs·n·d) | O(d) | Linear in n; SGD/minibatch handles huge data. |
| Perceptron | O(epochs·n·d) | O(d) | Online, cheap. |
| Decision tree | ~O(d·n log n · depth) (sort per node) | O(depth) | Fast prediction; scales well, trees are easy to parallelize in ensembles. |

The timing table at the end of `results.txt` agrees: kNN grows ~4x per doubling of n (the test set also grows),
others roughly linearly. Pure-Python timings are only relative.
