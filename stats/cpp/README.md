# From-scratch C++ classifiers

`classifiers.cpp` (C++17, standard library only) implements: k-NN, Gaussian Naive Bayes,
logistic regression (batch GD, L2), CART decision tree (Gini), and an averaged perceptron.

Build/run: `g++ -O2 -std=c++17 -o classifiers classifiers.cpp && ./classifiers`
Output of one run (seed 42) is in `results_cpp.txt`.

## Mock dataset
Two Gaussian classes in 10 dimensions; only the first 2 dims are informative (class means at ±sep, sep=1),
the other 8 are pure noise. A fraction `noise` of training labels is flipped; at 0.5 the labels are
pure random. Test labels are always clean, so test accuracy shows how well the underlying signal was recovered.
(Note: with a 0.5 flip rate the "clean" test accuracy of ~0.47–0.53 is chance, as expected.)

## Results summary
- Clean data: GNB, logistic regression and perceptron ≈ 0.91 (data matches their assumptions: linear / Gaussian). Tree and kNN ≈ 0.87.
- Up to 40% label noise the linear/probabilistic models still reach 0.86–0.88 on clean test data; the tree drops to 0.66 and kNN to 0.62.
- Pure random labels: every model is at chance (~0.5) on test. Tree (train 0.69) and kNN (train 0.70) *memorize* noise, giving a large train/test gap; linear models only reach ~0.55 train.

## Pros / cons and behaviour on noisy or random data
| Method | Advantages | Disadvantages | Noisy / random data |
|---|---|---|---|
| k-NN | No training, nonlinear boundaries, simple | Slow prediction, needs all data in memory, hurt by irrelevant dims (8 of 10 here) and distance concentration | Low-bias/high-variance: fits noisy labels (train acc stays high at 0.5 noise), generalization falls fast. Larger k smooths. |
| Gaussian NB | Very fast, few parameters, works with little data | Assumes feature independence and Gaussian shape; linear/quadratic boundary only | High bias, low variance: robust to label noise; on random labels gives near-chance on both train and test. |
| Logistic regression | Calibrated probabilities, interpretable, regularizable | Linear boundary only; needs feature scaling and iterative optimization | Very robust (averages over all points); noise mostly shrinks weights. Random labels → weights ≈ 0, ~chance. |
| Decision tree | Nonlinear, handles mixed features, interpretable | Unstable, overfits; axis-aligned splits | Grows to fit noise (train 0.69 on random labels). Needs depth limit/pruning/ensembling. |
| Perceptron | Online, trivial, fast | Only converges if linearly separable; no probabilities; sensitive to order (averaging helps) | Plain version oscillates on noise; averaging makes it behave much like logistic regression. |

## Scalability (n samples, d features)
| Method | Train | Predict (per sample) | Notes |
|---|---|---|---|
| k-NN | O(1) (store) | O(n·d) brute force; trees/ANN help in low d | Measured: 5.4 s for 2000 predictions at n=32k, growing linearly with n. Memory O(n·d). |
| Gaussian NB | O(n·d) single pass | O(d) | Fastest; ~2 ms at n=32k. Trivially parallel/streamable. |
| Logistic regression | O(n·d·epochs) | O(d) | Linear in n; SGD/minibatch scales to huge data. |
| Decision tree | ≈O(d·n log n·depth) (sorting per node) | O(depth) | Measured slowest to train (1 s at 32k); histogram splits scale better. |
| Perceptron | O(n·d·epochs) | O(d) | Cheap, online. |

Caveat: single run, one seed; timings are indicative only and accuracy differences of ~1% are within noise.
