// From-scratch classifiers in C++17 (no external libraries).
// Build: g++ -O2 -std=c++17 -o classifiers classifiers.cpp && ./classifiers
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;
typedef vector<double> Vec;
struct Data { vector<Vec> X; vector<int> y; };

// Mock dataset: two Gaussian blobs in `d` dims (only first 2 dims informative
// by mean shift), then `noise` fraction of labels flipped. noise=0.5 => pure random.
Data make_data(int n, int d, double sep, double noise, mt19937& rng) {
    normal_distribution<double> N(0, 1);
    uniform_real_distribution<double> U(0, 1);
    Data D;
    for (int i = 0; i < n; i++) {
        int c = i % 2;
        Vec x(d);
        for (int j = 0; j < d; j++) x[j] = N(rng) + (j < 2 ? (c ? sep : -sep) : 0.0);
        int label = c;
        if (U(rng) < noise) label = 1 - label;
        D.X.push_back(x); D.y.push_back(label);
    }
    return D;
}

struct Classifier {
    virtual ~Classifier() {}
    virtual string name() const = 0;
    virtual void fit(const Data&) = 0;
    virtual int predict(const Vec&) const = 0;
};

// k-Nearest Neighbours (brute force)
struct KNN : Classifier {
    int k; Data tr;
    KNN(int k) : k(k) {}
    string name() const override { return "kNN(k=" + to_string(k) + ")"; }
    void fit(const Data& D) override { tr = D; }
    int predict(const Vec& x) const override {
        vector<pair<double,int>> dist;
        for (size_t i = 0; i < tr.X.size(); i++) {
            double s = 0;
            for (size_t j = 0; j < x.size(); j++) { double t = x[j] - tr.X[i][j]; s += t * t; }
            dist.push_back({s, tr.y[i]});
        }
        int kk = min<int>(k, dist.size());
        nth_element(dist.begin(), dist.begin() + kk - 1, dist.end());
        int v = 0;
        for (int i = 0; i < kk; i++) v += dist[i].second;
        return 2 * v > kk;
    }
};

// Gaussian Naive Bayes
struct GNB : Classifier {
    Vec mu[2], var[2]; double prior[2];
    string name() const override { return "GaussianNB"; }
    void fit(const Data& D) override {
        size_t d = D.X[0].size();
        for (int c = 0; c < 2; c++) {
            mu[c].assign(d, 0); var[c].assign(d, 0);
            int n = 0;
            for (size_t i = 0; i < D.X.size(); i++) if (D.y[i] == c) {
                n++; for (size_t j = 0; j < d; j++) mu[c][j] += D.X[i][j];
            }
            for (auto& m : mu[c]) m /= n;
            for (size_t i = 0; i < D.X.size(); i++) if (D.y[i] == c)
                for (size_t j = 0; j < d; j++) var[c][j] += pow(D.X[i][j] - mu[c][j], 2);
            for (auto& v : var[c]) v = v / n + 1e-9;
            prior[c] = double(n) / D.X.size();
        }
    }
    int predict(const Vec& x) const override {
        double ll[2];
        for (int c = 0; c < 2; c++) {
            ll[c] = log(prior[c]);
            for (size_t j = 0; j < x.size(); j++)
                ll[c] += -0.5 * log(2 * M_PI * var[c][j]) - pow(x[j] - mu[c][j], 2) / (2 * var[c][j]);
        }
        return ll[1] > ll[0];
    }
};

// Logistic regression, batch gradient descent with L2
struct LogReg : Classifier {
    Vec w; double b = 0, lr, l2; int epochs;
    LogReg(double lr = 0.1, double l2 = 1e-3, int epochs = 300) : lr(lr), l2(l2), epochs(epochs) {}
    string name() const override { return "LogisticRegression"; }
    double z(const Vec& x) const { double s = b; for (size_t j = 0; j < x.size(); j++) s += w[j] * x[j]; return s; }
    void fit(const Data& D) override {
        size_t d = D.X[0].size(), n = D.X.size();
        w.assign(d, 0); b = 0;
        for (int e = 0; e < epochs; e++) {
            Vec g(d, 0); double gb = 0;
            for (size_t i = 0; i < n; i++) {
                double err = 1 / (1 + exp(-z(D.X[i]))) - D.y[i];
                for (size_t j = 0; j < d; j++) g[j] += err * D.X[i][j];
                gb += err;
            }
            for (size_t j = 0; j < d; j++) w[j] -= lr * (g[j] / n + l2 * w[j]);
            b -= lr * gb / n;
        }
    }
    int predict(const Vec& x) const override { return z(x) > 0; }
};

// Decision tree (CART, Gini, axis-aligned splits)
struct Tree : Classifier {
    struct Node { int feat = -1; double thr = 0; int label = 0; int l = -1, r = -1; };
    vector<Node> nodes; int max_depth, min_leaf; const Data* D = nullptr;
    Tree(int md = 8, int ml = 2) : max_depth(md), min_leaf(ml) {}
    string name() const override { return "DecisionTree(depth=" + to_string(max_depth) + ")"; }
    static double gini(int a, int n) { if (!n) return 0; double p = double(a) / n; return 2 * p * (1 - p); }
    int build(vector<int>& idx, int depth) {
        int id = nodes.size(); nodes.push_back(Node());
        int n = idx.size(), ones = 0;
        for (int i : idx) ones += D->y[i];
        nodes[id].label = 2 * ones >= n;
        if (depth >= max_depth || n < 2 * min_leaf || ones == 0 || ones == n) return id;
        double best = gini(ones, n) - 1e-12; int bf = -1; double bt = 0;
        int d = D->X[0].size();
        for (int f = 0; f < d; f++) {
            sort(idx.begin(), idx.end(), [&](int a, int b) { return D->X[a][f] < D->X[b][f]; });
            int lo = 0;
            for (int i = 0; i + 1 < n; i++) {
                lo += D->y[idx[i]];
                if (D->X[idx[i]][f] == D->X[idx[i + 1]][f]) continue;
                int nl = i + 1, nr = n - nl;
                if (nl < min_leaf || nr < min_leaf) continue;
                double g = (nl * gini(lo, nl) + nr * gini(ones - lo, nr)) / n;
                if (g < best) { best = g; bf = f; bt = (D->X[idx[i]][f] + D->X[idx[i + 1]][f]) / 2; }
            }
        }
        if (bf < 0) return id;
        vector<int> L, R;
        for (int i : idx) (D->X[i][bf] <= bt ? L : R).push_back(i);
        int l = build(L, depth + 1), r = build(R, depth + 1);
        nodes[id].feat = bf; nodes[id].thr = bt; nodes[id].l = l; nodes[id].r = r;
        return id;
    }
    void fit(const Data& d) override {
        nodes.clear(); D = &d;
        vector<int> idx(d.X.size()); iota(idx.begin(), idx.end(), 0);
        build(idx, 0);
    }
    int predict(const Vec& x) const override {
        int i = 0;
        while (nodes[i].feat >= 0) i = x[nodes[i].feat] <= nodes[i].thr ? nodes[i].l : nodes[i].r;
        return nodes[i].label;
    }
};

// Perceptron (averaged)
struct Perceptron : Classifier {
    Vec w, wsum; double b = 0, bsum = 0; int epochs; mt19937 rng{1};
    Perceptron(int e = 20) : epochs(e) {}
    string name() const override { return "AveragedPerceptron"; }
    void fit(const Data& D) override {
        size_t d = D.X[0].size(), n = D.X.size();
        w.assign(d, 0); wsum.assign(d, 0); b = bsum = 0;
        vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
        for (int e = 0; e < epochs; e++) {
            shuffle(idx.begin(), idx.end(), rng);
            for (int i : idx) {
                double s = b; for (size_t j = 0; j < d; j++) s += w[j] * D.X[i][j];
                int t = D.y[i] ? 1 : -1;
                if (t * s <= 0) { for (size_t j = 0; j < d; j++) w[j] += t * D.X[i][j]; b += t; }
                for (size_t j = 0; j < d; j++) wsum[j] += w[j];
                bsum += b;
            }
        }
    }
    int predict(const Vec& x) const override {
        double s = bsum; for (size_t j = 0; j < x.size(); j++) s += wsum[j] * x[j];
        return s > 0;
    }
};

double accuracy(const Classifier& c, const Data& D) {
    int ok = 0;
    for (size_t i = 0; i < D.X.size(); i++) ok += c.predict(D.X[i]) == D.y[i];
    return double(ok) / D.X.size();
}
double secs(chrono::steady_clock::time_point a) {
    return chrono::duration<double>(chrono::steady_clock::now() - a).count();
}
vector<unique_ptr<Classifier>> make_all() {
    vector<unique_ptr<Classifier>> v;
    v.emplace_back(new KNN(5)); v.emplace_back(new GNB()); v.emplace_back(new LogReg());
    v.emplace_back(new Tree(8)); v.emplace_back(new Perceptron());
    return v;
}

int main() {
    mt19937 rng(42);
    printf("== Experiment 1: label noise (n_train=1000, n_test=2000 clean labels, d=10, sep=1.0) ==\n");
    printf("%-8s", "noise");
    for (auto& c : make_all()) printf("%-24s", c->name().c_str());
    printf("\n");
    for (double noise : {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}) {
        Data tr = make_data(1000, 10, 1.0, noise, rng), te = make_data(2000, 10, 1.0, 0.0, rng);
        printf("%-8.2f", noise);
        for (auto& c : make_all()) {
            c->fit(tr);
            char buf[64]; snprintf(buf, sizeof buf, "tr=%.3f te=%.3f", accuracy(*c, tr), accuracy(*c, te));
            printf("%-24s", buf);
        }
        printf("\n");
    }
    printf("\n== Experiment 2: scalability (d=10, sep=1.0, noise=0.1; fit s / predict-2000 s) ==\n");
    printf("%-8s", "n");
    for (auto& c : make_all()) printf("%-24s", c->name().c_str());
    printf("\n");
    Data te = make_data(2000, 10, 1.0, 0.0, rng);
    for (int n : {500, 2000, 8000, 32000}) {
        Data tr = make_data(n, 10, 1.0, 0.1, rng);
        printf("%-8d", n);
        for (auto& c : make_all()) {
            auto t0 = chrono::steady_clock::now(); c->fit(tr); double f = secs(t0);
            t0 = chrono::steady_clock::now(); accuracy(*c, te); double p = secs(t0);
            char buf[64]; snprintf(buf, sizeof buf, "%.4f / %.4f", f, p);
            printf("%-24s", buf);
        }
        printf("\n");
    }
    return 0;
}
