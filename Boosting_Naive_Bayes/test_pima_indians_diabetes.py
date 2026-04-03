from AdaBoost import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from naive_bayes import WeightedCategoricalNB
import numpy as np
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "datasets", "Pima_indians_diabetes")

def load_ripley(filename):
    """Carica i file di Ripley (pima.tr, pima.te, pima.tr2)."""
    with open(os.path.join(BASE, filename)) as f:
        lines = f.readlines()
    data, labels = [], []
    for line in lines[1:]:
        parts = line.split()
        data.append([float(x) if x != "NA" else np.nan for x in parts[:-1]])
        labels.append(1 if parts[-1] == "Yes" else 0)
    return np.array(data), np.array(labels)

def discretize_equal_width(X_train, X_test, n_bins=10):
    """10 equal-width bins per OGNI feature (Elkan 1997, Eq. 4)."""
    X_tr_d, X_te_d = X_train.copy(), X_test.copy()
    for col in range(X_train.shape[1]):
        valid = X_train[:, col][~np.isnan(X_train[:, col])]
        if len(valid) == 0 or valid.min() == valid.max():
            continue
        bins = np.linspace(valid.min(), valid.max(), n_bins + 1)
        for X_src, X_dst in [(X_train, X_tr_d), (X_test, X_te_d)]:
            mask = ~np.isnan(X_src[:, col])
            X_dst[mask, col] = np.digitize(X_src[mask, col], bins[1:-1])
    return X_tr_d, X_te_d

def main():
    # 1. Caricamento split di Ripley (1996)
    X_train_c, y_train_c = load_ripley("pima.tr")     # 200 completi
    X_test, y_test       = load_ripley("pima.te")      # 332 test
    X_train_m, y_train_m = load_ripley("pima.tr2")     # 200 completi + 100 incompleti

    n_bins = 10

    # 2. Discretizzazione: 10 equal-width bins
    X_train_c_d, X_test_c_d = discretize_equal_width(X_train_c, X_test, n_bins)
    X_train_m_d, X_test_m_d = discretize_equal_width(X_train_m, X_test, n_bins)

    # 3. Preparazione per Sklearn (NaN -> Categoria Extra)
    X_train_sk = np.nan_to_num(X_train_m_d, nan=n_bins).astype(int)
    X_test_sk  = np.nan_to_num(X_test_m_d, nan=n_bins).astype(int)
    all_combined = np.vstack((X_train_sk, X_test_sk))
    n_categories_per_feature = (np.max(all_combined, axis=0) + 1).astype(int)

    # --- TRAINING ---
    # Modello custom: AdaBoost continuo (Elkan 1997), MLE quasi-puro
    def make_nb():
        return WeightedCategoricalNB(alpha=0.01)

    my_boost_c = AdaBoostClassifier(n_estimators=10, estimator=make_nb, algorithm="continuous")
    my_boost_c.fit(X_train_c_d, y_train_c)

    my_boost_m = AdaBoostClassifier(n_estimators=10, estimator=make_nb, algorithm="continuous")
    my_boost_m.fit(X_train_m_d, y_train_m)

    # Sklearn (SAMME, solo caso mixed)
    sk_boost = SkAdaBoost(
        estimator=CategoricalNB(min_categories=n_categories_per_feature),
        n_estimators=10,
        algorithm='SAMME'
    )
    sk_boost.fit(X_train_sk, y_train_m)

    # --- CONFRONTO ---
    paper_c = [24.7, 23.8, 22.9, 22.6, 22.9, 22.9, 22.9, 22.9, 22.6, 22.0]
    paper_m = [20.2, 20.5, 19.9, 19.6, 19.6, 19.3, 19.0, 19.0, 19.0, 18.7]

    orig_c_a, orig_c_m = my_boost_c.alphas, my_boost_c.models
    orig_m_a, orig_m_m = my_boost_m.alphas, my_boost_m.models
    sk_staged = list(sk_boost.staged_predict(X_test_sk))

    print(f"\n{'Iter':<5} | {'Complete':>10} | {'Paper(C)':>10} | {'Mixed':>10} | {'Paper(M)':>10} | {'Sk Mixed':>10}")
    print("-" * 70)

    for t in range(1, 11):
        # Complete
        if t <= len(orig_c_m):
            my_boost_c.alphas, my_boost_c.models = orig_c_a[:t], orig_c_m[:t]
            err_c = (1 - accuracy_score(y_test, my_boost_c.predict(X_test_c_d))) * 100
        else:
            err_c = float('nan')

        # Mixed
        if t <= len(orig_m_m):
            my_boost_m.alphas, my_boost_m.models = orig_m_a[:t], orig_m_m[:t]
            err_m = (1 - accuracy_score(y_test, my_boost_m.predict(X_test_m_d))) * 100
        else:
            err_m = float('nan')

        # Sklearn
        sk_err = (1 - accuracy_score(y_test, sk_staged[t-1])) * 100 if t <= len(sk_staged) else float('nan')

        print(f"{t:<5} | {err_c:>8.1f}%  | {paper_c[t-1]:>7.1f}%  | {err_m:>8.1f}%  | {paper_m[t-1]:>7.1f}%  | {sk_err:>8.1f}%")

    my_boost_c.alphas, my_boost_c.models = orig_c_a, orig_c_m
    my_boost_m.alphas, my_boost_m.models = orig_m_a, orig_m_m

if __name__ == "__main__":
    main()
