import argparse
import os
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from naive_bayes import WeightedCategoricalNB
from helper import load_dataset

from AdaBoost import AdaBoostClassifier


class SklearnWeightedCategoricalNB(BaseEstimator, ClassifierMixin):
    """Adapter sklearn per usare WeightedCategoricalNB in AdaBoost sklearn.

    sklearn richiede estimator clonabili (get_params/set_params) e API standard.
    BaseEstimator fornisce automaticamente get_params/set_params.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model_ = None

    def fit(self, X, y, sample_weight=None):
        self.model_ = WeightedCategoricalNB(alpha=self.alpha)
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def clean_pima_data(X, y):
    """Rimuove campioni con 0 biologicamente implausibili nel dataset Pima."""
    missing_cols = [1, 2, 3, 4, 5]
    mask = np.ones(X.shape[0], dtype=bool)
    for col in missing_cols:
        mask &= (X[:, col] != 0)
    return X[mask], y[mask]


def discretize_data(X):
    X_discrete = X.copy()
    for col in range(X.shape[1]):
        if len(np.unique(X[:, col])) > 20:
            bins = np.linspace(X[:, col].min(), X[:, col].max(), 21)
            X_discrete[:, col] = np.digitize(X[:, col], bins)
    return X_discrete


def load_german_credit(base_dir, test_size, random_state):
    X, y = load_dataset("German_credit")
    X = discretize_data(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_monk_split(base_dir, problem_id, split):
    X, y = load_dataset("Monk_problem", split=split, problem_id=problem_id)
    return X, y


def load_monk(base_dir, problem_id):
    X_train, y_train = load_monk_split(base_dir, problem_id, "train")
    X_test, y_test = load_monk_split(base_dir, problem_id, "test")
    return X_train, X_test, y_train, y_test


def load_pima(base_dir, test_size, random_state):
    X, y = load_dataset("Pima_indians_diabetes")
    
    # Discretizza
    X = discretize_data(X)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def our_staged_test_accuracy(model, X_test, y_test):
    original_alphas = model.alphas
    original_models = model.models
    values = []

    for t in range(1, len(original_models) + 1):
        model.alphas = original_alphas[:t]
        model.models = original_models[:t]
        y_pred = model.predict(X_test)
        values.append(accuracy_score(y_test, y_pred))

    model.alphas = original_alphas
    model.models = original_models
    return values


def build_sklearn_model(base, n_estimators, random_state):
    if base == "nb":
        estimator = SklearnWeightedCategoricalNB(alpha=1.0)
    elif base == "stump":
        estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    else:
        raise ValueError("base must be 'nb' or 'stump'")

    return SkAdaBoostClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        algorithm="SAMME",
        random_state=random_state,
    )


def print_summary(name, staged_acc):
    if len(staged_acc) == 0:
        print(f"{name}: no estimators used")
        return

    best_t = int(np.argmax(staged_acc)) + 1
    best_acc = staged_acc[best_t - 1]
    final_acc = staged_acc[-1]
    print(f"{name}: best_test={best_acc:.4f} at iter={best_t}, final_test={final_acc:.4f}, used={len(staged_acc)}")


def main():
    parser = argparse.ArgumentParser(description="Compare custom AdaBoost with sklearn AdaBoost")
    parser.add_argument("--dataset", choices=["german", "monk", "pima"], default="pima")
    parser.add_argument("--problem", type=int, choices=[1, 2, 3], default=1, help="Used when --dataset monk")
    parser.add_argument("--estimators", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2, help="Used when --dataset german")
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--sk-base", choices=["nb", "stump"], default="nb")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "german":
        X_train, X_test, y_train, y_test = load_german_credit(base_dir, args.test_size, args.random_state)
        dataset_label = "German Credit"
    elif args.dataset == "monk":
        X_train, X_test, y_train, y_test = load_monk(base_dir, args.problem)
        dataset_label = f"MONK-{args.problem}"
    else:  # pima
        X_train, X_test, y_train, y_test = load_pima(base_dir, args.test_size, args.random_state)
        dataset_label = "Pima Indians Diabetes (cleaned)"

    print(f"Dataset: {dataset_label}")
    print(f"Train size: {len(y_train)}  Test size: {len(y_test)}")
    print(f"Estimators requested: {args.estimators}")
    print(f"sklearn base estimator: {args.sk_base}")
    print()

    our_model = AdaBoostClassifier(n_estimators=args.estimators)
    our_model.fit(X_train, y_train)
    our_staged = our_staged_test_accuracy(our_model, X_test, y_test)

    sk_model = build_sklearn_model(args.sk_base, args.estimators, args.random_state)
    sk_model.fit(X_train, y_train)
    sk_staged = [accuracy_score(y_test, y_pred) for y_pred in sk_model.staged_predict(X_test)]

    print_summary("Our AdaBoost", our_staged)
    print_summary("sklearn AdaBoost", sk_staged)
    print()

    print(f"{'iter':<6} {'our_test_acc':<12} {'sk_test_acc':<12}")
    print("-" * 36)
    max_len = max(len(our_staged), len(sk_staged))
    for i in range(max_len):
        ours = f"{our_staged[i]:.4f}" if i < len(our_staged) else "-"
        sk = f"{sk_staged[i]:.4f}" if i < len(sk_staged) else "-"
        print(f"{i + 1:<6} {ours:<12} {sk:<12}")


if __name__ == "__main__":
    main()
