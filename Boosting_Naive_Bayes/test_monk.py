import argparse
import numpy as np

from sklearn.metrics import accuracy_score
from AdaBoost import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB
from naive_bayes import WeightedCategoricalNB
from helper import load_dataset


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




def run_problem(problem_id, n_estimators):
    X_train, y_train, _ = load_dataset("Monk_problem", split="train", problem_id=problem_id, return_ids=True)
    X_test, y_test, _ = load_dataset("Monk_problem", split="test", problem_id=problem_id, return_ids=True)

    model = AdaBoostClassifier(n_estimators=n_estimators, estimator = CategoricalNB)
    model.fit(X_train, y_train)
    sklearn_model = SkAdaBoostClassifier(n_estimators=n_estimators, estimator=CategoricalNB(), random_state=42)
    sklearn_model.fit(X_train, y_train)

    print(f"MONK-{problem_id}")
    print(f"  estimators used: {len(model.models)} / requested: {n_estimators}")
    if len(model.models) == 0:
        print("  no estimators were accepted by AdaBoost")
        print()
        return

    original_alphas = model.alphas
    original_models = model.models
    sk_alphas = sklearn_model.estimator_weights_
    sk_models = sklearn_model.estimators_

    print(f"{'iter':<6} {'train_acc':<10} {'test_acc':<10}")
    print("-" * 30)
    for t in range(1, len(original_models) + 1):
        model.alphas = original_alphas[:t]
        model.models = original_models[:t]
        sklearn_model.estimator_weights_ = sk_alphas[:t]
        sklearn_model.estimators_ = sk_models[:t]

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        y_pred_train_sklearn = sklearn_model.predict(X_train)
        y_pred_test_sklearn = sklearn_model.predict(X_test)
        train_acc_sklearn = accuracy_score(y_train, y_pred_train_sklearn)
        test_acc_sklearn = accuracy_score(y_test, y_pred_test_sklearn)

        print(f"{t:<6} {train_acc:<10.4f} {test_acc:<10.4f} (sklearn: {train_acc_sklearn:.4f} {test_acc_sklearn:.4f})")

    # Restore full model state after progressive evaluation.
    model.alphas = original_alphas
    model.models = original_models

    final_train_acc = accuracy_score(y_train, model.predict(X_train))
    final_test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  final train accuracy : {final_train_acc:.4f}")
    print(f"  final test accuracy  : {final_test_acc:.4f}")
    print(f"  sklearn final test accuracy: {accuracy_score(y_test, sklearn_model.predict(X_test)):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run AdaBoost on MONK datasets")
    parser.add_argument("--estimators", type=int, default=30, help="Number of boosting iterations")
    parser.add_argument(
        "--problem",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single MONK problem (1, 2, or 3). If omitted, run all.",
    )
    args = parser.parse_args()

    if args.problem is None:
        for pid in (1, 2, 3):
            run_problem(pid, args.estimators)
    else:
        run_problem(args.problem, args.estimators)


if __name__ == "__main__":
    main()
