import argparse
import os
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.base import BaseEstimator, ClassifierMixin
from naive_bayes import WeightedCategoricalNB

from helper import load_dataset, discretize_data

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


def main():
    
    case_name = "German_credit"
    
    X, y = load_dataset(case_name)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_test = discretize_data(X_train_raw, X_test_raw)
    
    max_estimators = 100
    
    adaboost_model_1 = SkAdaBoostClassifier(n_estimators=max_estimators, estimator=SklearnWeightedCategoricalNB())
    adaboost_model_1.fit(X_train, y_train)
    adaboost_model_2 = SkAdaBoostClassifier(n_estimators=max_estimators, estimator=CategoricalNB())
    adaboost_model_2.fit(X_train, y_train)
    
    print(f"\n{accuracy_score(y_test, adaboost_model_1.predict(X_test)):.4f} (WeightedCategoricalNB)")
    print(f"{accuracy_score(y_test, adaboost_model_2.predict(X_test)):.4f} (CategoricalNB)")
    
    
if __name__ == "__main__":
    main()