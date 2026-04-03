from AdaBoost import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from helper import discretize_data, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

def clean_data(X, y):
    """Sostituisce gli 0 con NaN nelle colonne specificate."""
    X_clean = X.copy()
    cols_with_zeros_as_missing = [1, 2, 3, 4, 5]
    for col in cols_with_zeros_as_missing:
        X_clean[X_clean[:, col] == 0, col] = np.nan
    return X_clean, y

def sample_elkan_scenario(X_nan, y, scenario="mixed"):
    """
    Replica il campionamento di Elkan: 
    200 campioni completi + 100 incompleti per il training.
    """
    mask_complete = ~np.isnan(X_nan).any(axis=1)
    X_comp, y_comp = X_nan[mask_complete], y[mask_complete]
    
    mask_inc = np.isnan(X_nan).any(axis=1)
    X_inc, y_inc = X_nan[mask_inc], y[mask_inc]
    
    # Split 200 completi
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
        X_comp, y_comp, train_size=200, random_state=2
    )
    
    if scenario == "mixed":
        # Aggiunta 100 incompleti
        X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(
            X_inc, y_inc, train_size=100, random_state=2
        )
        X_train = np.vstack((X_c_train, X_i_train))
        y_train = np.concatenate((y_c_train, y_i_train))
        X_test = np.vstack((X_c_test, X_i_test))
        y_test = np.concatenate((y_c_test, y_i_test))
        return X_train, y_train, X_test, y_test
    
    return X_c_train, y_c_train, X_c_test, y_c_test

def main():
    # 1. Caricamento e Pulizia (NaN al posto degli 0)
    X_raw, y_raw = load_dataset("Pima_indians_diabetes")
    X_nan, y = clean_data(X_raw, y_raw)
    
    # 2. Campionamento
    X_train_nan, y_train, X_test_nan, y_test = sample_elkan_scenario(X_nan, y, scenario="mixed")
    
    # 3. Discretizzazione (I NaN rimangono NaN)
    n_bins = 10
    X_train_my, X_test_my = discretize_data(X_train_nan, X_test_nan, n_bins=n_bins, strategy="uniform")
    
    # 4. Preparazione per Sklearn (NaN -> Categoria Extra)
    # Usiamo n_bins come indice per la categoria "Missing"
    X_train_sk = np.nan_to_num(X_train_my, nan=n_bins).astype(int)
    X_test_sk = np.nan_to_num(X_test_my, nan=n_bins).astype(int)
    
    # --- TRAINING ---
    # Tuo Modello
    my_boost = AdaBoostClassifier(n_estimators=20)
    my_boost.fit(X_train_my, y_train)
    
    # 4. Preparazione per Sklearn
    X_train_sk = np.nan_to_num(X_train_my, nan=n_bins).astype(int)
    X_test_sk = np.nan_to_num(X_test_my, nan=n_bins).astype(int)
    
    # Calcoliamo il numero massimo di categorie realmente presenti nei dati discretizzati
    # Questo evita l'IndexError se np.digitize ha creato bin extra
    all_combined = np.vstack((X_train_sk, X_test_sk))
    n_categories_per_feature = (np.max(all_combined, axis=0) + 1).astype(int)
    
    # --- TRAINING ---
    # Sklearn (SAMME)
    sk_boost = SkAdaBoost(
        # min_categories deve essere un array che dice quanti bin esistono per OGNI feature
        estimator=CategoricalNB(min_categories=n_categories_per_feature),
        n_estimators=20,
        algorithm='SAMME'
    )
    sk_boost.fit(X_train_sk, y_train)
    
    # --- CONFRONTO ---
    print(f"\n{'Iter':<5} | {'My Test Acc':<12} | {'Sk Test Acc':<12} | {'Diff':<10}")
    print("-" * 50)
    
    sk_staged = sk_boost.staged_predict(X_test_sk)
    orig_alphas, orig_models = my_boost.alphas, my_boost.models
    
    for t, sk_pred in enumerate(sk_staged, 1):
        if t > len(orig_models): break
        
        my_boost.alphas, my_boost.models = orig_alphas[:t], orig_models[:t]
        my_acc = accuracy_score(y_test, my_boost.predict(X_test_my))
        sk_acc = accuracy_score(y_test, sk_pred)
        
        print(f"{t:<5} | {my_acc:<12.4f} | {sk_acc:<12.4f} | {my_acc - sk_acc:<10.4f}")

    my_boost.alphas, my_boost.models = orig_alphas, orig_models

if __name__ == "__main__":
    main()