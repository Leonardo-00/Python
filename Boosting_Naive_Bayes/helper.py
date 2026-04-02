
import numpy as np
import os

import numpy as np

def discretize_data(X_train, X_test, n_bins=10, strategy='uniform'):
    """
    Discretizza le feature continue.
    Strategy: 
    - 'uniform': Equal Width (intervalli uguali)
    - 'quantile': Equal Frequency (stessa quantità di dati per bin)
    """
    X_train_disc = X_train.copy()
    X_test_disc = X_test.copy()
    
    for col in range(X_train.shape[1]):
        # Estraiamo i dati validi (non NaN) dal training per calcolare i bin
        train_col_valid = X_train[:, col][~np.isnan(X_train[:, col])]
        
        # Consideriamo continua una feature con molti valori unici
        if len(np.unique(train_col_valid)) > 20:
            if strategy == 'uniform':
                # Equal Width: divisione lineare tra min e max
                bins = np.linspace(np.nanmin(train_col_valid), 
                                   np.nanmax(train_col_valid), n_bins + 1)
            
            elif strategy == 'quantile':
                # Equal Frequency: divisione basata sui percentili
                # Usiamo np.unique sui quantili per evitare bin identici se ci sono troppi duplicati
                bins = np.unique(np.nanpercentile(train_col_valid, 
                                                  np.linspace(0, 100, n_bins + 1)))
            
            # Crea una maschera per i valori non NaN
            mask_train = ~np.isnan(X_train[:, col])
            mask_test = ~np.isnan(X_test[:, col])

            # Applica digitize solo dove i dati esistono
            X_train_disc[mask_train, col] = np.digitize(X_train[mask_train, col], bins[1:-1])
            X_test_disc[mask_test, col] = np.digitize(X_test[mask_test, col], bins[1:-1])
            
    return X_train_disc, X_test_disc

def load_dataset(case_name, split=None, problem_id=None, return_ids=False):
    """Carica dataset in modo uniforme.

    - Dataset singoli (es. German_credit, Pima_indians_diabetes):
      legge datasets/<case_name>/data.csv
    - Dataset MONK con split: richiede split in {train,test} e problem_id in {1,2,3},
      legge datasets/Monk_problem/monk_<problem_id>_<split>.csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "datasets", case_name)

    if split is None:
        data_path = os.path.join(dataset_dir, "data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if problem_id not in (1, 2, 3):
        raise ValueError("problem_id must be one of: 1, 2, 3")

    file_name = f"monk_{problem_id}_{split}.csv"
    data_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    raw = np.genfromtxt(data_path, delimiter=",", dtype=str, skip_header=1)
    y = raw[:, 0].astype(int)
    X = raw[:, 1:7].astype(int)
    ids = raw[:, 7]

    if return_ids:
        return X, y, ids
    return X, y