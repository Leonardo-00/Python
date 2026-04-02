from sklearn.metrics import accuracy_score
from AdaBoost import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from helper import discretize_data, load_dataset

if __name__ == "__main__":
    
    # Load dataset

    case_name = "German_credit"
    X, y = load_dataset(case_name)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the dataset
    X_train, X_test = discretize_data(X_train_raw, X_test_raw)

    max_estimators = 100
    train_accuracies = []
    test_accuracies = []

    # Alleniamo il modello una volta sola con il numero massimo di stimatori
    adaboost_model = AdaBoostClassifier(n_estimators=max_estimators)
    adaboost_model.fit(X_train, y_train)

    print(f"\n{'Iter':<5} | {'Train Acc':<10} | {'Test Acc':<10} | {'Error':<10}")
    print("-" * 35)

    # Simuliamo l'aggiunta progressiva di stimatori
    for t in range(1, len(adaboost_model.models) + 1):
        # Creiamo un "modello parziale" che usa solo i primi 't' stimatori
        # (Per farlo senza riscrivere la classe, possiamo temporaneamente 
        # troncare le liste alphas e models)
        original_alphas = adaboost_model.alphas
        original_models = adaboost_model.models
        
        adaboost_model.alphas = original_alphas[:t]
        adaboost_model.models = original_models[:t]
        
        # Predizione e calcolo accuratezza
        train_preds = adaboost_model.predict(X_train)
        test_preds = adaboost_model.predict(X_test)
        
        acc_train = accuracy_score(y_train, train_preds)
        acc_test = accuracy_score(y_test, test_preds)
        
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        
        print(f"{t:<5} | {acc_train:<10.4f} | {acc_test:<10.3f} | {1 - acc_test:<10.3f}")
        
        # Ripristiniamo per la prossima iterazione
        adaboost_model.alphas = original_alphas
        adaboost_model.models = original_models
    
    