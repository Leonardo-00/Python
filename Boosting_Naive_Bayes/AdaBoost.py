import numpy as np
from naive_bayes import WeightedCategoricalNB

class AdaBoostClassifier:
    def __init__(self, n_estimators=50, estimator=WeightedCategoricalNB, algorithm='discrete'):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.algorithm = algorithm  # 'discrete' (M1) o 'continuous' (Elkan 1997)
        self.alphas = []
        self.models = []
        self.classes_ = None

    def _initialize_weights(self, n_samples):
        """Inizializza i pesi in modo uniforme: 1/N."""
        return np.full(n_samples, (1.0 / n_samples))

    def _calculate_error(self, y_true, y_pred, weights):
        """Calcola l'errore pesato epsilon_t."""
        incorrect = (y_pred != y_true)
        epsilon = np.sum(weights[incorrect]) / np.sum(weights)
        # Clipping per stabilità numerica
        return np.clip(epsilon, 1e-12, 1.0 - 1e-12), incorrect

    def _calculate_alpha(self, epsilon):
        """Calcola l'importanza del classificatore (Standard AdaBoost)."""
        # Se epsilon >= 0.5, il modello è inutile (non meglio del caso casuale)
        if epsilon >= 0.5:
            return None
        return 0.5 * np.log((1.0 - epsilon) / epsilon)

    def _update_weights(self, weights, alpha, incorrect):
        """
        Aggiorna i pesi degli esempi.
        Usa 2*alpha sugli errori e 1 sui corretti (Asimmetrico stabile).
        """
        modifier = np.exp(2.0 * alpha * incorrect.astype(float))
        new_weights = weights * modifier
        return new_weights / np.sum(new_weights) # Normalizzazione

    def fit(self, X, y):
        self.alphas = []
        self.models = []
        self.classes_ = np.unique(y)
        
        weights = self._initialize_weights(X.shape[0])

        for t in range(self.n_estimators):
            # 1. Training del Weak Learner
            clf = self.estimator()
            clf.fit(X, y, sample_weight=weights)

            if self.algorithm == 'continuous':
                # Variante continua (Elkan 1997 / Freund & Schapire 1995):
                # il weak learner produce probabilita, l'errore e continuo.
                proba = clf.predict_proba(X)
                abs_diff = np.abs(y.astype(float) - proba)
                epsilon = np.clip(np.sum(weights * abs_diff), 1e-12, 1.0 - 1e-12)

                if epsilon >= 0.5:
                    if t == 0:
                        self.alphas.append(1e-6)
                        self.models.append(clf)
                    break

                beta = epsilon / (1.0 - epsilon)
                self.alphas.append(np.log(1.0 / beta))
                self.models.append(clf)
                weights = weights * (beta ** (1.0 - abs_diff))
                weights = weights / np.sum(weights)
            else:
                # 2. Valutazione
                preds = clf.predict(X)
                epsilon, incorrect = self._calculate_error(y, preds, weights)

                # 3. Calcolo importanza (Alpha)
                alpha = self._calculate_alpha(epsilon)
                if alpha is None: # Stop se il modello è troppo debole
                    if t == 0: # Gestione caso limite: almeno un modello
                        self.alphas.append(1e-6)
                        self.models.append(clf)
                    break

                # 4. Storage e Update
                self.alphas.append(alpha)
                self.models.append(clf)
                weights = self._update_weights(weights, alpha, incorrect)

    def predict(self, X):
        """Predizione finale (seleziona variante in base ad algorithm)."""
        if self.algorithm == 'continuous':
            # Media pesata delle probabilita, soglia a 0.5
            X = np.asarray(X)
            total_alpha = sum(self.alphas)
            weighted_proba = np.zeros(X.shape[0])
            for t in range(len(self.models)):
                weighted_proba += self.alphas[t] * self.models[t].predict_proba(X)
            return (weighted_proba / total_alpha >= 0.5).astype(int)

        # Voto a maggioranza pesato (Weighted Majority Voting)
        all_preds = np.array([m.predict(X) for m in self.models])

        final_predictions = []
        for i in range(X.shape[0]):
            votes = {c: 0.0 for c in self.classes_}
            for t in range(len(self.models)):
                pred_label = all_preds[t, i]
                votes[pred_label] += self.alphas[t]

            final_predictions.append(max(votes, key=votes.get))

        return np.array(final_predictions)