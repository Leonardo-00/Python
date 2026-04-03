"""
Naive Bayes categorico pesato, pensato come weak learner per AdaBoost.

Questo modello assume feature discrete/categoriali (interi o simboli discreti) e
supporta sample weights, quindi puo essere addestrato sui pesi aggiornati da
AdaBoost a ogni iterazione.

Notazione usata:
    - x: esempio da classificare
    - x_j: valore della feature j-esima
    - y in C: classe
    - w_i: peso del campione i-esimo (da AdaBoost)
    - N_w(c): somma pesata dei campioni con classe c
    - N_w(c,j,v): somma pesata dei campioni con classe c e x_j=v

Score in log-space:
    score(c | x) = log P(c) + sum_j log P(x_j | c)

Stime con smoothing additivo alpha:
    P(c) = (N_w(c) + alpha) / (sum_i w_i + alpha * |C|)
    P(x_j=v | c) = (N_w(c,j,v) + alpha) / (N_w(c) + alpha * V_j)
    dove V_j e la cardinalita della feature j.
"""

import numpy as np
from collections import defaultdict


class WeightedCategoricalNB:
    """Naive Bayes categorico con conteggi pesati e smoothing di Laplace.

    Parametri:
        alpha (float): coefficiente di smoothing additivo.

    Attributi principali:
        alpha: smoothing additivo (Laplace/Add-alpha).
        classes_: insieme C delle classi osservate in training.
        class_log_prior_: mappa c -> log P(c).
        feature_cardinality_: mappa j -> V_j (numero valori distinti della
            feature j).
        log_likelihood_: mappa c -> {(j, v): log P(x_j=v | c)}.
        class_weight_sum_: mappa c -> N_w(c), usata anche nel fallback per
            valori non osservati in training.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = {}
        self.feature_cardinality_ = {}
        self.log_likelihood_ = defaultdict(dict)
        self.class_weight_sum_ = {}

    def fit(self, X, y, sample_weight=None):
        """Stima prior e likelihood con conteggi pesati.

        Args:
            X: matrice (n_samples, n_features) con feature discrete.
            y: vettore etichette (n_samples,).
            sample_weight: vettore w di dimensione n_samples. Se None, usa
                pesi uniformi 1/n_samples.

        Variabili locali principali:
            n_samples: numero di campioni.
            n_features: numero di feature.
            total_w: somma dei pesi (normalmente 1 dopo normalizzazione).
            c: classe corrente.
            idx_c: maschera booleana dei campioni con y==c.
            w_c: pesi dei soli campioni della classe c.
            X_c: sotto-matrice X dei soli campioni della classe c.
            Nw_c: N_w(c) = sum_i w_i * I(y_i=c).
            j: indice feature.
            vals_j: valori osservati della feature j nel training.
            Vj: cardinalita della feature j.
            denom: denominatore N_w(c) + alpha * V_j.
            xj_c: colonna j di X_c.
            v: valore discreto della feature j.
            num: numeratore N_w(c,j,v) + alpha.

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float) / n_samples
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            # Normalizziamo i pesi per evitare problemi numerici e garantire che sommino a 1.
            sample_weight = sample_weight / sample_weight.sum()

        self.classes_ = np.unique(y)

        # Cardinalità per feature (usata nello smoothing)
        # calcola per ciascuna feature j il numero di valori distinti osservati in training
        for j in range(n_features):
            col = X[:, j]
            self.feature_cardinality_[j] = len(np.unique(col[~np.isnan(col)]))

        # Somma pesata totale dei campioni, usata nei calcoli di prior e fallback.
        total_w = sample_weight.sum()

        for c in self.classes_:
            
            # idx_c: maschera booleana per selezionare i campioni di classe c
            idx_c = (y == c)
            
            # w_c: pesi dei campioni di classe c
            w_c = sample_weight[idx_c]
            
            # X_c: sotto-matrice di X con solo i campioni di classe c
            X_c = X[idx_c]

            Nw_c = w_c.sum()
            # Log-prior con smoothing, calcolato sulla massa pesata di classe.
            self.class_weight_sum_[c] = Nw_c
            self.class_log_prior_[c] = np.log((Nw_c + self.alpha) / (total_w + self.alpha * len(self.classes_)))

            for j in range(n_features):
                
                # vals_j: valori distinti NON-NaN osservati della feature j
                all_vals = X[:, j]
                vals_j = np.unique(all_vals[~np.isnan(all_vals)])
                
                # Vj: cardinalita della feature j, usata nello smoothing
                Vj = self.feature_cardinality_[j]
                
                xj_c = X_c[:, j]
                # Seleziona solo gli indici dove il dato NON è mancante (NaN)
                valid_idx = ~np.isnan(xj_c)
                
                denom = np.sum(w_c[valid_idx]) + self.alpha * Vj

                # Calcoliamo log P(x_j=v | c) per ogni valore v osservato della feature j.
                
                # xj_c: colonna j di X_c, usata per contare i campioni di classe c con x_j=v
                xj_c = X_c[:, j]
                
                for v in vals_j:
                    # N_w(c,j,v): massa pesata dei campioni di classe c con x_j=v.
                    num = np.sum(w_c[xj_c == v]) + self.alpha
                    self.log_likelihood_[c][(j, v)] = np.log(num / denom)

        return self

    def _log_prob_class(self, x, c):
        """Calcola score(c|x) non normalizzato in log-space.

        Variabili:
            x: singolo esempio da classificare.
            c: classe candidata.
            s: accumulatore dello score logaritmico.
            key: coppia (j, v) usata come chiave della likelihood.

        Nota:
            Se (j, v) non e presente per la classe c, si usa la probabilita
            di fallback alpha / (N_w(c) + alpha * V_j), coerente con
            smoothing additivo.
        """
        s = self.class_log_prior_[c]
        for j, v in enumerate(x):
            if np.isnan(v):
                continue
            key = (j, v)
            if key in self.log_likelihood_[c]:
                s += self.log_likelihood_[c][key]
            else:
                # fallback per valore mai visto
                Vj = self.feature_cardinality_[j]
                denom = self.class_weight_sum_[c] + self.alpha * Vj
                s += np.log(self.alpha / denom)
        return s

    def predict_proba(self, X):
        """Ritorna P(C=classe_positiva|x) per ogni esempio."""
        X = np.asarray(X)
        probas = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            log_scores = {c: self._log_prob_class(x, c) for c in self.classes_}
            max_log = max(log_scores.values())
            log_denom = max_log + np.log(sum(np.exp(s - max_log) for s in log_scores.values()))
            probas[i] = np.exp(log_scores[self.classes_[-1]] - log_denom)
        return probas

    def predict(self, X):
        """Predice la classe con score massimo per ciascun esempio.

        Variabili:
            X: matrice esempi da predire.
            preds: lista delle classi predette.
            x: singolo esempio.
            scores: mappa classe->score(c|x) in log-space.
        """
        X = np.asarray(X)
        preds = []
        for x in X:
            scores = {c: self._log_prob_class(x, c) for c in self.classes_}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)