"""
Clasificador Naive Bayes y Vectorizador TF-IDF Desarrollado desde Cero (Scratch)
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Implementacion matematicamente rigurosa sin librerias de caja negra,
utilizando unicamente operaciones vectoriales con NumPy y estructuras nativas de Python:
- Vectorizador TF-IDF con frecuencia de termino normalizada y suavizado de IDF.
- Clasificador Multinomial Naive Bayes con suavizado de Laplace (alpha = 1.0).
- Normalizacion Softmax para generacion de probabilidades calibradas.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import numpy as np


class ScratchTfidfVectorizer:
    """Vectorizador TF-IDF implementado desde fundamentos algebraicos."""

    def __init__(self, min_df: int = 1, max_features: int = 2000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.num_docs: int = 0

    def fit(self, documents: List[List[str]]) -> "ScratchTfidfVectorizer":
        """Calcula el vocabulario y los pesos de frecuencia inversa de documento (IDF)."""
        self.num_docs = len(documents)
        df_counter: Counter = Counter()

        for doc in documents:
            unique_terms = set(doc)
            df_counter.update(unique_terms)

        # Filtrar terminos por frecuencia minima
        valid_terms = [
            (term, count) for term, count in df_counter.items()
            if count >= self.min_df
        ]

        # Seleccionar las caracteristicas mas frecuentes si se excede max_features
        valid_terms.sort(key=lambda x: x[1], reverse=True)
        if self.max_features and len(valid_terms) > self.max_features:
            valid_terms = valid_terms[:self.max_features]

        # Mapeo de terminos a indices
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(valid_terms)}
        vocab_size = len(self.vocabulary)

        # Calcular IDF suavizado: idf(t) = ln((1 + N) / (1 + df(t))) + 1
        self.idf = np.zeros(vocab_size)
        for term, idx in self.vocabulary.items():
            df = df_counter[term]
            self.idf[idx] = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0

        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transforma documentos en una matriz densa normalizada en norma L2."""
        if not self.vocabulary:
            raise ValueError("El vectorizador debe entrenarse con fit() antes de transformar.")

        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            if not doc:
                continue
            doc_len = len(doc)
            term_counts = Counter(doc)

            for term, count in term_counts.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    # Term Frequency normalizado: TF = count / doc_len
                    tf = count / doc_len
                    matrix[doc_idx, term_idx] = tf * self.idf[term_idx]

            # Normalizacion L2 euclidiana
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        return matrix

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        return self.fit(documents).transform(documents)


class ScratchMultinomialNB:
    """Clasificador Bayesiano Ingenuo Multinomial implementado desde cero."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Parametro de suavizado de Laplace
        self.classes: np.ndarray = np.array([])
        self.class_priors: np.ndarray = np.array([])
        self.feature_log_prob: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: List[int]) -> "ScratchMultinomialNB":
        """Calcula probabilidades previas P(C) y de verosimilitud condicional P(X_i | C)."""
        y_arr = np.array(y)
        self.classes = np.unique(y_arr)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_priors = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y_arr == c]
            n_c = X_c.shape[0]

            # Probabilidad previa P(C): log(n_c / total)
            self.class_priors[idx] = math.log(n_c / len(y_arr))

            # Conteo de caracteristicas con suavizado de Laplace
            total_feature_counts = np.sum(X_c, axis=0) + self.alpha
            total_tokens_in_class = np.sum(total_feature_counts)

            # Log-verosimilitud condicional: log(P(w | c))
            self.feature_log_prob[idx] = np.log(total_feature_counts / total_tokens_in_class)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula el logaritmo del posteri P(C | X) = log P(C) + X * log P(W | C)."""
        # Multiplicacion matricial: (n_samples, n_features) @ (n_features, n_classes)
        joint_log_likelihood = X @ self.feature_log_prob.T + self.class_priors
        return joint_log_likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula probabilidades posteriores calibradas utilizando Softmax numericamente estable."""
        log_prob = self.predict_log_proba(X)
        # Sustraccion del maximo para evitar desbordamiento numerico en exp
        max_log = np.max(log_prob, axis=1, keepdims=True)
        exp_prob = np.exp(log_prob - max_log)
        return exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice la clase que maximiza la probabilidad a posteriori (MAP)."""
        log_prob = self.predict_log_proba(X)
        best_indices = np.argmax(log_prob, axis=1)
        return self.classes[best_indices]
