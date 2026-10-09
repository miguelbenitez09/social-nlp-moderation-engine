"""
Pipeline de Produccion Optimizado con Scikit-Learn y N-Gramas
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Pipeline optimizado para baja latencia en produccion:
- Vectorizador TF-IDF con n-gramas (1, 2) y sublinear TF scaling.
- Regresion Logistica regularizada con calibracion de probabilidad.
- Serializacion y persistencia binaria con joblib.
"""

import os
import joblib
from typing import List, Tuple, Any, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


class ProductionPipeline:
    """Pipeline de clasificacion para despliegue en microservicio REST."""

    def __init__(self, c_param: float = 2.0, max_features: int = 5000):
        self.c_param = c_param
        self.max_features = max_features
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=self.max_features,
                strip_accents="unicode"
            )),
            ("classifier", LogisticRegression(
                C=self.c_param,
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[int]) -> "ProductionPipeline":
        """Entrena el pipeline completo."""
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predice clases binarias (0 = Limpio, 1 = Toxico/Spam)."""
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Retorna probabilidades calibradas [P(Limpio), P(Toxico)]."""
        return self.pipeline.predict_proba(texts)

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Calcula metricas formales de evaluacion."""
        preds = self.predict(texts)
        probs = self.predict_proba(texts)[:, 1]
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        report = classification_report(labels, preds, output_dict=True)
        return {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "detailed_report": report
        }

    def save(self, filepath: str) -> None:
        """Persiste el artefacto entrenado en disco."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProductionPipeline":
        """Carga el modelo serializado desde disco."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        instance.is_trained = True
        return instance
