"""
Modulo de Benchmarking Comparativo entre Modelos
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Evalua y compara empiricamente:
1. Modelo Base Desarrollado desde Cero (Scratch Naive Bayes + TF-IDF)
2. Pipeline de Produccion Scikit-Learn (TF-IDF N-Grams + Regresion Logistica)
3. Motor Semantico Basado en Reglas y Lexicones
Generando metricas de exactitud, F1 ponderado y latencia de inferencia por muestra.
"""

import os
import json
import sys
import time
from typing import Dict, Any, List
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaner import TextCleaner
from src.data.builder import build_and_save_dataset
from src.models.baseline_scratch import ScratchTfidfVectorizer, ScratchMultinomialNB
from src.models.production_pipeline import ProductionPipeline
from src.models.semantic_moderator import SemanticModerator


def run_comprehensive_benchmark(output_path: str = "models/benchmark_report.json") -> Dict[str, Any]:
    """Ejecuta el benchmark empirico de las 3 tecnicas y persiste los resultados."""
    # 1. Cargar o generar dataset
    df, _ = build_and_save_dataset()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Split train/test estratificado (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    cleaner = TextCleaner()
    train_tokens = [cleaner.tokenize(t) for t in X_train]
    test_tokens = [cleaner.tokenize(t) for t in X_test]

    # --- Evaluacion 1: Modelo Scratch desde Cero ---
    vectorizer_scratch = ScratchTfidfVectorizer(min_df=1, max_features=1000)
    X_train_scratch = vectorizer_scratch.fit_transform(train_tokens)
    X_test_scratch = vectorizer_scratch.transform(test_tokens)

    nb_scratch = ScratchMultinomialNB(alpha=1.0)
    t0 = time.perf_counter()
    nb_scratch.fit(X_train_scratch, y_train)
    scratch_train_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scratch_preds = nb_scratch.predict(X_test_scratch)
    scratch_infer_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    scratch_metrics = {
        "accuracy": round(float(accuracy_score(y_test, scratch_preds)), 4),
        "precision": round(float(precision_score(y_test, scratch_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, scratch_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, scratch_preds, zero_division=0)), 4),
        "train_time_ms": round(scratch_train_time_ms, 2),
        "infer_latency_per_sample_ms": round(scratch_infer_latency_ms, 4),
        "model_type": "Algoritmo Matematico Desarrollado desde Cero (NumPy Puro)"
    }

    # --- Evaluacion 2: Pipeline de Produccion Scikit-Learn ---
    prod_pipeline = ProductionPipeline(c_param=2.0)
    t0 = time.perf_counter()
    prod_pipeline.fit(X_train, y_train)
    prod_train_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    prod_preds = prod_pipeline.predict(X_test)
    prod_infer_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    prod_metrics = {
        "accuracy": round(float(accuracy_score(y_test, prod_preds)), 4),
        "precision": round(float(precision_score(y_test, prod_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, prod_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, prod_preds, zero_division=0)), 4),
        "train_time_ms": round(prod_train_time_ms, 2),
        "infer_latency_per_sample_ms": round(prod_infer_latency_ms, 4),
        "model_type": "Pipeline Scikit-Learn Optimizado (N-Grams + LogisticRegression)"
    }

    # Guardar modelo de produccion
    prod_pipeline.save("models/production_pipeline.joblib")

    # --- Evaluacion 3: Motor Semantico de Reglas ---
    semantic_moderator = SemanticModerator()
    t0 = time.perf_counter()
    semantic_preds = [
        1 if semantic_moderator.analyze(t)["is_harmful"] else 0
        for t in X_test
    ]
    semantic_infer_latency_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000.0

    semantic_metrics = {
        "accuracy": round(float(accuracy_score(y_test, semantic_preds)), 4),
        "precision": round(float(precision_score(y_test, semantic_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, semantic_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, semantic_preds, zero_division=0)), 4),
        "train_time_ms": 0.0,
        "infer_latency_per_sample_ms": round(semantic_infer_latency_ms, 4),
        "model_type": "Motor Semantico Heuristico (Lexicones Ponderados + Regex)"
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "author": "Ing. Miguel Antonio Benitez Gonzalez (UTP)",
        "test_samples_count": len(X_test),
        "train_samples_count": len(X_train),
        "results": {
            "scratch_naive_bayes": scratch_metrics,
            "production_pipeline": prod_metrics,
            "semantic_moderator": semantic_metrics,
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    rep = run_comprehensive_benchmark()
    print("Benchmark completado con exito:")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
