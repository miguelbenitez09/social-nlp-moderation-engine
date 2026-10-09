"""
Pruebas de Inferencia, Modelos y Microservicio REST
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)
"""

import pytest
from fastapi.testclient import TestClient
from src.models.baseline_scratch import ScratchTfidfVectorizer, ScratchMultinomialNB
from src.models.semantic_moderator import SemanticModerator
from src.serving.api import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_scratch_naive_bayes_math():
    docs = [
        ["bueno", "excelente", "feliz"],
        ["malo", "estupido", "mierda"],
        ["excelente", "proyecto", "bueno"],
        ["odio", "muerte", "basura"]
    ]
    labels = [0, 1, 0, 1]

    vec = ScratchTfidfVectorizer(min_df=1)
    X = vec.fit_transform(docs)
    assert X.shape[0] == 4
    assert X.shape[1] > 0

    nb = ScratchMultinomialNB(alpha=1.0)
    nb.fit(X, labels)

    preds = nb.predict(X)
    assert len(preds) == 4
    # Comprobar consistencia basica de prediccion
    assert preds[0] == 0
    assert preds[1] == 1


def test_semantic_moderator_detection():
    engine = SemanticModerator()
    
    # Texto limpio
    res_clean = engine.analyze("Muchas gracias por el contenido educativo")
    assert res_clean["verdict"] == "CLEAN"
    assert res_clean["is_harmful"] is False

    # Spam
    res_spam = engine.analyze("GANA 5000 DOLARES INVIERTE EN CRYPTO WHATSAPP YA")
    assert res_spam["is_harmful"] is True
    assert res_spam["category"] == "spam"

    # Odio / Toxicidad
    res_toxic = engine.analyze("Eres un idiota de mierda")
    assert res_toxic["is_harmful"] is True


def test_fastapi_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_fastapi_moderate_endpoint(api_client):
    payload = {
        "text": "Excelente analisis sobre la situacion economica actual 👏",
        "platform": "twitter",
        "engine": "hybrid"
    }
    response = api_client.post("/api/v1/moderate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "CLEAN"
    assert data["is_harmful"] is False
    assert data["latency_ms"] >= 0.0
