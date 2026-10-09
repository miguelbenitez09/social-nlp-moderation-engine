"""
Microservicio REST FastAPI de Moderacion y Clasificacion de Contenido
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Expone contratos OpenAPI para clasificacion individual, por lotes,
metricas de benchmark y streaming multi-plataforma.
"""

import os
import sys
import time
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.text_cleaner import TextCleaner
from src.models.production_pipeline import ProductionPipeline
from src.models.semantic_moderator import SemanticModerator
from src.models.benchmark import run_comprehensive_benchmark
from src.scrapers.social_feed_simulator import SocialFeedSimulator, SAMPLE_STREAMS

app = FastAPI(
    title="Social Media Moderation & Sentiment API",
    description="Motor de Inferencia y Clasificacion de Mensajes Multi-Plataforma (Twitter/X, Instagram, Facebook, YouTube) v1.0.0 por Ing. Miguel Antonio Benitez Gonzalez (UTP)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializacion de componentes de inferencia
cleaner = TextCleaner()
semantic_engine = SemanticModerator()
prod_pipeline: Optional[ProductionPipeline] = None

MODEL_PATH = "models/production_pipeline.joblib"
BENCHMARK_PATH = "models/benchmark_report.json"


def get_production_pipeline() -> ProductionPipeline:
    """Carga o entrena de forma diferida el pipeline de produccion."""
    global prod_pipeline
    if prod_pipeline is None:
        if os.path.exists(MODEL_PATH):
            prod_pipeline = ProductionPipeline.load(MODEL_PATH)
        else:
            # Si no existe, ejecutar benchmark para entrenar y guardar
            run_comprehensive_benchmark(BENCHMARK_PATH)
            prod_pipeline = ProductionPipeline.load(MODEL_PATH)
    return prod_pipeline


class ModerationRequest(BaseModel):
    text: str = Field(..., example="Excelente analisis sobre la situacion economica actual 👏", min_length=1)
    platform: Optional[str] = Field("twitter", example="twitter", description="Plataforma de origen: twitter, instagram, facebook, youtube")
    engine: Optional[str] = Field("hybrid", example="hybrid", description="Motor de inferencia: 'production', 'semantic', o 'hybrid'")


class ModerationResponse(BaseModel):
    verdict: str
    is_harmful: bool
    confidence: float
    category: str
    platform: str
    latency_ms: float
    meta: Dict[str, Any]
    engine_used: str


class BatchModerationRequest(BaseModel):
    items: List[ModerationRequest]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "social-nlp-moderation-engine",
        "version": "1.0.0",
        "author": "Ing. Miguel Antonio Benitez Gonzalez (UTP)",
        "model_loaded": os.path.exists(MODEL_PATH)
    }


@app.get("/api/v1/platforms")
def list_platforms():
    return {
        "supported_platforms": list(SAMPLE_STREAMS.keys()),
        "description": "Plataformas con soporte de normalizacion contextual y analisis semantico"
    }


@app.post("/api/v1/moderate", response_model=ModerationResponse)
def moderate_text(request: ModerationRequest):
    t0 = time.perf_counter()
    pipeline = get_production_pipeline()
    
    clean_text = cleaner.clean(request.text)
    metadata = cleaner.extract_metadata(request.text)

    # 1. Analisis con motor semantico
    semantic_res = semantic_engine.analyze(request.text)

    # 2. Inferencia de ML en produccion
    probs = pipeline.predict_proba([clean_text])[0]
    prob_harmful = float(probs[1])
    prob_clean = float(probs[0])

    # 3. Decision Hibrida
    if request.engine == "semantic":
        verdict = semantic_res["verdict"]
        is_harmful = semantic_res["is_harmful"]
        confidence = semantic_res["toxicity_score"]
        category = semantic_res["category"]
        engine_used = "semantic_lexicon"
    elif request.engine == "production":
        is_harmful = prob_harmful >= 0.5
        verdict = "TOXIC/HARMFUL" if is_harmful else "CLEAN"
        confidence = prob_harmful if is_harmful else prob_clean
        category = "toxic" if is_harmful else "clean"
        engine_used = "production_ml"
    else:
        # Enfoque Hibrido: Si la semantica detecta odio o spam explicito, prevalece;
        # de lo contrario, pondera con el modelo estadistico de produccion
        if semantic_res["is_harmful"] and semantic_res["toxicity_score"] >= 0.8:
            verdict = semantic_res["verdict"]
            is_harmful = True
            confidence = semantic_res["toxicity_score"]
            category = semantic_res["category"]
            engine_used = "hybrid (semantic_override)"
        else:
            is_harmful = prob_harmful >= 0.45 or semantic_res["is_harmful"]
            confidence = max(prob_harmful, semantic_res["toxicity_score"]) if is_harmful else prob_clean
            category = semantic_res["category"] if semantic_res["is_harmful"] else ("toxic" if is_harmful else "clean")
            verdict = category.upper() if is_harmful else "CLEAN"
            engine_used = "hybrid_consensus"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ModerationResponse(
        verdict=verdict,
        is_harmful=is_harmful,
        confidence=round(confidence, 4),
        category=category,
        platform=request.platform or "general",
        latency_ms=round(latency_ms, 2),
        meta={
            "metadata": metadata,
            "semantic_triggers": semantic_res["matched_triggers"],
            "ml_probability_harmful": round(prob_harmful, 4)
        },
        engine_used=engine_used
    )


@app.post("/api/v1/moderate/batch")
def moderate_batch(batch: BatchModerationRequest):
    results = [moderate_text(item) for item in batch.items]
    return {
        "count": len(results),
        "results": results
    }


@app.get("/api/v1/benchmark")
def get_benchmark_report():
    if not os.path.exists(BENCHMARK_PATH):
        run_comprehensive_benchmark(BENCHMARK_PATH)
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/v1/feed/simulate")
def simulate_feed(platform: str = Query("twitter", enum=["twitter", "instagram", "facebook", "youtube"]), limit: int = 5):
    simulator = SocialFeedSimulator(platform=platform)
    stream = simulator.fetch_stream(limit=limit)
    moderated_stream = []
    
    for item in stream:
        req = ModerationRequest(text=item["text"], platform=item["platform"], engine="hybrid")
        res = moderate_text(req)
        item["moderation"] = res.dict()
        moderated_stream.append(item)
        
    return {
        "platform": platform,
        "items": moderated_stream
    }
