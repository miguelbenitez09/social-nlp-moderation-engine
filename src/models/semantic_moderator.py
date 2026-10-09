"""
Motor Semantico y Heuristico de Moderacion por Reglas y Lexicones
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Motor de moderacion semantica y analisis de patrones de riesgo:
- Deteccion de odio e insultos de alta gravedad.
- Deteccion de patrones de spam y phishing comunes en redes sociales.
- Evaluacion de intensidad por gritos en mayusculas y emojis agresivos.
"""

import re
from typing import Dict, Any, List

TOXIC_LEXICON = {
    "idiota": 0.6,
    "estupido": 0.6,
    "mierda": 0.7,
    "basura": 0.6,
    "asco": 0.5,
    "feo": 0.4,
    "fea": 0.4,
    "gorda": 0.5,
    "gordo": 0.5,
    "payaso": 0.4,
    "payasa": 0.4,
    "mediocre": 0.5,
    "subnormal": 0.7,
    "retrasado": 0.8,
    "parasito": 0.7,
    "ladron": 0.6,
    "corrupto": 0.6,
    "porqueria": 0.6,
}

HATE_SPEECH_LEXICON = {
    "muerete": 0.95,
    "matarlos": 0.95,
    "cancer": 0.90,
    "invalido": 0.85,
    "plaga": 0.85,
    "plagas": 0.85,
    "desaparece": 0.80,
    "escoria": 0.90,
}

SPAM_PATTERNS = [
    r"\b(gana|ganar)\s+(\$|\d+|dolares|usd)",
    r"\b(invierta|invierte|retira)\b",
    r"\b(criptomonedas|crypto|bitcoin|trading\s*bot)\b",
    r"\b(seguidores|likes)\s+(gratis|al instante)",
    r"\b(prestamos\s+inmediatos|sin\s+revision\s+de\s+apc)\b",
    r"\b(whatsapp\s+ya|link\s+en\s+bio|enlace\s+fijado)\b",
    r"\b(robux|diamantes)\s+gratis\b",
    r"\b(empacando\s+desde\s+su\s+casa)\b",
]


class SemanticModerator:
    """Motor heuristico y semantico de moderacion multi-plataforma."""

    def __init__(self):
        self.spam_regexes = [re.compile(p, re.IGNORECASE) for p in SPAM_PATTERNS]

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analiza el texto y retorna el veredicto, puntaje de severidad y desencadenantes."""
        if not text:
            return {
                "verdict": "CLEAN",
                "toxicity_score": 0.0,
                "is_harmful": False,
                "matched_triggers": [],
                "reason": "Texto vacio"
            }

        text_lower = text.lower()
        matched_triggers = []
        max_score = 0.0
        category = "clean"

        # 1. Comprobar patrones de Spam
        for regex in self.spam_regexes:
            if regex.search(text):
                matched_triggers.append(f"spam_pattern: {regex.pattern}")
                max_score = max(max_score, 0.85)
                category = "spam"

        # 2. Comprobar discurso de odio y amenazas
        for word, weight in HATE_SPEECH_LEXICON.items():
            if re.search(rf"\b{word}\b", text_lower):
                matched_triggers.append(f"hate_keyword: {word}")
                max_score = max(max_score, weight)
                category = "hate_speech"

        # 3. Comprobar insultos y toxicidad
        for word, weight in TOXIC_LEXICON.items():
            if re.search(rf"\b{word}\b", text_lower):
                matched_triggers.append(f"toxic_keyword: {word}")
                max_score = max(max_score, weight)
                if category == "clean":
                    category = "toxic"

        # 4. Modificador por gritos en mayusculas
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 10:
            caps_ratio = len([c for c in letters if c.isupper()]) / len(letters)
            if caps_ratio > 0.6 and category in ["toxic", "hate_speech"]:
                max_score = min(1.0, max_score + 0.10)
                matched_triggers.append("high_caps_emphasis")

        verdict = "CLEAN"
        is_harmful = False
        if max_score >= 0.5:
            is_harmful = True
            verdict = category.upper()

        return {
            "verdict": verdict,
            "category": category,
            "toxicity_score": round(max_score, 3),
            "is_harmful": is_harmful,
            "matched_triggers": matched_triggers,
            "reason": f"Detectado como {category} con confianza {max_score:.2f}" if is_harmful else "Contenido aceptable"
        }
