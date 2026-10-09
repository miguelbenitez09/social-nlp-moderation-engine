"""
Modulo de Normalizacion y Preprocesamiento de Texto Multi-Plataforma
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Este modulo procesa comentarios y mensajes de redes sociales (Twitter/X, Instagram,
Facebook y YouTube) limpiando ruido sintactico mientras preserva senales semanticas
criticas como emojis de alta polaridad, signos de exclamacion y enfasis en mayusculas.
"""

import re
import unicodedata
from typing import Dict, Any, List

# Diccionario de emojis y su carga emocional o semantica
EMOJI_SENTIMENT_MAP = {
    "❤️": " amor ",
    "😍": " encanto ",
    "😊": " feliz ",
    "👍": " bien ",
    "🔥": " genial ",
    "👏": " aplauso ",
    "😡": " furia ",
    "🤬": " odio ",
    "🤮": " asco ",
    "💩": " basura ",
    "👎": " mal ",
    "🤡": " payaso ",
    "💀": " muerte ",
    "⚠️": " alerta ",
}

# Stopwords esenciales en espanol para filtrado opcional
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a",
    "en", "para", "por", "con", "y", "o", "u", "al", "se", "su", "sus", "es",
    "son", "que", "lo", "como", "mas", "pero", "este", "esta", "estos", "estas"
}

# Stopwords esenciales en ingles
ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "it", "this", "that", "these", "those"
}


class TextCleaner:
    """Limpiador y extractor de caracteristicas textuales multi-plataforma."""

    def __init__(self, preserve_emojis: bool = True, lower_case: bool = True):
        self.preserve_emojis = preserve_emojis
        self.lower_case = lower_case

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extrae metadatos estructurales del texto antes de la normalizacion."""
        if not text:
            return {
                "length": 0,
                "has_urls": False,
                "has_mentions": False,
                "has_hashtags": False,
                "caps_ratio": 0.0,
                "exclamation_count": 0,
                "question_count": 0,
            }

        length = len(text)
        letters = [c for c in text if c.isalpha()]
        caps = [c for c in letters if c.isupper()]
        caps_ratio = len(caps) / len(letters) if letters else 0.0

        return {
            "length": length,
            "has_urls": bool(re.search(r"https?://\S+|www\.\S+", text)),
            "has_mentions": bool(re.search(r"@\w+", text)),
            "has_hashtags": bool(re.search(r"#\w+", text)),
            "caps_ratio": round(caps_ratio, 4),
            "exclamation_count": text.count("!") + text.count("¡"),
            "question_count": text.count("?") + text.count("¿"),
        }

    def clean(self, text: str) -> str:
        """Limpia el texto eliminando enlaces, normalizando menciones y espacios."""
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # 1. Reemplazar emojis si esta habilitado
        if self.preserve_emojis:
            for emoji, translation in EMOJI_SENTIMENT_MAP.items():
                cleaned = cleaned.replace(emoji, translation)

        # 2. Reemplazar URLs por token neutro
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

        # 3. Remover o normalizar menciones de usuario (@usuario)
        cleaned = re.sub(r"@\w+", " ", cleaned)

        # 4. Remover hashtags (#tema -> tema)
        cleaned = re.sub(r"#(\w+)", r" \1 ", cleaned)

        # 5. Normalizar risas repetidas (jajaja, hahaha, lolol)
        cleaned = re.sub(r"\b(ja|je|ji|ha|he|lol)+\b", " risa ", cleaned, flags=re.IGNORECASE)

        # 6. Reducir caracteres repetidos (ej. 'buenooooo' -> 'bueno')
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        # 7. Minusculas si corresponde
        if self.lower_case:
            cleaned = cleaned.lower()

        # 8. Remover caracteres no alfanumericos pero mantener espacios
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

        # 9. Normalizar espacios en blanco
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """Tokeniza el texto limpio en una lista de palabras."""
        cleaned = self.clean(text)
        tokens = cleaned.split()
        if remove_stopwords:
            all_stops = SPANISH_STOPWORDS.union(ENGLISH_STOPWORDS)
            tokens = [t for t in tokens if t not in all_stops and len(t) > 1]
        return tokens
