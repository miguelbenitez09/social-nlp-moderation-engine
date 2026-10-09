"""
Pruebas Unitarias para el Modulo de Limpieza y Preprocesamiento de Texto
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)
"""

import pytest
from src.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner(preserve_emojis=True, lower_case=True)


def test_clean_urls_and_mentions(cleaner):
    text = "Mira este enlace https://example.com/test y consulta a @usuario_oficial"
    cleaned = cleaner.clean(text)
    assert "https" not in cleaned
    assert "usuario_oficial" not in cleaned
    assert "mira este enlace" in cleaned


def test_emoji_translation(cleaner):
    text = "Me encanta este proyecto ❤️😍"
    cleaned = cleaner.clean(text)
    assert "amor" in cleaned or "encanto" in cleaned


def test_metadata_extraction(cleaner):
    text = "OFERTA INCREIBLE GANA DINERO YA!! Consulta https://oferta.com @soporte"
    meta = cleaner.extract_metadata(text)
    assert meta["has_urls"] is True
    assert meta["has_mentions"] is True
    assert meta["exclamation_count"] >= 2
    assert meta["caps_ratio"] > 0.3


def test_tokenize(cleaner):
    text = "Excelente aplicacion de inteligencia artificial"
    tokens = cleaner.tokenize(text, remove_stopwords=True)
    assert "excelente" in tokens
    assert "inteligencia" in tokens
    assert "de" not in tokens  # Stopword eliminada
