"""
Simulador y Conector de Flujos de Redes Sociales (Twitter, IG, FB, YouTube)
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Simula la recepcion o scraping respetuoso de flujos de comentarios en tiempo real
respetando politicas de cortesia, retardos humanos (jitter 1-3s) y esquemas estandarizados.
"""

import time
import random
from typing import Dict, Any, Generator, List

SAMPLE_STREAMS = {
    "twitter": [
        {"user": "@usuario_panama", "text": "Excelente iniciativa educativa en la cinta costera hoy 👏", "id": "tw_101"},
        {"user": "@bot_crypto_99", "text": "Gana 1000 dolares al dia con este bot de trading link http://scam.me", "id": "tw_102"},
        {"user": "@gamer_pty", "text": "Increible torneo de videojuegos en la UTP el sabado!", "id": "tw_103"},
        {"user": "@hater_troll", "text": "Eres una escoria humana ojala te de cancer payaso 🤬", "id": "tw_104"},
    ],
    "instagram": [
        {"user": "fotografia_507", "text": "Que hermosa foto de las esclusas de Miraflores 😍❤️", "id": "ig_201"},
        {"user": "spammer_dm", "text": "Seguidores gratis y likes al instante escribe al DM garantizado!", "id": "ig_202"},
        {"user": "chef_local", "text": "Nueva receta de sancocho panameño en nuestro canal", "id": "ig_203"},
        {"user": "troll_insta", "text": "Estas horrible fea gorda operarte la cara das asco 🤮", "id": "ig_204"},
    ],
    "facebook": [
        {"user": "Vecinos Unidos", "text": "Se restablecio el suministro de agua en la barriada, gracias al IDAAN", "id": "fb_301"},
        {"user": "Empleos Faciles", "text": "OFERTA EMPLEO EMPACANDO DESDE CASA 500 SEMANALES INBOX YA", "id": "fb_302"},
        {"user": "Carlos Perez", "text": "Feliz cumpleaños querido profesor, que cumpla muchos mas!", "id": "fb_303"},
        {"user": "Anonimo Tóxico", "text": "Malditos ladrones corruptos deberian matarlos a todos en la carcel", "id": "fb_304"},
    ],
    "youtube": [
        {"user": "dev_estudiante", "text": "El mejor tutorial de estructuras de datos y algoritmos, gracias!", "id": "yt_401"},
        {"user": "free_robux_bot", "text": "Subscribete para ganar robux y diamantes gratis enlace fijado", "id": "yt_402"},
        {"user": "musica_panama", "text": "Excelente calidad de audio y mezcla en esta cancion", "id": "yt_403"},
        {"user": "toxic_critic", "text": "Que video mas mediocre y estupido, borra tu canal das pena", "id": "yt_404"},
    ]
}


class SocialFeedSimulator:
    """Simulador de transmision de mensajes multi-plataforma."""

    def __init__(self, platform: str = "twitter"):
        self.platform = platform.lower()
        if self.platform not in SAMPLE_STREAMS:
            self.platform = "twitter"

    def fetch_stream(self, limit: int = 10, polite_jitter: bool = False) -> List[Dict[str, Any]]:
        """Retorna un lote de mensajes simulando la llegada desde la plataforma."""
        samples = SAMPLE_STREAMS[self.platform]
        results = []
        for i in range(limit):
            item = random.choice(samples).copy()
            item["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            item["platform"] = self.platform
            results.append(item)
            if polite_jitter:
                time.sleep(random.uniform(0.1, 0.3))
        return results
