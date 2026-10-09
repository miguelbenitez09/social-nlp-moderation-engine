"""
Generador y Consolidado de Dataset Multi-Plataforma de Redes Sociales
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Genera y valida un corpus balanceado de comentarios y publicaciones de:
- Twitter / X
- Instagram
- Facebook
- YouTube
con anotaciones binarias (0 = Limpio / Aceptable, 1 = Toxico / Ofensivo / Spam)
y categorias detalladas (clean, toxic, hate_speech, spam).
"""

import os
import pandas as pd
from typing import Tuple

DATASET_SAMPLES = [
    # Twitter / X
    ("Excelente analisis sobre la situacion economica actual, muy certero el punto de vista 👏", "twitter", 0, "clean"),
    ("Increible partido hoy, que bien jugo el equipo! Vamos por la final! 🔥⚽", "twitter", 0, "clean"),
    ("La aplicacion se cierra cada vez que intento abrir el menu de ajustes, favor revisar", "twitter", 0, "clean"),
    ("Alguien sabe a que hora abre el metro de Panama los domingos? Gracias de antemano", "twitter", 0, "clean"),
    ("Que conferencia tan enriquecedora, felicidades a todos los organizadores y ponentes", "twitter", 0, "clean"),
    ("Eres un idiota inservible, ojala te despidan pedazo de basura @usuario 🤬", "twitter", 1, "toxic"),
    ("Gente como tu no deberia existir, das asco total, desaparece del mundo", "twitter", 1, "hate_speech"),
    ("GANA $5000 DOLARES AL DIA DESDE CASA SIN HACER NADA HAZ CLIC EN EL LINK YA http://bit.ly/scam99", "twitter", 1, "spam"),
    ("CALLATE LA BOCA ESTUPIDO DE MIERDA NO SABES NADA 😡😡😡", "twitter", 1, "toxic"),
    ("Que asco de ser humano, eres una plaga para la sociedad", "twitter", 1, "hate_speech"),
    ("Nueva publicacion sobre machine learning e inteligencia artificial en nuestro blog", "twitter", 0, "clean"),
    ("Gracias al equipo de soporte por resolver mi incidencia tan rapido!", "twitter", 0, "clean"),
    ("Maldito ladron corrupto deberias estar en la carcel pudriendote", "twitter", 1, "toxic"),
    ("SIGUEME Y TE SIGO! Promociona tu perfil gratis entra a este link ahora mismo", "twitter", 1, "spam"),
    ("Que hermosa vista del atardecer desde la cinta costera hoy 😍", "twitter", 0, "clean"),

    # Instagram
    ("Que foto tan hermosa! Me encanta la iluminacion y los colores 😍❤️", "instagram", 0, "clean"),
    ("Felicidades por tu graduacion amiga, te mereces todo el exito del mundo! 🎓✨", "instagram", 0, "clean"),
    ("Donde compraste esa chaqueta? Esta increible el diseño!", "instagram", 0, "clean"),
    ("Excelente receta, la prepare el fin de semana y a mi familia le fascino!", "instagram", 0, "clean"),
    ("Que viaje tan espectacular, disfruten mucho las vacaciones!", "instagram", 0, "clean"),
    ("Estas horrible das asco deberias operarte la cara payasa 🤡🤮", "instagram", 1, "toxic"),
    ("Muerete fea de mierda nadie te quiere aqui", "instagram", 1, "hate_speech"),
    ("Invierte 50 dolares y retira 1000 en 24 horas escribeme al DM garantizado 100%", "instagram", 1, "spam"),
    ("Que cuerpo tan deforme das verguenza ajena ridicula", "instagram", 1, "toxic"),
    ("SEGUIDORES REALES Y LIKES AL INSTANTE! Visita mi perfil para ofertas exclusivas", "instagram", 1, "spam"),
    ("El mejor cafe de la ciudad sin duda alguna, servicio impecable", "instagram", 0, "clean"),
    ("Mucho talento en esta sesion fotografica, mis respetos!", "instagram", 0, "clean"),
    ("Gorda asquerosa deja de subir fotos dando lastima", "instagram", 1, "toxic"),
    ("Cryptomonedas faciles trabajando 1 hora diaria telegram link en bio", "instagram", 1, "spam"),
    ("Hermoso mensaje de inspiracion, justo lo que necesitaba leer hoy", "instagram", 0, "clean"),

    # Facebook
    ("Buen dia vecinos, informo que ya se restablecio el servicio de agua en el sector", "facebook", 0, "clean"),
    ("Comparto esta iniciativa de reforestacion comunitaria para este sabado, estan todos invitados", "facebook", 0, "clean"),
    ("Muchas gracias al doctor y enfermeros del hospital por su gran vocacion y atencion", "facebook", 0, "clean"),
    ("Feliz cumpleaños tio querido, que Dios lo bendiga con mucha salud y años de vida", "facebook", 0, "clean"),
    ("Alguien recomienda un buen taller mecanico en la zona de Betania?", "facebook", 0, "clean"),
    ("Viejos ignorantes comunistas de mierda deberian matarlos a todos", "facebook", 1, "hate_speech"),
    ("Eres un parasito social retrasado mental no sirves para nada", "facebook", 1, "toxic"),
    ("OFERTA DE EMPLEO URGENTE EMPACANDO DESDE SU CASA 500 SEMANALES CONTACTARSE AL INBOX", "facebook", 1, "spam"),
    ("Ojala te atropellen y quedes invalido por desgraciado", "facebook", 1, "hate_speech"),
    ("PRESTAMOS INMEDIATOS SIN REVISION DE APC INTERES BAJO WHATSAPP YA", "facebook", 1, "spam"),
    ("Orgulloso del desempeño de mi hijo en las olimpiadas de matematicas", "facebook", 0, "clean"),
    ("Excelente explicacion sobre los requisitos del tramite municipal, muy claro", "facebook", 0, "clean"),
    ("Malditos criminales extranjeros fuera de nuestro pais plagas", "facebook", 1, "hate_speech"),
    ("Gana un iPhone 15 gratis compartiendo esta publicacion en 5 grupos", "facebook", 1, "spam"),
    ("Agradecido con Dios por un dia mas de vida y trabajo junto a mi familia", "facebook", 0, "clean"),

    # YouTube
    ("El mejor tutorial de Python que he visto, directo al grano y muy bien explicado, gracias!", "youtube", 0, "clean"),
    ("La edicion de video y el diseño sonoro de este documental son de otro nivel, felicidades", "youtube", 0, "clean"),
    ("Me ayudo muchisimo para aprobar mi examen de calculo, te ganaste un suscriptor!", "youtube", 0, "clean"),
    ("Cancion epica, me transporta a mi infancia cada vez que la escucho", "youtube", 0, "clean"),
    ("Gran debate respetuoso y con argumentos tecnicos solidos, excelente contenido", "youtube", 0, "clean"),
    ("Que video mas estupido y mediocre, el autor tiene retraso mental severo", "youtube", 1, "toxic"),
    ("Pedazo de subnormal no sabes ni hablar borra tu canal das pena", "youtube", 1, "toxic"),
    ("Subscribete a mi canal para ganar robux y diamantes gratis enlace fijado", "youtube", 1, "spam"),
    ("Eres una escoria humana ojala te de cancer a ti y a tu familia", "youtube", 1, "hate_speech"),
    ("TRADING BOT AUTOMATICO 99% ACIERTOS TELEGRAM @TRADER_PRO_SCAM", "youtube", 1, "spam"),
    ("En el minuto 14:25 esta la clave de todo el algoritmo, muy buen detalle", "youtube", 0, "clean"),
    ("Gracias por subir este contenido educativo gratuito para la comunidad", "youtube", 0, "clean"),
    ("Maldita porqueria de contenido, desuscribanse todos de este payaso", "youtube", 1, "toxic"),
    ("WHATSAPP HACKEADO VER CONVERSACIONES PRIVADAS LINK EN DESCRIPCION", "youtube", 1, "spam"),
    ("La calidad del audio mejoro bastante respecto al video anterior, buen trabajo", "youtube", 0, "clean"),
]


def build_and_save_dataset(output_dir: str = "data") -> Tuple[pd.DataFrame, str]:
    """Crea y serializa el dataset de moderacion en formatos CSV y Parquet."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(
        DATASET_SAMPLES,
        columns=["text", "platform", "label", "category"]
    )
    
    csv_path = os.path.join(output_dir, "social_media_moderation.csv")
    parquet_path = os.path.join(output_dir, "social_media_moderation.parquet")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass  # Parquet opcional si pyarrow no esta instalado
        
    return df, csv_path


if __name__ == "__main__":
    df, path = build_and_save_dataset()
    print(f"Dataset generado con {len(df)} registros en {path}")
    print(df["platform"].value_counts())
    print(df["category"].value_counts())
