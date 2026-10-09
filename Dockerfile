# Dockerfile para Social NLP Moderation Engine v1.0.0
# Desarrollado por Ing. Miguel Antonio Benitez Gonzalez (UTP)
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fuente
COPY . .

# Entrenar modelo y generar benchmark si no existen
RUN python src/models/benchmark.py

# Exponer puertos para FastAPI (8000) y Streamlit (8501)
EXPOSE 8000 8501

# Comando por defecto: arrancar FastAPI
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
