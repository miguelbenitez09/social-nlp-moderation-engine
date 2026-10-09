# 🛡️ Social Media NLP Moderation & Toxicity Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Pure_Math_Scratch-informational.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
[![Autor](https://img.shields.io/badge/Autor-Ing._Miguel_Antonio_Benítez_González_(UTP)-informational.svg)](https://github.com/miguelbenitez09)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Firma Oficial:** **`Social NLP Moderation Engine v1.0.0 • developed by Miguel Benitez`**  
> **Framework de Procesamiento de Lenguaje Natural (NLP) para moderación de contenido y detección de toxicidad, discurso de odio y spam en redes sociales (Twitter/X, Instagram, Facebook y YouTube), combinando un clasificador matemático desarrollado desde cero con un pipeline optimizado de producción.**

---

## 👨‍💻 Autor

**Ing. Miguel Antonio Benítez González** (Universidad Tecnológica de Panamá - UTP)
- 🎓 **Título:** Ingeniero en Sistemas y Computación
- 📜 **Idoneidad Profesional:** Junta Técnica de Ingeniería y Arquitectura de Panamá (JTIA)
- 📧 **Email:** mbenitezg01@gmail.com
- 💻 **GitHub:** [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)
- 💼 **LinkedIn:** [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto y Enfoque](#-descripción-del-proyecto-y-enfoque)
2. [Arquitectura Dual de Modelado](#-arquitectura-dual-de-modelado)
3. [Cobertura Multi-Plataforma](#-cobertura-multi-plataforma)
4. [Dataset, Ética y Régimen Normativo](#-dataset-ética-y-régimen-normativo)
5. [Resultados Empíricos y Benchmarking](#-resultados-empíricos-y-benchmarking)
6. [Estructura del Repositorio](#-estructura-del-repositorio)
7. [Instalación y Despliegue](#-instalación-y-despliegue)
8. [API REST (FastAPI)](#-api-rest-fastapi)
9. [Dashboard Interactivo (Streamlit)](#-dashboard-interactivo-streamlit)
10. [Licencia](#-licencia)

---

## 📖 Descripción del Proyecto y Enfoque

En el entorno digital contemporáneo, la proliferación de contenido nocivo (ciberacoso, toxicidad, estafas financieras y desinformación) en redes sociales exige sistemas automatizados de moderación que operen con latencia extremadamente baja y alta confiabilidad.

Este proyecto fue concebido y desarrollado desde mi perspectiva como **Ingeniero en Sistemas y Computación de la Universidad Tecnológica de Panamá (UTP)** para abordar dos desafíos esenciales:
1. **Rigor Teórico y Algorítmico:** Demostrar cómo se construyen las operaciones vectoriales de álgebra lineal y probabilidad condicional (TF-IDF y Naive Bayes Multinomial con suavizado de Laplace) **completamente desde cero** usando únicamente NumPy, sin depender de bibliotecas de alto nivel.
2. **Ingeniería de Producción (MLOps):** Empaquetar un pipeline de alto rendimiento utilizando Scikit-Learn con enriquecimiento de n-gramas sublineales, calibración de probabilidad, microservicio REST OpenAPI con FastAPI, interfaz visual en Streamlit y contenerización Docker.

---

## 🏗️ Arquitectura Dual de Modelado

El motor opera bajo un diseño desacoplado que permite comparar y consensuar tres estrategias de inferencia:

```
                                 ┌──────────────────────────────────┐
                                 │   Texto Crudo Red Social         │
                                 │  (Twitter, IG, FB, YouTube)      │
                                 └─────────────────┬────────────────┘
                                                   │
                                                   ▼
                                 ┌──────────────────────────────────┐
                                 │   Preprocesador y Normalizador   │
                                 │   (Regex, Emojis, Metadatos)     │
                                 └─────────┬──────────────┬─────────┘
                                           │              │
                   ┌───────────────────────┘              └───────────────────────┐
                   ▼                                                              ▼
┌──────────────────────────────────────┐                      ┌──────────────────────────────────────┐
│  Rama 1: Baseline Matemático Scratch │                      │  Rama 2: Pipeline de Producción MLOps│
│  - ScratchTfidfVectorizer (NumPy)    │                      │  - TfidfVectorizer (N-Grams 1,2)     │
│  - ScratchMultinomialNB (Laplace)    │                      │  - LogisticRegression Calibrada      │
│  - Latencia: < 0.05 ms por muestra   │                      │  - Serializado en joblib             │
└──────────────────┬───────────────────┘                      └──────────────────┬───────────────────┘
                   │                                                             │
                   └───────────────────────┬─────────────────────────────────────┘
                                           │
                                           ▼
                                 ┌──────────────────────────────────┐
                                 │  Rama 3: Motor Semántico Heurístico│
                                 │  (Lexicones de Odio, Spam & Phish│
                                 └─────────────────┬────────────────┘
                                                   │
                                                   ▼
                                 ┌──────────────────────────────────┐
                                 │  Motor de Decisión Híbrido       │
                                 │  (Consenso y Mitigación de Error)│
                                 └─────────────────┬────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
        ┌──────────────────────────────────┐                ┌──────────────────────────────────┐
        │       Microservicio FastAPI      │                │       Dashboard Streamlit        │
        │       (src/serving/api.py)       │                │       (src/ui/app.py)            │
        └──────────────────────────────────┘                └──────────────────────────────────┘
```

---

## 🌐 Cobertura Multi-Plataforma

El sistema incorpora adaptaciones para cuatro canales clave:
- **Twitter / X:** Manejo de menciones `@usuario`, etiquetas `#hashtag`, URLs de acortadores y límites cortos de caracteres.
- **Instagram:** Énfasis en emojis de alta polaridad (positivos y descalificativos) y detección de spam de seguidores en biografía.
- **Facebook:** Discusiones extensas, detección de publicaciones de préstamos informales sin verificación crediticia y empleo falso.
- **YouTube:** Comentarios de video, insultos directos al creador, enlaces fijados de estafas (`robux`, trading bots).

---

## ⚖️ Dataset, Ética y Régimen Normativo

### Política de Obtención Ética de Datos
1. **Finalidad Exclusivamente Educativa y de Investigación:** Este desarrollo se realiza con propósitos estrictamente académicos para investigar la interacción entre procesamiento semántico y modelos estadísticos.
2. **Protección a la Infraestructura de Plataformas:** Se prohíbe el scraping abusivo o no controlado. Todo extractor o simulador incluye:
   - Retardos de cortesía humana (*jitter* aleatorio entre 1 y 3 segundos).
   - Respeto a los códigos de respuesta HTTP y directivas de cabeceras de usuario.
3. **Privacidad de Usuarios:** Los datos han sido despersonalizados y anonimizados, sustituyendo nombres reales y direcciones sensibles por identificadores sintéticos para proteger la identidad individual de acuerdo con las normativas vigentes de protección de datos personales.

---

## 📊 Resultados Empíricos y Benchmarking

Evaluación formal sobre partición estratificada de prueba (30% de datos independientes no vistos durante entrenamiento):

| Enfoque / Algoritmo | Exactitud (Accuracy) | F1-Score | Latencia por Muestra | Características del Motor |
| :--- | :---: | :---: | :---: | :--- |
| **Pipeline de Producción (Scikit-Learn)** | **94.4%** | **0.94** | 0.12 ms | TF-IDF N-Gramas (1,2) + Regresión Logística L2 |
| **Modelo Scratch desde Cero (NumPy Puro)** | **88.9%** | **0.89** | **0.03 ms** | TF-IDF algebraico + Naive Bayes con Laplace ($\alpha=1$) |
| **Motor Semántico Heurístico** | 83.3% | 0.83 | 0.08 ms | Lexicones ponderados de odio + Regex de phishing/spam |

> **Conclusión Técnica:** El pipeline de producción ofrece la mejor generalización en contexto complejo, mientras que el modelo implementado desde cero demuestra una velocidad extrema de inferencia sin requerir paquetes externos de ML, haciéndolo ideal para entornos embebidos o microservicios ultra-livianos.

---

## 📁 Estructura del Repositorio

```
modelo-clasificacion-mensajes_twitter/
├── data/                                 # Datasets limpios (CSV y Parquet)
│   └── social_media_moderation.csv
├── models/                               # Artefactos y reportes serializados
│   ├── benchmark_report.json
│   └── production_pipeline.joblib
├── src/
│   ├── data/
│   │   └── builder.py                    # Generador y validador de datasets multi-plataforma
│   ├── models/
│   │   ├── baseline_scratch.py           # TF-IDF y Naive Bayes implementado desde cero
│   │   ├── production_pipeline.py        # Pipeline de producción optimizado con Scikit-Learn
│   │   ├── semantic_moderator.py         # Motor semántico por reglas y lexicones
│   │   └── benchmark.py                  # Evaluador empírico comparativo
│   ├── preprocessing/
│   │   └── text_cleaner.py               # Normalización, regex y mapeo de emojis
│   ├── scrapers/
│   │   └── social_feed_simulator.py      # Simulador de ingestión de flujos en tiempo real
│   ├── serving/
│   │   └── api.py                        # Microservicio REST FastAPI con OpenAPI
│   └── ui/
│       └── app.py                        # Dashboard interactivo Streamlit
├── tests/
│   ├── test_text_cleaner.py              # Tests unitarios del preprocesador
│   └── test_moderation_core.py           # Tests de inferencia y contratos de API
├── Dockerfile                            # Empaquetado de producción
├── requirements.txt                      # Dependencias fijadas
├── LICENSE                               # Licencia MIT (v1.0.0 Miguel Benitez)
└── README.md                             # Documentación técnica maestra
```

---

## 🚀 Instalación y Despliegue

### Requisitos Previos
- Python 3.11+
- Git

### 1. Clonar el Repositorio
```bash
git clone https://github.com/miguelbenitez09/social-nlp-moderation-engine.git
cd social-nlp-moderation-engine
```

### 2. Crear Entorno Virtual e Instalar Dependencias
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Ejecutar Pruebas Unitarias
```bash
pytest tests/ -v
```

---

## ⚡ API REST (FastAPI)

Para iniciar el servidor de inferencia:
```bash
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000 --reload
```
Acceder a la documentación Swagger interactiva en: `http://127.0.0.1:8000/docs`

### Ejemplo de Petición cURL:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/moderate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Excelente analisis sobre la situacion economica actual 👏", "platform": "twitter", "engine": "hybrid"}'
```

### Respuesta JSON:
```json
{
  "verdict": "CLEAN",
  "is_harmful": false,
  "confidence": 0.8245,
  "category": "clean",
  "platform": "twitter",
  "latency_ms": 1.45,
  "meta": {
    "metadata": {
      "length": 62,
      "has_urls": false,
      "has_mentions": false,
      "caps_ratio": 0.02
    },
    "ml_probability_harmful": 0.0821
  },
  "engine_used": "hybrid_consensus"
}
```

---

## 🖥️ Dashboard Interactivo (Streamlit)

Para lanzar la aplicación visual de moderación:
```bash
streamlit run src/ui/app.py
```
Abre automáticamente tu navegador en `http://localhost:8501`, permitiendo:
- Probar mensajes en vivo con visualización de señales de riesgo.
- Simular la llegada de publicaciones en tiempo real desde Twitter, Instagram, Facebook y YouTube.
- Analizar el cuadro comparativo de métricas de precisión y latencia.

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT** - consulta el archivo [LICENSE](LICENSE) para más detalles.  
`Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez UTP - MIT License`
