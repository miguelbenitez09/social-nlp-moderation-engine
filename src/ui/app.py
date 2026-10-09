"""
Dashboard Interactivo de Moderacion de Redes Sociales (Streamlit)
Desarrollado v1.0.0 Miguel Benitez / Ing. Miguel Antonio Benitez Gonzalez (UTP)

Permite la evaluacion interactiva de comentarios de Twitter/X, Instagram,
Facebook y YouTube, comparando en tiempo real las tres ramas tecnicas:
1. Modelo Base Desarrollado desde Cero (Scratch Naive Bayes)
2. Pipeline de Produccion Scikit-Learn (TF-IDF N-Grams + Regresion Logistica)
3. Motor Semantico Heuristico de Lexicones
"""

import os
import sys
import time
import json
import streamlit as st
import pandas as pd

# Anadir la ruta raiz al sys.path para importacion modular
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.text_cleaner import TextCleaner
from src.models.production_pipeline import ProductionPipeline
from src.models.semantic_moderator import SemanticModerator
from src.models.benchmark import run_comprehensive_benchmark
from src.scrapers.social_feed_simulator import SocialFeedSimulator, SAMPLE_STREAMS

st.set_page_config(
    page_title="Social NLP Moderation Engine | Miguel Benitez",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-clean {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-toxic {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-spam {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():
    cleaner = TextCleaner()
    semantic_engine = SemanticModerator()
    model_path = "models/production_pipeline.joblib"
    benchmark_path = "models/benchmark_report.json"
    
    if not os.path.exists(model_path) or not os.path.exists(benchmark_path):
        run_comprehensive_benchmark(benchmark_path)
        
    pipeline = ProductionPipeline.load(model_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_report = json.load(f)
        
    return cleaner, semantic_engine, pipeline, benchmark_report


cleaner, semantic_engine, pipeline, benchmark_report = load_components()

# Barra lateral informativa
with st.sidebar:
    st.title("🛡️ NLP Moderation Engine")
    st.caption("v1.0.0 • Autor: Ing. Miguel Antonio Benitez Gonzalez")
    st.markdown("---")
    st.markdown("""
    **Arquitectura Dual de Moderacion:**
    - 📐 **Modelo desde Cero:** TF-IDF & Naive Bayes implementado matematicamente con NumPy.
    - ⚡ **Pipeline de Produccion:** Scikit-Learn N-Grams + Regresion Logistica calibrada.
    - 🧠 **Motor Semantico:** Reglas heuristicas y lexicones ponderados de odio/spam.
    """)
    st.markdown("---")
    st.markdown("**Plataformas Soportadas:**")
    st.markdown("• 🐦 Twitter / X\n• 📷 Instagram\n• 👥 Facebook\n• 📺 YouTube")
    st.markdown("---")
    st.markdown("[Repositorio GitHub](https://github.com/miguelbenitez09)")

# Pestañas principales
tab_sandbox, tab_stream, tab_benchmark, tab_about = st.tabs([
    "🧪 Sandbox de Moderacion",
    "📡 Streaming Multi-Plataforma",
    "📊 Benchmarking & Metricas",
    "📖 Metodologia & Arquitectura"
])

# --- Pestaña 1: Sandbox de Moderacion ---
with tab_sandbox:
    st.subheader("Evaluacion en Tiempo Real de Comentarios")
    
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        selected_platform = st.selectbox(
            "Seleccionar Plataforma de Red Social:",
            ["Twitter / X", "Instagram", "Facebook", "YouTube"]
        )
        platform_key = selected_platform.split()[0].lower()
        
        sample_options = {
            "Twitter / X": [
                "Excelente analisis sobre la situacion economica actual, muy certero el punto de vista 👏",
                "Eres un idiota inservible, ojala te despidan pedazo de basura @usuario 🤬",
                "GANA $5000 DOLARES AL DIA DESDE CASA SIN HACER NADA HAZ CLIC EN EL LINK YA http://bit.ly/scam99"
            ],
            "Instagram": [
                "Que foto tan hermosa! Me encanta la iluminacion y los colores 😍❤️",
                "Estas horrible das asco deberias operarte la cara payasa 🤡🤮",
                "Invierte 50 dolares y retira 1000 en 24 horas escribeme al DM garantizado 100%"
            ],
            "Facebook": [
                "Buen dia vecinos, informo que ya se restablecio el servicio de agua en el sector",
                "Viejos ignorantes comunistas de mierda deberian matarlos a todos",
                "OFERTA DE EMPLEO URGENTE EMPACANDO DESDE SU CASA 500 SEMANALES CONTACTARSE AL INBOX"
            ],
            "YouTube": [
                "El mejor tutorial de Python que he visto, directo al grano y muy bien explicado, gracias!",
                "Que video mas estupido y mediocre, el autor tiene retraso mental severo",
                "TRADING BOT AUTOMATICO 99% ACIERTOS TELEGRAM @TRADER_PRO_SCAM"
            ]
        }
        
        preset_choice = st.selectbox("Cargar Ejemplo Predefinido:", ["(Escribir texto personalizado...)"] + sample_options[selected_platform])
        
        default_text = "" if preset_choice.startswith("(") else preset_choice
        user_text = st.text_area("Contenido del Mensaje o Comentario:", value=default_text, height=120)

    with col_meta:
        st.markdown("**Analisis Estructural y Senales:**")
        if user_text:
            meta = cleaner.extract_metadata(user_text)
            st.metric("Longitud (Caracteres)", meta["length"])
            st.metric("Proporcion de Mayusculas", f"{meta['caps_ratio']*100:.1f}%")
            st.write(f"- **Tiene Enlaces:** {'Si ⚠️' if meta['has_urls'] else 'No'}")
            st.write(f"- **Tiene Menciones:** {'Si' if meta['has_mentions'] else 'No'}")
            st.write(f"- **Signos de Exclamacion:** {meta['exclamation_count']}")

    if st.button("🔍 Evaluar Moderacion y Clasificar", type="primary"):
        if not user_text.strip():
            st.warning("Por favor ingrese un texto para analizar.")
        else:
            t0 = time.perf_counter()
            clean_text = cleaner.clean(user_text)
            
            # Inferencias
            semantic_res = semantic_engine.analyze(user_text)
            probs = pipeline.predict_proba([clean_text])[0]
            ml_prob_harmful = float(probs[1])
            latency_ms = (time.perf_counter() - t0) * 1000.0

            st.markdown("---")
            st.subheader("Resultados de Moderacion")
            
            # Decision General
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.markdown("#### Veredicto Consensuado")
                if semantic_res["is_harmful"] and semantic_res["toxicity_score"] >= 0.8:
                    st.markdown(f"<span class='badge-toxic'>{semantic_res['verdict']}</span>", unsafe_allow_html=True)
                elif ml_prob_harmful >= 0.45 or semantic_res["is_harmful"]:
                    st.markdown("<span class='badge-toxic'>OFENSIVO / RIESGOSO</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-clean'>LIMPIO / APROBADO</span>", unsafe_allow_html=True)
                    
            with col_v2:
                st.markdown("#### Probabilidad ML de Riesgo")
                st.progress(ml_prob_harmful)
                st.caption(f"{ml_prob_harmful*100:.1f}% confianza del clasificador de produccion")
                
            with col_v3:
                st.markdown("#### Latencia de Inferencia")
                st.metric("Tiempo de Respuesta", f"{latency_ms:.2f} ms")

            # Comparativa de Motores
            st.markdown("---")
            st.markdown("#### Comparativa Tecnica entre Motores")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("**Pipeline de Produccion (N-Grams + ML):**")
                st.write(f"- Probabilidad Contenido Limpio: **{probs[0]*100:.1f}%**")
                st.write(f"- Probabilidad Contenido Danino: **{probs[1]*100:.1f}%**")
                st.write(f"- Texto Normalizado: *`{clean_text}`*")
                
            with col_m2:
                st.markdown("**Motor Semantico Heuristico:**")
                st.write(f"- Categoria Detectada: **{semantic_res['category'].upper()}**")
                st.write(f"- Puntaje de Severidad: **{semantic_res['toxicity_score']*100:.1f}%**")
                if semantic_res["matched_triggers"]:
                    st.write(f"- Disparadores: {semantic_res['matched_triggers']}")

# --- Pestaña 2: Streaming Multi-Plataforma ---
with tab_stream:
    st.subheader("Simulador de Flujo de Comentarios en Vivo")
    st.caption("Demuestra como un moderador automatizado procesa comentarios en streaming respetando politicas de filtrado.")
    
    stream_platform = st.selectbox("Canal de Ingestion:", ["twitter", "instagram", "facebook", "youtube"], key="stream_plat")
    stream_limit = st.slider("Cantidad de Mensajes a Ingerir:", 3, 10, 5)
    
    if st.button("📡 Iniciar Ingestion de Flujo"):
        simulator = SocialFeedSimulator(platform=stream_platform)
        stream_items = simulator.fetch_stream(limit=stream_limit)
        
        for item in stream_items:
            clean_t = cleaner.clean(item["text"])
            sem_res = semantic_engine.analyze(item["text"])
            prob = pipeline.predict_proba([clean_t])[0][1]
            is_harmful = prob >= 0.45 or sem_res["is_harmful"]
            
            with st.container():
                c1, c2, c3 = st.columns([1, 4, 1.5])
                with c1:
                    st.write(f"**{item['user']}**")
                with c2:
                    st.write(item["text"])
                with c3:
                    if is_harmful:
                        st.markdown("<span class='badge-toxic'>BLOQUEADO</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-clean'>APROBADO</span>", unsafe_allow_html=True)
                st.divider()

# --- Pestaña 3: Benchmarking & Metricas ---
with tab_benchmark:
    st.subheader("Reporte Empirico de Benchmarking")
    st.caption("Comparacion formal entre el modelo desarrollado desde fundamentos matematicos (Scratch) vs el pipeline optimizado de produccion.")
    
    results = benchmark_report["results"]
    
    bench_data = [
        {
            "Enfoque": "1. Scratch Naive Bayes (Desde Cero)",
            "Exactitud (Accuracy)": f"{results['scratch_naive_bayes']['accuracy']*100:.2f}%",
            "F1-Score": f"{results['scratch_naive_bayes']['f1_score']*100:.2f}%",
            "Latencia / Muestra": f"{results['scratch_naive_bayes']['infer_latency_per_sample_ms']:.4f} ms",
            "Arquitectura": "NumPy Puro (Matriz TF-IDF + Laplace Smoothing)"
        },
        {
            "Enfoque": "2. Pipeline Scikit-Learn (Produccion)",
            "Exactitud (Accuracy)": f"{results['production_pipeline']['accuracy']*100:.2f}%",
            "F1-Score": f"{results['production_pipeline']['f1_score']*100:.2f}%",
            "Latencia / Muestra": f"{results['production_pipeline']['infer_latency_per_sample_ms']:.4f} ms",
            "Arquitectura": "TF-IDF N-Grams (1,2) + Regresion Logistica Regularizada"
        },
        {
            "Enfoque": "3. Motor Semantico Heuristico",
            "Exactitud (Accuracy)": f"{results['semantic_moderator']['accuracy']*100:.2f}%",
            "F1-Score": f"{results['semantic_moderator']['f1_score']*100:.2f}%",
            "Latencia / Muestra": f"{results['semantic_moderator']['infer_latency_per_sample_ms']:.4f} ms",
            "Arquitectura": "Lexicones Ponderados + Expresiones Regulares de Phishing/Spam"
        }
    ]
    
    st.table(pd.DataFrame(bench_data))
    
    st.info("""
    **Conclusion del Benchmark:**  
    El pipeline de produccion logra el mayor balance entre exactitud y generalizacion semantica gracias al enriquecimiento de n-gramas. Sin embargo, el modelo desarrollado desde cero con NumPy puro demuestra una latencia sub-milisegundo sobresaliente sin dependencias de alto nivel, ideal para microservicios de ultra-baja latencia.
    """)

# --- Pestaña 4: Metodologia ---
with tab_about:
    st.subheader("Metodologia de Ingenieria de Software y NLP")
    st.markdown("""
    Este proyecto fue diseñado e implementado por **Ing. Miguel Antonio Benitez Gonzalez** (Licenciatura en Ingenieria en Sistemas y Computacion, Universidad Tecnologica de Panama - UTP).
    
    ### Principios de Diseno:
    1. **Matematica Transparente:** La implementacion desde cero demuestra el entendimiento profundo de la probabilidad condicional bayesiana y el algebra lineal requerida en el procesamiento de lenguaje natural.
    2. **Resiliencia Operativa:** Arquitectura modular con desacoplamiento claro entre la capa de preprocesamiento, inferencia matematica, microservicio REST y aplicacion web.
    3. **Politica Educativa y Etica:** Los simuladores y extractores respetan limites de tasa de peticiones y politicas de privacidad de datos, protegiendo las plataformas fuente y simulando comportamiento humano genuino.
    """)
