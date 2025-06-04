import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import os
import time
import re
import base64
import tempfile
import random
import itertools
import gc
import networkx as nx
import squarify
from collections import Counter, defaultdict
from wordcloud import WordCloud
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis

# --- Language Toggle ---
st.set_page_config(page_title="IBA DCX Tool", layout="wide")
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"], key="lang")

# --- Translations Dictionary ---
TRANSLATIONS = {
    "How to Use": {"English": "How to Use", "Español": "Cómo usar"},
    "Photos & Reviews": {"English": "Photos & Reviews", "Español": "Fotos y Reseñas"},
    "Word Cloud": {"English": "Word Cloud", "Español": "Nube de Palabras"},
    "Treemap": {"English": "Treemap", "Español": "Mapa de árbol"},
    "Network Analysis": {"English": "Network Analysis", "Español": "Análisis de Red"},
    "Topic Modeling": {"English": "Topic Modeling", "Español": "Modelado de Temas"},
    "Customer Satisfaction Analysis": {
        "English": "Customer Satisfaction Analysis", 
        "Español": "Análisis de Satisfacción del Cliente"
    },
    "Select Region and Store": {
        "English": "Select Region and Store",
        "Español": "Selecciona Región y Negocio"
    },
    "Please select a region": {
        "English": "Please select a region",
        "Español": "Selecciona una región"
    },
    "Please select a store": {
        "English": "Please select a store",
        "Español": "Selecciona un negocio"
    },
    "Region/Store Selection Finalized": {
        "English": "Region/Store Selection Finalized",
        "Español": "Región/Negocio seleccionado"
    },
    "Please select a feature": {
        "English": "Please select a feature",
        "Español": "Selecciona una función"
    },
    "⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.": {
        "English": "⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.",
        "Español": "⚠️ Por favor selecciona primero la región y el negocio, luego presiona 'Confirmar' para activar las funciones."
    },
    # Add more translations as needed...
}

def T(key):
    """Returns the translation for the current language."""
    return TRANSLATIONS.get(key, {}).get(lang, key)

# --- Styles ---
st.markdown("""
<style>
body, .stApp { background-color: white !important; color: black !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], .css-1d391kg, .css-1v0mbdj {
    background-color: white !important;
    color: black !important;
}
.markdown-text-container { color: black !important; }
label[for^=""] { color: black !important; font-weight: 600; }
div[data-testid="stMarkdownContainer"] p { color: black !important; }
</style>
""", unsafe_allow_html=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

FONT_PATH = "./NanumGothic-Regular.ttf"
font_prop = fm.FontProperties(fname=FONT_PATH)
fm.fontManager.addfont(FONT_PATH)
font_name = font_prop.get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# Dataset mapping
KEYWORD_COLUMNS_KO = ['맛', '서비스', '가격', '위치', '분위기', '위생']
KEYWORD_COLUMNS_EN = ['Taste', 'Service', 'Price', 'Location', 'Atmosphere', 'Hygiene']
KEYWORD_ENGLISH_MAP = dict(zip(KEYWORD_COLUMNS_KO, KEYWORD_COLUMNS_EN))
DATASET_MAP = {
    'Pusan National University': 'IBA-DCX_Analytics_2.0_PNU.csv',
    'Kyung Hee University': 'IBA-DCX_Analytics_2.0_KHU.csv',
    'Jeju Island': 'IBA-DCX_Analytics_2.0_Jeju.csv'
}
LOCATION_ENGLISH_MAP = {
    'Pusan National University': 'Pusan National University',
    'Kyung Hee University': 'Kyung Hee University',
    'Jeju Island': 'Jeju Island'
}

@st.cache_resource
def get_classifier():
    return pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

@st.cache_data
def load_dataset(dataset_name: str) -> pd.DataFrame:
    import gdown
    file_ids = {
        'IBA-DCX_Analytics_2.0_PNU.csv': '1jfMMwnXi5zUOGE6F34B-KjQvfH5jjKmu',
        'IBA-DCX_Analytics_2.0_KHU.csv': '1pqbNRLg8SdsmnZgi9JnqkxjDp7VUPlb4',
        'IBA-DCX_Analytics_2.0_Jeju.csv': '1OeB_VE4bWYCLFAI85ozT7DwiL8V1W7yR'
    }
    file_id = file_ids.get(dataset_name)
    output = f".cache_{dataset_name}"
    if not os.path.exists(output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=True)
    use_cols = ['Name', 'Content', 'Tokens', 'Image_Links'] + KEYWORD_COLUMNS_KO + ['review_sentences', 'Date']
    df = pd.read_csv(output, usecols=use_cols)
    df = df.rename(columns=KEYWORD_ENGLISH_MAP)
    return df

@st.cache_resource
def train_lda_model(corpus, _dictionary, num_topics=10):
    return LdaModel(corpus, num_topics=num_topics, id2word=_dictionary, passes=5)

@st.cache_resource
def get_lda_vis_data(_model, corpus, _dictionary):
    return gensimvis.prepare(_model, corpus, _dictionary)

# -- SIDEBAR (with language support) --
st.sidebar.image("DCX_Tool.png", use_container_width=True)
st.sidebar.title(T("Select Region and Store"))

if 'location_locked' not in st.session_state:
    st.session_state['location_locked'] = False

if not st.session_state['location_locked']:
    location = st.sidebar.selectbox(T("Please select a region"), [''] + list(DATASET_MAP.keys()), key="loc")
    if location:
        df = load_dataset(DATASET_MAP[location])
        stores = df['Name'].value_counts().index.tolist()
        store = st.sidebar.selectbox(T("Please select a store"), [''] + stores, key="store")
        if store and st.sidebar.button(T("Region/Store Selection Finalized")):
            st.session_state.update({
                'location_locked': True,
                'selected_location': location,
                'selected_store': store
            })
else:
    location = st.session_state.get('selected_location')
    store = st.session_state.get('selected_store')
    st.sidebar.markdown(f"🔒 {T('Region')}: {location}\n\n🔒 {T('Store')}: {store}")
    df = load_dataset(DATASET_MAP[location])

# --- Info (English/Spanish) ---
if lang == "Español":
    st.sidebar.markdown("""
    ## **Esta herramienta DCX solo está permitida para:**
    * Usos educativos en universidades o para investigación
    * Dueños de pequeños negocios para sus propios negocios
    * Actividades sin fines de lucro lideradas por estudiantes universitarios

    <span style="color:red; font-weight:bold">
    Cualquier otro uso comercial o reutilización de los datos está estrictamente prohibido.
    </span>
    <br><br>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    ## **This DCX analysis tool is only permitted for use in the following cases:**
    * When used in educational settings such as universities for student education and research
    * When used by small business owners for their own business purposes
    * When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners

    <span style="color:red; font-weight:bold">
    Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.
    </span>
    <br><br>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; font-size:16px; font-weight:bold; margin-bottom:10px;">
📬 Inquiries & Information / Consultas e Información
</div>
<a href="mailto:peter@pusan.ac.kr">
    <button style="
        background-color:#f59f00;
        color:white;
        padding:8px 14px;
        border:none;
        border-radius:5px;
        font-size:14px;
        width:100%;
        margin-bottom:8px;
        cursor:pointer;">
        📧 Contact / Contacto
    </button>
</a>
<a href="https://ibalab.quv.kr/" target="_blank">
    <button style="
        background-color:#1c7ed6;
        color:white;
        padding:8px 14px;
        border:none;
        border-radius:5px;
        font-size:14px;
        width:100%;
        cursor:pointer;">
        🌐 IBA LAB Homepage
    </button>
</a>
""", unsafe_allow_html=True)

# --- TABS (translated) ---
TABS = [
    T("How to Use"),
    T("Photos & Reviews"),
    T("Word Cloud"),
    T("Treemap"),
    T("Network Analysis"),
    T("Topic Modeling"),
    T("Customer Satisfaction Analysis")
]
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = T("How to Use")

if st.session_state.get("location_locked", False):
    selected_tab = st.selectbox("✅ " + T("Please select a feature"), TABS)
    if st.session_state['current_tab'] != selected_tab:
        keys_to_clear = [
            key for key in st.session_state.keys()
            if key not in (
                'selected_location', 
                'selected_store', 
                'location_locked'
            )
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        plt.clf()
        plt.close('all')
        gc.collect()
        st.session_state['current_tab'] = selected_tab
else:
    selected_tab = T("How to Use")
    st.warning(T("⚠️ Please select the region and store first, then press 'Confirm' to activate the functions."))

# ---- ANALYSIS MODULES (Show translated text by lang) ----
def render_usage_tab():
    st.header(T("How to Use"))
    if lang == "Español":
        st.markdown("""
        <div style="background-color: #f5f8fa; padding: 20px; border-radius: 12px; border-left: 6px solid #0d6efd;">
        <p style="font-size:16px;">
        <strong>IBA DCX Tool</strong> es una herramienta que ayuda a establecer estrategias de gestión basadas en la experiencia del cliente mediante el análisis de reseñas online.<br>
        Puedes hacer lo siguiente con esta herramienta:
        </p>
        <ul style="padding-left: 20px; font-size:15px; line-height: 1.6;">
            <li>Generar nubes de palabras</li>
            <li>Crear gráficos treemap</li>
            <li>Análisis de red por frecuencia</li>
            <li>Modelado de temas (LDA)</li>
            <li>Análisis de satisfacción del cliente (sentimiento)</li>
        </ul>
        </div>
        <br>
        <br>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### ✅ ¿Cómo usar?
        <div style="padding: 16px; background-color: #f9f9f9; border-radius: 10px; font-size: 15px; line-height: 1.7;">
            <ol>
                <li>En la <strong>barra lateral</strong>, selecciona una <span style="color:#0d6efd;">región</span> y <span style="color:#0d6efd;">negocio</span>, luego pulsa <strong>'Confirmar'</strong>.</li>
                <li>Selecciona la función que deseas en el <strong>menú desplegable</strong>.</li>
                <li>Para comenzar de nuevo, <strong>recarga la página</strong>.</li>
                <li>El análisis de sentimiento puede tardar más según el número de reseñas.</li>
                <li>Esta herramienta está diseñada para <strong>modo claro</strong>.</li>
            </ol>
            <p style="font-size:14px; color:gray;">
            ⚠️ Si tienes problemas, por favor contacta al email de la barra lateral.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f5f8fa; padding: 20px; border-radius: 12px; border-left: 6px solid #0d6efd;">
        <p style="font-size:16px;">
        <strong>IBA DCX Tool</strong> is a tool that supports customer experience-based management strategy establishment through online review analysis.<br>
        You can perform the following functions with this tool.
        </p>
        <ul style="padding-left: 20px; font-size:15px; line-height: 1.6;">
            <li>Word Cloud Generation</li>
            <li>Treemap Chart Creation</li>
            <li>Frequency-Based Network Analysis</li>
            <li>LDA Topic Modeling</li>
            <li>Customer Satisfaction Analysis via Sentiment Analysis</li>
        </ul>
        </div>
        <br>
        <br>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### ✅ How to Use
        <div style="padding: 16px; background-color: #f9f9f9; border-radius: 10px; font-size: 15px; line-height: 1.7;">
            <ol>
                <li>In the <strong>sidebar</strong>, select a <span style="color:#0d6efd;">region</span> and <span style="color:#0d6efd;">store name</span>, then click the <strong>‘Confirm’</strong> button.</li>
                <li>Choose the desired analysis function from the <strong>function selection dropdown</strong>.</li>
                <li>To start a new analysis, <strong>refresh the page</strong> and begin again.</li>
                <li><strong>Sentiment analysis</strong> may take longer depending on the number of reviews.</li>
                <li>This tool is designed for <strong>Light Mode</strong>.</li>
            </ol>
            <p style="font-size:14px; color:gray;">
            ⚠️ If you encounter issues, please contact the email address provided in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---- (You can add similar Spanish/English branches for each analysis function, using lang == "Español") ----
# ---- For brevity, keep all analysis tabs unchanged, but wrap all text with T() or lang == "Español" checks ----

# e.g. for rendering tab:
if selected_tab == T("How to Use"):
    render_usage_tab()
elif selected_tab == T("Photos & Reviews"):
    render_review_tab(df, store, lang)
elif selected_tab == T("Word Cloud"):
    render_wordcloud_tab(df, store, lang)
elif selected_tab == T("Treemap"):
    render_treemap_tab(df, store, lang)
elif selected_tab == T("Network Analysis"):
    render_network_tab(df, store, lang)
elif selected_tab == T("Topic Modeling"):
    render_topic_tab(df, store, lang)
elif selected_tab == T("Customer Satisfaction Analysis"):
    classifier = get_classifier()
    render_sentiment_dashboard(df, store, classifier, lang)
