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
import urllib.request
import datetime
import pytz
import uuid
from collections import Counter, defaultdict
from wordcloud import WordCloud
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import altair as alt
import gspread
from random import choice
from google.oauth2.service_account import Credentials



# --- Bilingual UI Setup ---
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"], key="lang")

TRANSLATIONS = {
    "The maximum number of users has been reached. Estimated waiting time: {m}분":
        {"English": "The maximum number of users has been reached. Estimated waiting time: {m} min",
         "Español": "Se alcanzó el número máximo de usuarios. Tiempo estimado de espera: {m} min"},
    "⏰ Your session time has ended. Please reconnect.":
        {"English": "⏰ Your session time has ended. Please reconnect.",
         "Español": "⏰ Tu sesión ha terminado. Por favor vuelve a conectarte."},
    "⏳ Your expiration time: {expiration_str}":
        {"English": "⏳ Your expiration time: {expiration_str}",
         "Español": "⏳ Tu sesión expira: {expiration_str}"},
    "✅ Finish the session":
        {"English": "✅ Finish the session", "Español": "✅ Finalizar sesión"},
    "✅ The session has ended.":
        {"English": "✅ The session has ended.", "Español": "✅ La sesión ha terminado."},
    "📊 IBA-DCX Tool":
        {"English": "📊 IBA-DCX Tool", "Español": "📊 Herramienta IBA-DCX"},
    "How to Use": {"English": "How to Use", "Español": "Cómo usar"},
     "Review Summary and Images":
        {"English": "Review Summary and Images", "Español": "Resumen de Reseñas y Fotos"},
    "Review Indicators":
        {"English": "Review Indicators", "Español": "Indicadores de Reseñas"},
    "Total number of Reviews":
        {"English": "Total number of Reviews", "Español": "Total de Reseñas"},
    "Total number of Images":
        {"English": "Total number of Images", "Español": "Total de Fotos"},
    "images":
        {"English": "Images", "Español": "Fotos"},
    "reviews":
        {"English": "Reviews", "Español": "Reseñas"},
    "Average Review Length":
        {"English": "Average Review Length", "Español": "Longitud Promedio de Reseña"},
    "Top Reviews 🖼️":
        {"English": "Top Reviews 🖼️", "Español": "Reseñas Destacadas 🖼️"},
    "🔄 Look at other reviews":
        {"English": "🔄 Look at other reviews", "Español": "🔄 Ver otras reseñas"},
    "Wordcloud":
        {"English": "Wordcloud", "Español": "Nube de Palabras"},
    "No text available":
        {"English": "No text available", "Español": "No hay texto disponible"},
    "Treemap":
        {"English": "Treemap", "Español": "Treemap"},
    "No text available for {column}":
        {"English": "No text available for {column}", "Español": "No hay texto disponible para {column}"},
    "Color Descriptions":
        {"English": "Color Descriptions", "Español": "Descripción de los Colores"},
    "Network Analysis": {"English": "Network Analysis", "Español": "Análisis de Red"},
    "Insufficient reviews to perform network analysis.":
        {"English": "Insufficient reviews to perform network analysis.", "Español": "No hay suficientes reseñas para realizar el análisis de red."},
    "Setting the Word Filter Criteria":
        {"English": "Setting the Word Filter Criteria", "Español": "Configurar el filtro de palabras"},
    "Minimum word frequency":
        {"English": "Minimum word frequency", "Español": "Frecuencia mínima de palabra"},
    "No matching network found with current filter criteria.":
        {"English": "No matching network found with current filter criteria.", "Español": "No se encontró una red con los criterios de filtro actuales."},
    "Color Criteria":
        {"English": "Color Criteria", "Español": "Criterios de color"},
    "High Frequency words 30%":
        {"English": "High Frequency words 30%", "Español": "Palabras de alta frecuencia 30%"},
    "Low Frequency words 30%":
        {"English": "Low Frequency words 30%", "Español": "Palabras de baja frecuencia 30%"},
    "Medium Frequency words":
        {"English": "Medium Frequency words", "Español": "Palabras de frecuencia media"},
    "Topic Modeling": {"English": "Topic Modeling", "Español": "Modelado de Temas"},
    "Not enough reviews to run topic modeling.":
        {"English": "Not enough reviews to run topic modeling.", "Español": "No hay suficientes reseñas para ejecutar el modelado de temas."},
    "Execute Topic Modeling":
        {"English": "Execute Topic Modeling", "Español": "Ejecutar Modelado de Temas"},
    "Download LDA Result HTML":
        {"English": "📁 Download LDA Result HTML", "Español": "📁 Descargar HTML de resultados LDA"},
    "Customer Satisfaction Analysis":
        {"English": "Customer Satisfaction Analysis", "Español": "Análisis de Satisfacción del Cliente"},
    "Insufficient reviews to perform sentiment analysis.":
        {"English": "Insufficient reviews to perform sentiment analysis.", "Español": "No hay suficientes reseñas para realizar el análisis de sentimiento."},
    "🧠 Start Customer Satisfaction Analysis":
        {"English": "🧠 Start Customer Satisfaction Analysis", "Español": "🧠 Iniciar análisis de satisfacción del cliente"},
    "Click the button above to start the analysis.":
        {"English": "Click the button above to start the analysis.", "Español": "Haz clic en el botón de arriba para iniciar el análisis."},
    "🔎 Overall Sentiment Score Comparison":
        {"English": "🔎 Overall Sentiment Score Comparison", "Español": "🔎 Comparación General de Sentimiento"},
    "Current Store":
        {"English": "Current Store", "Español": "Negocio Actual"},
    "Average":
        {"English": "Average", "Español": "Promedio"},
    "points difference":
        {"English": "points difference", "Español": "puntos de diferencia"},
    "Keyword Sentiment Score Comparison":
        {"English": "Keyword Sentiment Score Comparison", "Español": "Comparación de Sentimiento por Palabra Clave"},
    "Insufficient reviews for analysis":
        {"English": "Insufficient reviews for analysis", "Español": "No hay suficientes reseñas para el análisis"},
    "Points":
        {"English": "Points", "Español": "Puntos"},
    "Regional Average":
        {"English": "Regional Average", "Español": "Promedio Regional"},
        "Select Region and Store":
        {"English": "Select Region and Store", "Español": "Selecciona Región y Negocio"},
    "Please select a region":
        {"English": "Please select a region", "Español": "Selecciona una región"},
    "Please select a store":
        {"English": "Please select a store", "Español": "Selecciona un negocio"},
    "✅Region/Store Selected":
        {"English": "✅Region/Store Selected", "Español": "✅Región/Negocio Seleccionado"},
    "Region":
        {"English": "Region", "Español": "Región"},
    "Store":
        {"English": "Store", "Español": "Negocio"},
    "This DCX analysis tool is only permitted for use in the following cases:":
        {"English": "This DCX analysis tool is only permitted for use in the following cases:",
         "Español": "Esta herramienta de análisis DCX solo se permite usar en los siguientes casos:"},
    "* When used in educational settings such as universities for student education and research":
        {"English": "* When used in educational settings such as universities for student education and research",
         "Español": "* Cuando se utiliza en entornos educativos como universidades para la educación e investigación estudiantil"},
    "* When used by small business owners for their own business purposes":
        {"English": "* When used by small business owners for their own business purposes",
         "Español": "* Cuando la utilizan pequeños empresarios para sus propios fines comerciales"},
    "* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners":
        {"English": "* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners",
         "Español": "* Cuando estudiantes universitarios o de posgrado la usan como parte de actividades de servicio comunitario sin fines de lucro para proveer estrategias a pequeños negocios locales"},
    "Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.":
        {"English": "Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.",
         "Español": "Excepto en los casos anteriores, cualquier uso comercial de esta herramienta y reutilización de los datos está estrictamente prohibido."},
    "Inquiries & Information":
        {"English": "Inquiries & Information", "Español": "Consultas & Información"},
    "Contact via Email":
        {"English": "Contact via Email", "Español": "Contacto por Email"},
    "IBA LAB Homepage":
        {"English": "IBA LAB Homepage", "Español": "Página IBA LAB"},
    "Photos & Reviews":
        {"English": "Photos & Reviews", "Español": "Fotos y Reseñas"},
    "Word Cloud":
        {"English": "Word Cloud", "Español": "Nube de Palabras"},
    "Network Analysis":
        {"English": "Network Analysis", "Español": "Análisis de Red"},
    "Topic Modeling":
        {"English": "Topic Modeling", "Español": "Modelado de Temas"},
    "Points":
        {"English": "Points", "Español": "Puntos"},
    "Customer Satisfaction Analysis":
        {"English": "Customer Satisfaction Analysis", "Español": "Análisis de Satisfacción del Cliente"},
    "✅ Please select a feature":
        {"English": "✅ Please select a feature", "Español": "✅ Selecciona una función"},
    "✅Region/Store Selection Finalized":
        {"English": "✅ Region/Store has been selected", "Español": "✅ Selección de Región/Negocio Confirmada"},
    "⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.":
        {"English": "⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.",
         "Español": "⚠️ Selecciona primero la región y el negocio y luego pulsa 'Confirmar' para activar las funciones."},
}

    # Add more keys/phrases below as you translate your UI!
def T(key):
    return TRANSLATIONS.get(key, {}).get(lang, key)

region_avg_scores = {
    'Pusan National University': {
        'total': 89.05,
        'Taste': 90.12,
        'Service': 87.86,
        'Price': 87.02,
        'Location': 81.43,
        'Atmosphere': 88.63,
        'Cleanliness': 89.17
    },
    'Kyung Hee University': {
        'total': 88.87,
        'Taste': 91.05,
        'Service': 87.88,
        'Price': 86.01,
        'Location': 78.23,
        'Atmosphere': 85.76,
        'Cleanliness': 89.53
    },
    'Jeju Island': {
        'total': 88.53,
        'Taste': 88.92,
        'Service': 88.00,
        'Price': 81.22,
        'Location': 81.47,
        'Atmosphere': 85.09,
        'Cleanliness': 89.87
    }
}

# Force Light Mode
st.markdown("""
<style>
body, .stApp { background-color: white !important; color: black !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], .css-1d391kg, .css-1v0mbdj {
    background-color: white !important;
    color: black !important;
}
.markdown-text-container { color: black !important; }
</style>
""", unsafe_allow_html=True)

# Global settings
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

TIMEZONE = pytz.timezone('Asia/Seoul')

# Dataset mapping
KEYWORD_COLUMNS_EN = ['Taste', 'Service', 'Price', 'Location', 'Atmosphere', 'Cleanliness']
DATASET_MAP = {
    'Pusan National University': 'IBA-DCX_Analytics_2.0_PNU.csv',
    'Kyung Hee University': 'IBA-DCX_Analytics_2.0_KHU.csv',
    'Jeju Island': 'IBA-DCX_Analytics_2.0_Jeju.csv'
}

# Location English Mapping
LOCATION_ENGLISH_MAP = {
    'Pusan National University': 'Pusan National University',
    'Kyung Hee University': 'Kyung Hee University',
    'Jeju Island': 'Jeju Island'
}

###############################################
# Resource management

@st.cache_resource
def get_classifier():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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
    use_cols = ['Name', 'Content', 'Tokens', 'Image_Links',
            'Taste', 'Service', 'Price', 'Location', 'Atmosphere', 'Cleanliness',
            'review_sentences', 'Date']
    df = pd.read_csv(output, usecols=use_cols)
    return df

@st.cache_resource
def train_lda_model(corpus, _dictionary, num_topics=10):
    return LdaModel(corpus, num_topics=num_topics, id2word=_dictionary, passes=5)

@st.cache_resource
def get_lda_vis_data(_model, corpus, _dictionary):
    return gensimvis.prepare(_model, corpus, _dictionary)

###############################################
# Here used to go the limitations of users
###############################################
# Functions

def compute_sentiment(text, classifier):
    if not isinstance(text, str):
        text = str(text)
    result = classifier(text)
    return result[0]['score'] if result[0]['label'] == 'LABEL_1' else 1 - result[0]['score']

def render_title(location, store):
    st.title(f"{location} - {store}")

def clean_memory(keys):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    plt.clf()
    plt.close('all')
    gc.collect()

def clean_tokens(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove commas, periods, etc.
    return text.split()

# Stopwords definition (unchanged)
stopwords = {
    # Particles / Pronouns / Demonstratives
    '이', '그', '저', '것', '거', '곳', '수', '좀', '처럼', '까지', '에도', '에도요', '이나', '라도',

    # Conjunctions / Connectors
    '그리고', '그래서', '그러나', '하지만', '또한', '즉', '결국', '때문에', '그래도',

    # Predicates / Endings / Auxiliary verbs
    '합니다', '해요', '했어요', '하네요', '하시네요', '하시던데요', '같아요', '있어요', '없어요',
    '되네요', '되었어요', '보여요', '느껴져요', '하겠습니다', '되겠습니다', '있습니다', '없습니다',
    '합니다', '이에요', '이라', '해서',

    # Interjections / Review-specific expressions
    'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', '^^', '^^;;', '~', '~~', '!!!', '??', '!?', '?!', '...', '!!', '~!!', '~^^!!',

    # Emphasis expressions
    '아주', '정말', '진짜', '엄청', '매우', '완전', '너무', '굉장히', '많이', '많아요', '적당히', '넘',

    # Others
    '정도', '느낌', '같은', '니당', '네요', '있네요', '이네요', '이라서',
    '해서요', '보니까', '봤어요', '먹었어요', '마셨어요', '갔어요', '봤습니다', '하는', '하게', '드네', '또시',
    '이랑', '하고', '해도', '해도요', '때문에요', '이나요', '정도에요'
}

###############################################
# Modules

# Usage
def render_usage_tab():
    st.header(T("📊 IBA-DCX Tool"))

    if lang == "Español":
        st.markdown("""
        <div style="background-color: #f5f8fa; padding: 20px; border-radius: 12px; border-left: 6px solid #0d6efd;">
            <p style="font-size:16px;">
            <strong>IBA DCX Tool</strong> es una herramienta que apoya el establecimiento de estrategias de gestión basadas en la experiencia del cliente a través del análisis de reseñas online.<br>
            Puedes realizar las siguientes funciones con esta herramienta.
            </p>
            <ul style="padding-left: 20px; font-size:15px; line-height: 1.6;">
                <li>Generación de Nube de Palabras</li>
                <li>Creación de Gráficos Treemap</li>
                <li>Análisis de Red por Frecuencia</li>
                <li>Modelado de Temas LDA</li>
                <li>Análisis de Satisfacción del Cliente vía Análisis de Sentimiento</li>
            </ul>
        </div>
        <br>
        <br>
        """, unsafe_allow_html=True)

        st.markdown("### ✅ Cómo usar")
        st.markdown("""
        <div style="padding: 16px; background-color: #f9f9f9; border-radius: 10px; font-size: 15px; line-height: 1.7;">
            <ol>
                <li>En la <strong>barra lateral</strong>, selecciona una <span style="color:#0d6efd;">ubicación</span> y <span style="color:#0d6efd;">nombre del negocio</span>, luego haz clic en el botón <strong>‘Confirmar’</strong>.</li>
                <li>Elige la función de análisis deseada del <strong>menú desplegable</strong>.</li>
                <li>Para comenzar un nuevo análisis, <strong>actualiza la página</strong> y comienza de nuevo.</li>
                <li><strong>El análisis de sentimiento</strong> puede demorar más dependiendo del número de reseñas.</li>
                <li>Esta herramienta está diseñada para <strong>modo claro</strong>. Puedes cambiar el tema desde el menú (⋮) en la esquina superior derecha.</li>
            </ol>
            <p style="font-size:14px; color:gray;">
            ⚠️ Si tienes problemas, contacta al correo en la barra lateral.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f5f8fa; padding: 20px; border-radius: 12px; border-left: 6px solid #0d6efd;">
            <p style="font-size:16px;">
            <strong>IBA DCX Tool</strong> is a tool that supports <strong>customer experience-based management strategy establishment</strong> through <strong>online review analysis</strong>.<br>
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

        st.markdown("### ✅ How to Use")

        st.markdown("""
        <div style="padding: 16px; background-color: #f9f9f9; border-radius: 10px; font-size: 15px; line-height: 1.7;">
            <ol>
                <li>In the <strong>sidebar</strong>, select a <span style="color:#0d6efd;">location</span> and <span style="color:#0d6efd;">store name</span>, then click the <strong>‘Confirm’</strong> button.</li>
                <li>Choose the desired analysis function from the <strong>function selection dropdown</strong>.</li>
                <li>To start a new analysis, <strong>refresh the page</strong> and begin again.</li>
                <li><strong>Sentiment analysis</strong> may take longer depending on the number of reviews.</li>
                <li>This tool is designed for <strong>Light Mode</strong>. You can change the theme via the menu (⋮) in the top-right corner.</li>
            </ol>
            <p style="font-size:14px; color:gray;">
            ⚠️ If you encounter issues, please contact the email address provided in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Review loading
def render_review_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: {T('Review Summary and Images')}")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)
    image_links = df_store['Image_Links'].tolist()
    reviews = df_store['Content'].fillna('').astype(str).tolist()
    image_pattern = r'https?://[\S]+\.(?:jpg|jpeg|png|gif)'
    all_links, all_reviews = [], []
    for idx, link_str in enumerate(image_links):
        if isinstance(link_str, str):
            links = re.findall(image_pattern, link_str)
            all_links.extend(links)
            all_reviews.extend([reviews[idx]] * len(links))

    display_reviews = all_reviews

    avg_length = np.mean([len(r) for r in reviews if isinstance(r, str)]) if reviews else 0
    st.markdown(f"### 📊 {T('Review Indicators')}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(T("Total number of Reviews"), f"{len(df_store)} reviews")
    with col2:
        st.metric(T("Total number of Images"), f"{len(all_links)} images")
    with col3:
        st.metric(T("Average Review Length"), f"{avg_length:.1f}")

    st.markdown(f"### {T('Top Reviews 🖼️')}")
    NUM_CARDS = 6
    if 'review_indices' not in st.session_state:
        st.session_state.review_indices = random.sample(range(len(all_links)), min(NUM_CARDS, len(all_links)))
    if st.button(T("🔄 Look at other reviews")):
        st.session_state.review_indices = random.sample(range(len(all_links)), min(NUM_CARDS, len(all_links)))
    for row_start in range(0, len(st.session_state.review_indices), 3):
        row_cols = st.columns(3)
        for i in range(3):
            if row_start + i >= len(st.session_state.review_indices):
                break
            idx = st.session_state.review_indices[row_start + i]
            with row_cols[i]:
                st.markdown(f"""
                <div style="height: 180px; overflow: hidden; border-radius: 8px;">
                    <img src="{all_links[idx]}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" />
                </div>
                """, unsafe_allow_html=True)

                # Show the translated review
                highlighted = display_reviews[idx]
                st.markdown(f"""
                <div style="padding:12px; background-color:#f9f9f9; border-radius:10px;
                            box-shadow:0 2px 4px rgba(0,0,0,0.08); margin-top:8px;
                            height:150px; overflow:auto;">
                    <p style="font-size:14px; color:#333;">{highlighted}</p>
                </div>
                """, unsafe_allow_html=True)

# Wordcloud
# Define vivid color list
VIVID_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e31a1c", "#17becf"]

# Random color function
def vivid_color_func(*args, **kwargs):
    return choice(VIVID_COLORS)

# Wordcloud tab rendering function
def render_wordcloud_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: {T('Wordcloud')}")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    columns_to_plot = ['Content'] + KEYWORD_COLUMNS_EN

    container = st.container()
    cols = container.columns(3)

    for idx, column in enumerate(columns_to_plot):
        col = cols[idx % 3]
        text = ' '.join(df_store[column].dropna().map(str))
        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        filtered_text = ' '.join(filtered_tokens)

        with col:
            st.markdown(
                f"<div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:5px;'>{column}</div>",
                unsafe_allow_html=True
            )

            if filtered_text.strip():
                wordcloud = WordCloud(
                    font_path=FONT_PATH,
                    width=800,
                    height=800,
                    contour_width=1.8,
                    contour_color='black',
                    background_color='white',
                    mode='RGB',
                    color_func=vivid_color_func,
                    collocations=False
                ).generate(filtered_text)

                fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
                ax.imshow(wordcloud, interpolation='nearest')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.markdown(f"""
                <div style="padding:10px; text-align:center; background-color:#f9f9f9;
                            border-radius:10px; min-height:200px; height:260px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="color:gray;">{T('No text available')}</span>
                </div>
                """, unsafe_allow_html=True)

# Treemap
def render_treemap_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: {T('Treemap')}")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    columns_to_plot = ['Content'] + KEYWORD_COLUMNS_EN
    container = st.container()
    cols = container.columns(3)

    for idx, column in enumerate(columns_to_plot):
        col = cols[idx % 3]
        text = ' '.join(df_store[column].dropna().map(str))

        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        word_count = Counter(filtered_tokens)

        with col:
            st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:5px;'>{column}</div>", unsafe_allow_html=True)

            if filtered_tokens and len(word_count) > 0:
                most_common = word_count.most_common(10)
                sizes = [count for _, count in most_common]
                labels = [f"{word} ({count})" for word, count in most_common]

                cmap = plt.cm.get_cmap("Blues")
                normed_sizes = [s / max(sizes) for s in sizes]
                colors = [cmap(0.3 + 0.7 * s) for s in normed_sizes]

                fig, ax = plt.subplots(figsize=(4, 4))
                squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, ax=ax, text_kwargs={'fontsize':10})
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.markdown(f"""
                <div style="padding:20px; text-align:center; background-color:#f9f9f9;
                            border-radius:10px; min-height:260px;
                            display:flex; align-items:center; justify-content:center;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.05);">
                    <span style="color:gray; font-size:16px;">{T('No text available for {column}').format(column=column)}</span>
                </div>
                """, unsafe_allow_html=True)

    with st.expander(f"📘 {T('Color Descriptions')}"):
        if lang == "Español":
            st.markdown("""
            - El **color del treemap representa la frecuencia relativa** de la palabra.
            - **Azul oscuro** indica una palabra mencionada más frecuentemente.
            - **Colores claros** representan palabras con menor frecuencia.
            """)
        else:
            st.markdown("""
            - The **color of the treemap represents the relative frequency** of the word.  
            - **Dark blue** indicates a more frequently mentioned word.  
            - **Light colors** represent words with lower frequency.
            """)

# Network analysis
def render_network_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: {T('Network Analysis')}")
    df_store = df[df['Name'] == store]

    if len(df_store) < 20:
        st.warning(T("Insufficient reviews to perform network analysis."))
        return

    def clean_tokens(text):
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()
    
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    st.subheader(T("Setting the Word Filter Criteria"))
    total_reviews = len(df_store)
    min_value = max(1, total_reviews // 20)
    max_value = max(2, total_reviews // 10)
    default_value = (min_value + max_value) // 2

    min_freq = st.slider(
        T("Minimum word frequency"),
        min_value=min_value,
        max_value=max_value,
        value=default_value
    )

    word_freq = Counter(itertools.chain(*df_store['Tokens']))
    filtered_words = {w for w, c in word_freq.items() if c >= min_freq}

    df_store['Filtered_Tokens'] = df_store['Tokens'].apply(
        lambda tokens: [w for w in tokens if w in filtered_words and w not in stopwords and len(w) > 1]
    )

    co_occurrence = defaultdict(int)
    for tokens in df_store['Filtered_Tokens']:
        for pair in itertools.combinations(set(tokens), 2):
            co_occurrence[tuple(sorted(pair))] += 1

    G = nx.Graph()
    for (w1, w2), freq in co_occurrence.items():
        G.add_edge(w1, w2, weight=freq)

    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        st.warning(T("No matching network found with current filter criteria."))
        return

    pos = nx.spring_layout(G, k=0.5, seed=42)
    degree_centrality = nx.degree_centrality(G)

    freq_dict = {node: word_freq.get(node, 0) for node in G.nodes()}
    freq_values = list(freq_dict.values())
    upper_thresh = np.percentile(freq_values, 70)
    lower_thresh = np.percentile(freq_values, 30)

    def get_color(freq):
        if freq >= upper_thresh:
            return 'green'
        elif freq <= lower_thresh:
            return 'crimson'
        else:
            return 'skyblue'

    node_colors = [get_color(freq_dict[n]) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(top=0.88, bottom=0.15)
    node_sizes = [1000 + len(n) * 250 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family=font_prop.get_name(), ax=ax)
    
    ax.set_title(f"{store} - {T('Network Analysis')}", fontproperties=font_prop, fontsize=16, pad=12)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    with st.expander(f"🌈 {T('Color Criteria')}"):
        if lang == "Español":
            st.markdown(f"""
            - 🟢 **Verde**: {T("High Frequency words 30%")}
            - 🔴 **Rojo**: {T("Low Frequency words 30%")}
            - 🔵 **Azul**: {T("Medium Frequency words")}
            """)
        else:
            st.markdown(f"""
            - 🟢 **Green**: {T("High Frequency words 30%")}  
            - 🔴 **Red**: {T("Low Frequency words 30%")}  
            - 🔵 **Blue**: {T("Medium Frequency words")}
            """)

# Topic modeling
def render_topic_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: {T('Topic Modeling')}")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)
    if len(df_store) < 50:
        st.warning(T("Not enough reviews to run topic modeling."))
        return

    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(str.split)
    if len(df_store) > 300:
        df_store = df_store.sample(300, random_state=42)

    dictionary = corpora.Dictionary(df_store['Tokens'])
    corpus = [dictionary.doc2bow(text) for text in df_store['Tokens']]

    if st.button(T("Execute Topic Modeling")):
        lda_model = train_lda_model(corpus, dictionary)
        vis_data = get_lda_vis_data(lda_model, corpus, dictionary)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".html") as f:
            pyLDAvis.save_html(vis_data, f.name)
            html_path = f.name
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        b64 = base64.b64encode(html_content.encode()).decode()
        st.markdown(f'<a href="data:text/html;base64,{b64}" download="lda_result.html">{T("Download LDA Result HTML")}</a>', unsafe_allow_html=True)
        del lda_model, vis_data, corpus, dictionary
        gc.collect()

# Sentiment analysis
def render_sentiment_dashboard(df, store, classifier):
    region_avg_scores = {
        'Pusan National University': {
            'total': 89.05,
            'Taste': 90.12,
            'Service': 87.86,
            'Price': 87.02,
            'Location': 81.43,
            'Atmosphere': 88.63,
            'Hygiene': 89.17
        },
        'Kyung Hee University': {
            'total': 88.87,
            'Taste': 91.05,
            'Service': 87.88,
            'Price': 86.01,
            'Location': 78.23,
            'Atmosphere': 85.76,
            'Hygiene': 89.53
        },
        'Jeju Island': {
            'total': 88.53,
            'Taste': 88.92,
            'Service': 88.00,
            'Price': 81.22,
            'Location': 81.47,
            'Atmosphere': 85.09,
            'Hygiene': 89.87
        }
    }
    st.header(f"{LOCATION_ENGLISH_MAP.get(st.session_state.get('selected_location', ''))} - {store}: {T('Customer Satisfaction Analysis')}")
    df_store = df[df['Name'] == store]

    if len(df_store) < 50:
        st.warning(T("Insufficient reviews to perform sentiment analysis."))
        return

    sentiment_key = f"sentiment_scores_{store}"

    if sentiment_key not in st.session_state:
        if st.button(T("🧠 Start Customer Satisfaction Analysis")):
            texts = df_store['review_sentences'].dropna().astype(str).tolist()
            keyword_inputs = {col: df_store[col].dropna().astype(str).tolist() for col in KEYWORD_COLUMNS_EN}
            total_steps = len(texts) + sum(len(v) for v in keyword_inputs.values())
            completed_steps = 0
            progress_bar = st.progress(0)

            total_scores = []
            for text in texts:
                result = classifier(text)[0]
                score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                total_scores.append(score)
                completed_steps += 1
                progress_bar.progress(completed_steps / total_steps)

            keyword_scores = {}
            for col, col_texts in keyword_inputs.items():
                if col_texts:
                    scores = []
                    for text in col_texts:
                        result = classifier(text)[0]
                        score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                        scores.append(score)
                        completed_steps += 1
                        progress_bar.progress(completed_steps / total_steps)
                    keyword_scores[col] = np.mean(scores) * 100
                else:
                    keyword_scores[col] = None

            st.session_state[sentiment_key] = {
                'total': np.mean(total_scores) * 100,
                'keywords': keyword_scores
            }
        else:
            st.info(T("Click the button above to start the analysis."))
            return

    # Visualize results
    region_name = st.session_state.get('selected_location', '')
    region_stats = region_avg_scores.get(region_name, {})
    sentiment_data = st.session_state[sentiment_key]

    # Overall score comparison
    st.subheader(T("🔎 Overall Sentiment Score Comparison"))

    store_total = sentiment_data['total']
    region_total = region_stats.get('total', None)

    if region_total is not None:
        diff = store_total - region_total
        trend_icon = "▲" if diff > 0 else ("▼" if diff < 0 else "▶")
        trend_color = "green" if diff > 0 else ("crimson" if diff < 0 else "gray")
        trend_text = f"{trend_icon} {abs(diff):.2f} {T('points difference')}"
    else:
        trend_text = "-"
        trend_color = "gray"

    col1, col2 = st.columns(2)

    box_style_total = """
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f5f5;
        text-align: center;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
"""

    with col1:
        st.markdown(f"""
        <div style="{box_style_total}">
            <div style="font-size:18px; font-weight:bold;">{T("Current Store")}</div>
            <div style="font-size:36px; font-weight:bold; color:#2b8a3e;">{store_total:.2f} Points</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{box_style_total}">
            <div style="font-size:18px; font-weight:bold;">{region_name} {T("Average")}</div>
            <div style="font-size:36px; font-weight:bold; color:#1c7ed6;">{region_total:.2f} Points</div>
            <div style="font-size:16px; color:{trend_color}; margin-top:5px;">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader(f"🔎 {T('Keyword Sentiment Score Comparison')}")
    keyword_data = sentiment_data["keywords"]
    cols = st.columns(3)

    for idx, keyword in enumerate(KEYWORD_COLUMNS_EN):
        with cols[idx % 3]:
            store_score = keyword_data.get(keyword)
            region_score = region_stats.get(keyword)

            box_style = """
                padding: 15px;
                border-radius: 10px;
                background-color: whitesmoke;
                text-align: center;
                box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
                min-height: 130px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            """

            if store_score is None:
                st.markdown(f"""
                    <div style="{box_style}">
                        <div style="font-size:18px; font-weight:bold">{keyword}</div>
                        <div style="font-size:16px; color:gray; margin-top:12px;">{T('Insufficient reviews for analysis')}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                diff = store_score - region_score if region_score else 0
                trend = "▲" if diff > 0 else ("▼" if diff < 0 else "-")
                color = "green" if diff > 0 else ("crimson" if diff < 0 else "gray")

                st.markdown(f"""
                    <div style="{box_style}">
                        <div style="font-size:18px; font-weight:bold">{keyword}</div>
                        <div style="font-size:28px; color:{color}">{store_score:.2f}{T(' Points')} {trend}</div>
                        <div style="font-size:14px; color:gray">{T('Regional Average')}: {region_score:.2f}{T(' Points')}</div>
                    </div>
                """, unsafe_allow_html=True)
       
###############################################
# UI

# Sidebar
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
        if store and st.sidebar.button(T("✅Region/Store Selection Finalized")):
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

# Usage rules (bilingual markdown)
if lang == "Español":
    st.sidebar.markdown(f"""
    ## **{T('This DCX analysis tool is only permitted for use in the following cases:')}**
    {T('* When used in educational settings such as universities for student education and research')}
    {T('* When used by small business owners for their own business purposes')}
    {T('* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners')}
    <span style="color:red; font-weight:bold">
    {T('Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.')}
    </span>
    <br><br><br>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    ## **{T('This DCX analysis tool is only permitted for use in the following cases:')}**
    {T('* When used in educational settings such as universities for student education and research')}
    {T('* When used by small business owners for their own business purposes')}
    {T('* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners')}
    <span style="color:red; font-weight:bold">
    {T('Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.')}
    </span>
    <br><br><br>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="text-align:center; font-size:16px; font-weight:bold; margin-bottom:10px;">
📬 {T("Inquiries & Information")}
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
        📧 {T("Contact via Email")}
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
        🌐 {T("IBA LAB Homepage")}
    </button>
</a>
""", unsafe_allow_html=True)

# Tab setup (all bilingual)
TABS = [
    T("How to Use"),
    T("Photos & Reviews"),
    T("WordCloud"),
    T("Treemap"),
    T("Network Analysis"),
    T("Topic Modeling"),
    T("Customer Satisfaction Analysis")
]

if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = T("How to Use")

# Force color for selectbox label and warning text
st.markdown("""
<style>
label[for^=""] {
    color: black !important;
    font-weight: 600;
}
div[data-testid="stMarkdownContainer"] p {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("location_locked", False):
    selected_tab = st.selectbox(T("✅ Please select a feature"), TABS)
    if st.session_state['current_tab'] != selected_tab:
        keys_to_clear = [
            key for key in st.session_state.keys()
            if key not in (
                'selected_location', 
                'selected_store', 
                'location_locked', 
                'user_id', 
                'queue_checked', 
                'start_time'
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

# Execute tab-specific functions (mapping bilingual tab names to functions)
tab_map = {
    T("How to Use"): render_usage_tab,
    T("Photos & Reviews"): render_review_tab,
    T("Word Cloud"): render_wordcloud_tab,
    T("Treemap"): render_treemap_tab,
    T("Network Analysis"): render_network_tab,
    T("Topic Modeling"): render_topic_tab,
    T("Customer Satisfaction Analysis"): lambda: render_sentiment_dashboard(df, store, get_classifier()),
}

if selected_tab in tab_map:
    # Photos & Reviews, Word Cloud, Treemap, Network Analysis, Topic Modeling take df, store
    if selected_tab in [T("Photos & Reviews"), T("Word Cloud"), T("Treemap"), T("Network Analysis"), T("Topic Modeling")]:
        tab_map[selected_tab](df, store)
    elif selected_tab == T("How to Use"):
        tab_map[selected_tab]()
    else:  # Sentiment
        tab_map[selected_tab]()

