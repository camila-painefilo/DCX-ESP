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
from collections import Counter, defaultdict
from wordcloud import WordCloud
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis

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

# Resource management
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
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

stopwords = {
    '이', '그', '저', '것', '거', '곳', '수', '좀', '처럼', '까지', '에도', '에도요', '이나', '라도',
    '그리고', '그래서', '그러나', '하지만', '또한', '즉', '결국', '때문에', '그래도',
    '합니다', '해요', '했어요', '하네요', '하시네요', '하시던데요', '같아요', '있어요', '없어요',
    '되네요', '되었어요', '보여요', '느껴져요', '하겠습니다', '되겠습니다', '있습니다', '없습니다',
    '합니다', '이에요', '이라', '해서',
    'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', '^^', '^^;;', '~', '~~', '!!!', '??', '!?', '?!', '...', '!!', '~!!', '~^^!!',
    '아주', '정말', '진짜', '엄청', '매우', '완전', '너무', '굉장히', '많이', '많아요', '적당히', '넘',
    '정도', '느낌', '같은', '니당', '네요', '있네요', '이네요', '이라서',
    '해서요', '보니까', '봤어요', '먹었어요', '마셨어요', '갔어요', '봤습니다', '하는', '하게', '드네', '또시',
    '이랑', '하고', '해도', '해도요', '때문에요', '이나요', '정도에요'
}

# --- UI Modules (no change to logic below here!) ---

def render_usage_tab():
    st.header("📊 IBA-DCX Tool")
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

# ... All render_* functions unchanged (just as in your original) ...

# -- Omitted for brevity in this message, but keep all render_review_tab, render_wordcloud_tab, render_treemap_tab, render_network_tab, render_topic_tab, render_sentiment_dashboard --

# -- Sidebar --
st.sidebar.image("DCX_Tool.png", use_container_width=True)
st.sidebar.title("Select Region and Store")

if 'location_locked' not in st.session_state:
    st.session_state['location_locked'] = False

if not st.session_state['location_locked']:
    location = st.sidebar.selectbox("Please select a region", [''] + list(DATASET_MAP.keys()), key="loc")
    if location:
        df = load_dataset(DATASET_MAP[location])
        stores = df['Name'].value_counts().index.tolist()
        store = st.sidebar.selectbox("Please select a store", [''] + stores, key="store")
        if store and st.sidebar.button("✅Region/Store Selection Finalized"):
            st.session_state.update({
                'location_locked': True,
                'selected_location': location,
                'selected_store': store
            })
else:
    location = st.session_state.get('selected_location')
    store = st.session_state.get('selected_store')
    st.sidebar.markdown(f"🔒 Region: {location}\n\n🔒 Store: {store}")
    df = load_dataset(DATASET_MAP[location])

st.sidebar.markdown("""
## **This DCX analysis tool is only permitted for use in the following cases:**
* When used in educational settings such as universities for student education and research
* When used by small business owners for their own business purposes
* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners

<span style="color:red; font-weight:bold">
Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.
</span>
<br>
<br>
<br>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; font-size:16px; font-weight:bold; margin-bottom:10px;">
📬 Inquiries & Information
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
        📧 Contact via Email
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

# Tab setup
TABS = ["How to Use", "Photos & Reviews", "Word Cloud", "Treemap", "Network Analysis", "Topic Modeling", "Customer Satisfaction Analysis"]

if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = "How to Use"

st.markdown("""
<style>
label[for^=""] { color: black !important; font-weight: 600; }
div[data-testid="stMarkdownContainer"] p { color: black !important; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("location_locked", False):
    selected_tab = st.selectbox("✅ Please select a feature", TABS)
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
    selected_tab = "How to Use"
    st.warning("⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.")

# Execute tab-specific functions
if selected_tab == "How to Use":
    render_usage_tab()
elif selected_tab == "Photos & Reviews":
    render_review_tab(df, store)
elif selected_tab == "Word Cloud":
    render_wordcloud_tab(df, store)
elif selected_tab == "Treemap":
    render_treemap_tab(df, store)
elif selected_tab == "Network Analysis":
    render_network_tab(df, store)
elif selected_tab == "Topic Modeling":
    render_topic_tab(df, store)
elif selected_tab == "Customer Satisfaction Analysis":
    classifier = get_classifier()
    render_sentiment_dashboard(df, store, classifier)
