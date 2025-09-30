# ==================================
#          MeloDetect App 
# ==================================

import streamlit as st
import pandas as pd
import pickle
import re
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==================================
# KONFIGURASI HALAMAN & GAYA (CSS)
# ==================================
st.set_page_config(
    page_title="MeloDetect",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto"
)

# === BLOK CSS PERBAIKAN SIDEBAR & JUDUL ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* Hilangkan tombol collapse sidebar */
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }

    /* Variabel Warna */
    :root {
        --primary-color: #8A0000;   /* merah bata */
        --secondary-color: #C83F12; /* oranye merah */
        --accent-color: #FFD700;    /* emas */
        --bg-color: #FFF8F0;
        --card-bg: #FFFFFF;
    }

    /* Global Styles - Gunakan Poppins tanpa merusak styling lain */
html, body, [class*="css"], div, span, input, textarea, button, select, label, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: inherit;   /* biar tetap pakai ketebalan bawaan */
    font-size: inherit;     /* biar ukuran bawaan Streamlit tidak rusak */
    color: inherit;         /* pakai warna bawaan, tidak dipaksa */
}


    /* Judul Utama */
    h1 {
        color: var(--primary-color); /* merah bata */
        text-align: center;
        font-weight: 800;
        font-size: 3.5em !important; /* lebih besar */
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        color: white;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: white !important;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--card-bg);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        transition: 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.15);
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        gap: 10px;
        justify-content: center !important; /* biar rata tengah */
        display: flex !important;
            
    .stTabs [role="tab"] {
        background-color: #eee;
        padding: 8px 16px;
        border-radius: 10px;
        color: #3B060A;
        font-weight: 500;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================================
# FUNGSI-FUNGSI LOGIKA UTAMA 
# ==================================

def preprocess_text(text):
    """Fungsi untuk membersihkan teks input (lirik)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_song(query_lyric, vectorizer, matrix, dataframe):
    """Mencari lagu yang paling mirip berdasarkan query lirik menggunakan cosine similarity."""
    if not query_lyric:
        return None, 0
    
    cleaned_query = preprocess_text(query_lyric)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, matrix).flatten()
    
    most_similar_song_index = np.argmax(similarities)
    highest_score = similarities[most_similar_song_index]
    
    if highest_score < 0.1:
        return None, highest_score
        
    song_details = dataframe.iloc[most_similar_song_index]
    return song_details, highest_score

def recommend_similar_songs(song_title, dataframe, matrix, top_n=5):
    """Merekomendasikan lagu berdasarkan kemiripan konten (lirik)."""
    try:
        song_idx = dataframe[dataframe['title'] == song_title].index[0]
        song_vector = matrix[song_idx]
        similarities = cosine_similarity(song_vector, matrix).flatten()
        similar_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
        return dataframe.iloc[similar_indices]
    except (IndexError, KeyError):
        return pd.DataFrame()

# ==================================
# MEMUAT & MEMPROSES DATA 
# ==================================

@st.cache_resource
def load_models_and_data():
    """Memuat model, vectorizer, dan dataframe mentah sekali saja."""
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
    dataframe = pd.read_pickle('lirik_data.pkl')
    return vectorizer, matrix, dataframe

@st.cache_data
def prepare_dashboard_data(_df):
    """Mempersiapkan data yang sudah diproses untuk dashboard agar cepat dimuat."""
    df_processed = _df.copy()
    if 'full' not in df_processed.columns:
        df_processed['full'] = "Lirik tidak tersedia."
    
    df_processed = df_processed.drop_duplicates(subset=['title', 'artist'])
    
    total_songs = len(df_processed)
    total_artists = df_processed['artist'].nunique()
    total_emotions = df_processed['emotion'].nunique()
    
    emotion_counts = df_processed['emotion'].value_counts()
    top_artists = df_processed['artist'].value_counts().head(10)
    
    all_lyrics = ' '.join(df_processed['full'].dropna())
    
    return {
        "stats": (total_songs, total_artists, total_emotions),
        "emotion_counts": emotion_counts,
        "top_artists": top_artists,
        "all_lyrics": all_lyrics
    }

# Panggil fungsi untuk memuat semua aset
try:
    tfidf_vectorizer, tfidf_matrix, df = load_models_and_data()
    dashboard_data = prepare_dashboard_data(df)
    emotions = [""] + sorted(list(df['emotion'].unique()))
    artist_list = [""] + sorted(list(df['artist'].unique()))
except FileNotFoundError:
    st.error("File model/data tidak ditemukan! Pastikan 'lirik_data.pkl', 'tfidf_vectorizer.pkl', dan 'tfidf_matrix.pkl' ada di folder yang sama.")
    st.stop()

# ==================================
# TAMPILAN ANTARMUKA (UI)
# ==================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Tentang MeloDetect</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        **MeloDetect** adalah aplikasi berbasis AI untuk mendeteksi, menjelajahi, dan mendapatkan rekomendasi musik berdasarkan lirik.
        
        **Fitur Utama:**
        - **Dashboard:** Visualisasi data dari koleksi musik.
        - **Temukan Lagu:** Cari lagu dari suara senandung atau ketikan lirik.
        - **Jelajahi Musik:** Temukan lagu berdasarkan mood atau artis favorit.
        """
    )
    st.markdown("---")
    st.info("Selamat menjelajahi dunia musik dengan MeloDetect.")

# --- KONTEN UTAMA ---
st.markdown("""
<div style="text-align:center; margin-top: -30px; margin-bottom: -10px;">
    <h1 style="color:#8A0000; font-size: 3em;">
        🎵 MeloDetect 🎵
    </h1>
    <p style="font-size:1.3em; color:#444; font-style:italic; margin-top: -10px;">
        Lirik nyangkut di kepala?<span style="color:#8A0000; font-weight:600;"> MeloDetect kasih tau lagunya! </span>
    </p>
</div>
""", unsafe_allow_html=True)

# Definisikan struktur tab utama
tab_dashboard, tab_find, tab_explore = st.tabs(["📊 **Dashboard Ringkasan**", "🔍 **Temukan Lagu**", "🧭 **Jelajahi Musik**"])

# --- TAB 1: DASHBOARD ---
with tab_dashboard:
    st.header("Ringkasan Dataset Musik", divider='rainbow')
    
    # === Metric Cards ===
    total_songs, total_artists, total_emotions = dashboard_data["stats"]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎶 Total Lagu</h3>
            <p style="font-size:2em; font-weight:700;">{total_songs:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧑‍🎤 Total Artis Unik</h3>
            <p style="font-size:2em; font-weight:700;">{total_artists:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Total Mood</h3>
            <p style="font-size:2em; font-weight:700;">{total_emotions}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # === Chart + Wordcloud ===
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📈 Distribusi Mood Lagu")
        st.bar_chart(dashboard_data["emotion_counts"], color="#C83F12")
        st.subheader("🏆 Artis Paling Produktif")
        st.bar_chart(dashboard_data["top_artists"], color="#8A0000")
        
    with col2:
        with st.container(border=False):
            st.subheader("☁️ Kata Paling Umum dalam Lirik")
            with st.spinner("Membuat word cloud..."):
                try:
                    wordcloud = WordCloud(
                        width=800, height=520,
                        background_color='white',
                        colormap='autumn'
                    ).generate(dashboard_data["all_lyrics"])
                    fig, ax = plt.subplots(figsize=(10, 6.5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except ValueError:
                    st.warning("Tidak cukup kata unik untuk membuat word cloud.")

# --- TAB 2: TEMUKAN LAGU ---
with tab_find:
    st.header("Temukan Lagu Spesifik", divider='rainbow')
    find_tab1, find_tab2 = st.tabs(["🎤 **Dari Suara Anda**", "📝 **Dari Ketikan Lirik**"])

    with find_tab1:
        st.info("Untuk hasil terbaik, bernyanyilah dengan jelas selama 10-15 detik di lingkungan yang tidak bising.", icon="💡")
        col1, col2 = st.columns(2)
        with col1:
            recording_duration = st.slider("Durasi rekaman (detik):", 5, 30, 15, 1, key="slider_voice")
        with col2:
            model_options = { "Dasar": "base", "Kecil": "small", "Menengah": "medium" }
            display_options = list(model_options.keys())
            selected_display_model = st.selectbox(
                "Model pengenalan suara:", options=display_options, index=1,
                key="model_voice", help="Model 'Dasar' atau 'Kecil' lebih cepat, 'Menengah' lebih akurat."
            )
            selected_model = model_options[selected_display_model]
        
        if st.button("🔴 Mulai Merekam & Deteksi", use_container_width=True, type="primary"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Menyesuaikan dengan suara sekitar...", icon="🤫")
                try: 
                    r.adjust_for_ambient_noise(source, duration=1)
                except Exception as e: 
                    st.warning(f"Gagal menyesuaikan noise: {e}")

                st.info(f"Mendengarkan selama {recording_duration} detik...", icon="🎧")
                try:
                    audio = r.record(source, duration=recording_duration)
                    st.success("Rekaman selesai, memproses audio...", icon="✅")
                    
                    with st.spinner(f"Menerjemahkan suara menjadi teks (Model: {selected_display_model})..."):
                        text = r.recognize_whisper(audio, language="id", model=selected_model)
                    
                    st.info(f"**Lirik Terdeteksi:** *'{text}'*")
                    if text:
                        with st.spinner('Mencocokkan lirik dengan database...'):
                            found_song, score = find_song(text, tfidf_vectorizer, tfidf_matrix, df)
                        st.session_state.found_song = found_song
                        st.session_state.score = score
                    else:
                        st.warning("Tidak ada lirik yang terdeteksi. Coba lagi.")
                except sr.UnknownValueError:
                    st.error("Tidak dapat memahami audio. Coba nyanyikan lebih jelas.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat rekaman: {e}")

    with find_tab2:
        lyric_input = st.text_input("Ketikkan sepenggal lirik:", placeholder="contoh: ketika mimpimu yang begitu indah", label_visibility="collapsed")
        if st.button("Cari Lagu Berdasarkan Teks", use_container_width=True, type="primary"):
            if lyric_input:
                with st.spinner('Mencocokkan lirik dengan database...'):
                    found_song, score = find_song(lyric_input, tfidf_vectorizer, tfidf_matrix, df)
                st.session_state.found_song = found_song
                st.session_state.score = score
            else:
                st.warning("Silakan masukkan lirik terlebih dahulu.")
    
    # Area untuk menampilkan hasil pencarian di luar tabs
    if 'found_song' in st.session_state:
        st.markdown("---")
        st.header("✨ Hasil Pencarian")
        song_details = st.session_state.found_song
        
        if song_details is not None:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{song_details['title']}")
                    st.caption(f"oleh **{song_details['artist']}** | Mood: **{song_details['emotion'].capitalize()}**")
                
                with col2:
                    st.progress(float(st.session_state.score), text=f"Kemiripan: {st.session_state.score:.0%}")
                    judul = song_details['title']
                    artis = song_details['artist']
                    youtube_search_url = f"https://www.youtube.com/results?search_query={artis.replace(' ', '+')}+{judul.replace(' ', '+')}+lyrics"
                    st.link_button("Dengarkan di YouTube", youtube_search_url, use_container_width=True)

            st.text_area("Lirik Lagu Lengkap", song_details.get('full', 'Lirik tidak tersedia.'), height=300)
            
            st.subheader("Rekomendasi Lagu Lain yang Mirip:")
            similar_songs = recommend_similar_songs(song_details['title'], df, tfidf_matrix)
            if not similar_songs.empty:
                cols = st.columns(len(similar_songs))
                for idx, (_, row) in enumerate(similar_songs.iterrows()):
                    with cols[idx]:
                        with st.container(border=True):
                           st.markdown(f"**{row['title']}**")
                           st.caption(f"oleh {row['artist']}")
            else:
                st.info("Tidak ada rekomendasi lagu serupa yang ditemukan.")
        else:
            st.error("Maaf, lagu tidak ditemukan. Coba lagi dengan lirik yang lebih jelas atau berbeda.", icon="😕")
        
        del st.session_state['found_song']
        del st.session_state['score']

# --- TAB 3: JELAJAHI MUSIK ---
with tab_explore:
    st.header("Jelajahi Musik Berdasarkan Kategori", divider='rainbow')
    explore_tab1, explore_tab2 = st.tabs(["😊 **Rekomendasi Berdasarkan Mood**", "🧑‍🎤 **Eksplorasi Artis**"])

    with explore_tab1:
        selected_mood = st.selectbox("Pilih mood Anda saat ini:", options=emotions, key="select_mood", label_visibility="collapsed")
        if selected_mood:
            st.success(f"Berikut 5 rekomendasi lagu acak dengan mood **{selected_mood}**:")
            recommended_songs = df[df['emotion'] == selected_mood]
            if not recommended_songs.empty:
                display_songs = recommended_songs.sample(min(5, len(recommended_songs)))[['title', 'artist']]
                st.dataframe(display_songs, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada lagu di database yang cocok dengan mood ini.")

    with explore_tab2:
        selected_artist = st.selectbox("Pilih artis untuk melihat lagu-lagunya:", options=artist_list, key="select_artist", label_visibility="collapsed")
        if selected_artist:
            st.subheader(f"Statistik untuk {selected_artist}", divider="gray")
            artist_songs = df[df['artist'] == selected_artist].reset_index(drop=True)
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.metric("Total Lagu", f"{len(artist_songs)} lagu")
                if not artist_songs.empty:
                    mood_dist = artist_songs['emotion'].value_counts().reset_index()
                    mood_dist.columns = ['Mood', 'Jumlah']
                    st.write("**Distribusi Mood:**")
                    st.dataframe(mood_dist, use_container_width=True, hide_index=True)
            with col2:
                st.write(f"**Daftar Lagu oleh {selected_artist}:**")
                st.dataframe(artist_songs[['title', 'emotion']], use_container_width=True, hide_index=True, height=300)