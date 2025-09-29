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
# KONFIGURASI HALAMAN
# ==================================
st.set_page_config(
    page_title="MeloDetect",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================
# FUNGSI-FUNGSI LOGIKA
# ==================================

# Fungsi untuk membersihkan teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk mencari lagu
def find_song(query_lyric, vectorizer, matrix, dataframe):
    if not query_lyric:
        return "Tidak Ditemukan", "Input kosong.", 0
    cleaned_query = preprocess_text(query_lyric)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, matrix).flatten()
    most_similar_song_index = np.argmax(similarities)
    song_details = dataframe.iloc[most_similar_song_index]
    highest_score = similarities[most_similar_song_index]
    
    # Menetapkan ambang batas kemiripan
    if highest_score < 0.1:
        return "Tidak Ditemukan", "Lirik tidak cocok dengan lagu manapun di database.", 0
        
    return song_details['title'], song_details['artist'], highest_score

# Fungsi untuk menampilkan kartu hasil pencarian
def display_song_result(judul, artis, skor):
    if judul != "Tidak Ditemukan":
        st.success(f"**Lagu Ditemukan!**")
        
        # UI Kartu Hasil
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{judul}")
                st.caption(f"Oleh: **{artis}**")
                st.progress(float(skor), text=f"Tingkat Kemiripan: {skor:.1%}")
            with col2:
                # Membuat URL pencarian YouTube
                youtube_search_url = f"https://www.youtube.com/results?search_query={artis.replace(' ', '+')}+{judul.replace(' ', '+')}+lyrics"
                st.link_button("Dengarkan di YouTube ↗", youtube_search_url, use_container_width=True)
    else:
        st.error("Maaf, lagu tidak ditemukan. Coba lagi dengan lirik yang lebih jelas atau berbeda.")

# ==================================
# MEMUAT DATA & MODEL (dengan caching)
# ==================================

@st.cache_resource
def load_all():
    """Memuat semua data dan model yang diperlukan sekali saja."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('tfidf_matrix.pkl', 'rb') as f:
            matrix = pickle.load(f)
        dataframe = pd.read_pickle('lirik_data.pkl')
        # Pastikan kolom yang dibutuhkan ada
        required_cols = ['title', 'artist', 'emotion', 'full']
        if not all(col in dataframe.columns for col in required_cols):
             st.error(f"File 'lirik_data.pkl' harus memiliki kolom: {', '.join(required_cols)}")
             st.stop()
        return vectorizer, matrix, dataframe
    except FileNotFoundError:
        st.error("File model/data tidak ditemukan! Pastikan 'lirik_data.pkl', 'tfidf_vectorizer.pkl', dan 'tfidf_matrix.pkl' ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        st.stop()

# Panggil fungsi load
tfidf_vectorizer, tfidf_matrix, df = load_all()
emotions_list = sorted(list(df['emotion'].unique()))

# Fungsi untuk Word Cloud (dengan caching data)
@st.cache_data
def generate_wordcloud(emotion, dataframe):
    """Membuat gambar word cloud untuk emosi tertentu."""
    text = ' '.join(dataframe[dataframe['emotion'] == emotion]['full'])
    if not text:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# ==================================
# TAMPILAN ANTARMUKA (UI)
# ==================================

st.title("🎵 MeloDetect: Analisis & Rekomendasi Musik")
st.markdown("Temukan lagu dari suara atau lirik, dapatkan rekomendasi sesuai mood, dan jelajahi tren musik Indonesia.")

# --- TABS UNTUK NAVIGASI ---
tab1, tab2, tab3 = st.tabs([
    "🎤 Deteksi & Cari Lagu", 
    "😊 Rekomendasi Mood", 
    "📊 Dashboard Analisis"
])


# ================= TAB 1: DETEKSI & CARI LAGU =================
with tab1:
    st.header("Temukan Judul Lagu Misterius")
    col_voice, col_text = st.columns(2)

    # --- FITUR 1: DETEKSI VIA SUARA ---
    with col_voice:
        st.subheader("Dari Suara Anda")
        st.caption("Untuk hasil terbaik, bernyanyilah dengan jelas di lingkungan yang tidak bising.")
        
        recording_duration = st.slider("Durasi rekaman (detik):", 5, 30, 10, key="slider_voice")
        
        selected_model = st.selectbox(
            "Pilih model pengenalan suara:",
            options=["base", "small", "medium"], index=1,
            help="Model 'base' paling cepat. 'Medium' paling akurat tapi lebih lambat."
        )

        if st.button("🎤 Mulai Merekam", use_container_width=True):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Menyesuaikan dengan suara sekitar...")
                try:
                    r.adjust_for_ambient_noise(source, duration=1)
                except Exception as e:
                    st.warning(f"Tidak dapat menyesuaikan noise: {e}")

                st.info(f"Mendengarkan {recording_duration} detik...")
                try:
                    audio = r.record(source, duration=recording_duration)
                    st.success("Rekaman selesai, memproses audio...")
                    
                    spinner_text = f"Menerjemahkan suara menjadi teks (Model: {selected_model})..."
                    with st.spinner(spinner_text):
                        text = r.recognize_whisper(audio, language="id", model=selected_model)
                    
                    st.write(f"**Lirik Terdeteksi:** *'{text}'*")
                    
                    if text:
                        with st.spinner('Mencocokkan lirik dengan database...'):
                            judul, artis, skor = find_song(text, tfidf_vectorizer, tfidf_matrix, df)
                        display_song_result(judul, artis, skor)
                    else:
                        st.warning("Tidak ada lirik yang terdeteksi. Coba lagi.")
                except sr.UnknownValueError:
                    st.error("Tidak dapat memahami audio. Coba nyanyikan lebih jelas.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    # --- FITUR 2: PENCARIAN VIA TEKS ---
    with col_text:
        st.subheader("Dari Potongan Lirik")
        st.caption("Ketik beberapa kata dari lirik yang Anda ingat.")
        
        lyric_query = st.text_area("Masukkan lirik di sini:", height=150, placeholder="contoh: hari ini aku gembira melangkah di udara...")
        
        if st.button("🔍 Cari Lagu", use_container_width=True):
            if lyric_query:
                with st.spinner('Mencari lagu yang cocok...'):
                    judul, artis, skor = find_song(lyric_query, tfidf_vectorizer, tfidf_matrix, df)
                display_song_result(judul, artis, skor)
            else:
                st.warning("Silakan masukkan potongan lirik terlebih dahulu.")


# ================= TAB 2: REKOMENDASI MOOD =================
with tab2:
    st.header("Playlist Sesuai Suasana Hati Anda")
    st.write("Butuh lagu yang pas dengan perasaanmu saat ini? Pilih mood di bawah ini.")

    selected_mood = st.selectbox(
        "Pilih mood Anda:",
        options=[""] + emotions_list,
        format_func=lambda x: "Pilih salah satu..." if x == "" else x.capitalize()
    )

    if selected_mood:
        st.subheader(f"Rekomendasi Lagu untuk Mood '{selected_mood.capitalize()}'")
        
        recommended_songs = df[df['emotion'] == selected_mood]
        
        if not recommended_songs.empty:
            display_songs = recommended_songs.sample(min(10, len(recommended_songs)))[['title', 'artist']]
            display_songs.reset_index(drop=True, inplace=True)
            display_songs.index += 1 # Mulai index dari 1
            st.dataframe(display_songs, use_container_width=True, height=387)
        else:
            st.write("Tidak ada lagu yang cocok dengan mood ini.")


# ================= TAB 3: DASHBOARD ANALISIS =================
with tab3:
    st.header("📊 Wawasan Dunia Musik Indonesia")
    st.write("Jelajahi tren dan pola menarik dari database lirik lagu kami.")
    
    # --- VISUALISASI 1: DISTRIBUSI EMOSI ---
    st.subheader("Peta Emosi Musik")
    st.caption("Bagaimana sebaran emosi dalam lagu-lagu yang ada di database?")
    emotion_counts = df['emotion'].value_counts()
    st.bar_chart(emotion_counts)
    st.markdown("""
    Chart ini menunjukkan jumlah lagu untuk setiap kategori emosi. Anda bisa melihat emosi mana yang paling dominan diwakili dalam koleksi lagu ini.
    """)
    st.divider()

    # --- VISUALISASI 2: ARTIS PER MOOD ---
    st.subheader("Raja & Ratu Tiap Suasana Hati")
    st.caption("Siapa musisi yang paling sering menciptakan lagu dengan mood tertentu?")
    
    mood_for_artist = st.selectbox("Pilih sebuah mood untuk melihat artis teratas:", options=emotions_list)
    if mood_for_artist:
        top_artists = df[df['emotion'] == mood_for_artist]['artist'].value_counts().nlargest(10)
        st.dataframe(
            top_artists,
            use_container_width=True,
            column_config={
                "artist": "Artis",
                "count": "Jumlah Lagu",
            }
        )
    st.divider()

    # --- VISUALISASI 3: WORD CLOUD ---
    st.subheader("Kata Kunci dalam Setiap Emosi")
    st.caption("Kata apa yang paling sering muncul dalam lirik berdasarkan emosinya?")

    mood_for_wordcloud = st.selectbox("Pilih sebuah mood untuk melihat Word Cloud-nya:", options=emotions_list, key="wc_select")
    if mood_for_wordcloud:
        with st.spinner(f"Membuat word cloud untuk mood '{mood_for_wordcloud}'..."):
            wordcloud_fig = generate_wordcloud(mood_for_wordcloud, df)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.warning("Tidak ada data lirik untuk menghasilkan word cloud pada mood ini.")
