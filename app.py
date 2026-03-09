import streamlit as st
import os, json, time, cv2
import numpy as np
import google.generativeai as genai
import mediapipe as mp
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="C3 Reach Napoli - AI Cameraman", layout="wide")

# Download del modello MediaPipe se non presente
if not os.path.exists('detector.tflite'):
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", "detector.tflite")

# --- SIDEBAR: PARAMETRI DI REGIA ---
with st.sidebar:
    st.image("https://images.squarespace-cdn.com/content/v1/5f1ef45f8e53995874492379/1596794646734-7B0X8W7L8O7B7X7X7X7X/C3-Global-Logo-Black.png", width=100)
    st.header("⚙️ Regia Cinematografica")
    api_key = st.text_input("Inserisci Gemini API Key", type="password")
    
    st.subheader("Parametri Camera")
    inerzia = st.slider("Fluidità (Inerzia)", 0.01, 0.20, 0.05, help="Più basso è, più la camera è 'pesante' e fluida.")
    dead_zone = st.slider("Dead Zone", 0.05, 0.30, 0.12, help="L'area centrale in cui il pastore può muoversi senza che la camera si sposti.")
    
    st.subheader("Stile Sottotitoli")
    color_sub = st.color_picker("Colore Testo", "#FFFFFF")
    font_size = st.number_input("Dimensione Font", 20, 100, 45)

# --- FUNZIONI CORE ---
def analizza_video(video_path, num_reel):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    video_ai = genai.upload_file(path=video_path)
    while video_ai.state.name == "PROCESSING":
        time.sleep(2)
        video_ai = genai.get_file(video_ai.name)
    
    prompt = f"Analizza il video. Trova i {num_reel} momenti più potenti con senso compiuto (frasi intere). Durata 30-50s. Rispondi SOLO JSON: [{{'start': 10, 'end': 50, 'title': 'Esempio'}}]"
    response = model.generate_content([video_ai, prompt])
    return json.loads(response.text[response.text.find("["):response.text.rfind("]")+1])

# --- INTERFACCIA PRINCIPALE ---
st.title("🎬 C3 Reach Napoli: AI Reel Factory")
st.write("Trasforma le tue predicazioni in contenuti verticali professionali con tracking cinematico.")

uploaded_file = st.file_uploader("Carica il video della predicazione (MP4)", type=["mp4"])

if uploaded_file and api_key:
    # Salvataggio temporaneo del file caricato
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🔍 Genera 10 Proposte di Reel"):
        with st.spinner("Gemini sta analizzando il messaggio..."):
            st.session_state.clips = analizza_video("temp_video.mp4", 10)
            st.success(f"Trovati {len(st.session_state.clips)} momenti epici!")

    # MOSTRA GRIGLIA ANTEPRIME
    if 'clips' in st.session_state:
        st.divider()
        cols = st.columns(2)
        for i, clip_data in enumerate(st.session_state.clips):
            with cols[i % 2]:
                with st.container(border=True):
                    st.subheader(f"Opzione {i+1}: {clip_data['title']}")
                    st.write(f"⏱ Inizio: {clip_data['start']}s | Fine: {clip_data['end']}s")
                    
                    if st.button(f"🎬 Renderizza Reel {i+1}", key=f"btn_{i}"):
                        # Qui inseriresti tutta la logica di rendering (MediaPipe + MoviePy)
                        # Per ora mostriamo un placeholder
                        st.warning("Inizio rendering... (Questo processo richiede potenza di calcolo)")
                        # [Logica di rendering V26 qui...]
                        st.success(f"Reel {i+1} completato! Scaricalo qui sotto.")
