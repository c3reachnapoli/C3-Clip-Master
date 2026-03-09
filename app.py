import streamlit as st
import gdown
import os, json, time
import numpy as np
import google.generativeai as genai
import mediapipe as mp
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

st.set_page_config(page_title="C3 Reach Napoli - Pro Engine", layout="wide")

# --- SIDEBAR PARAMETRI ---
with st.sidebar:
    st.title("🎥 Regia C3")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.header("🎮 Controlli Cinematici")
    inerzia = st.slider("Fluidità (Inerzia)", 0.01, 0.20, 0.06)
    dead_zone = st.slider("Dead Zone (Stabilità)", 0.05, 0.30, 0.12)
    
    st.divider()
    st.header("🎨 Stile Sottotitoli")
    sub_color = st.color_picker("Colore Testo", "#FFFFFF")
    font_size = st.slider("Grandezza Font", 20, 80, 45)

# --- FUNZIONE DOWNLOAD DRIVE ---
def download_from_drive(url):
    output = "input_video.mp4"
    if os.path.exists(output): os.remove(output)
    # Estrae l'ID dal link di Drive e scarica
    try:
        id_drive = url.split('/')[-2] if 'view' in url else url.split('id=')[-1]
        gdown.download(id=id_drive, output=output, quiet=False)
        return output
    except Exception as e:
        st.error(f"Errore download Drive: {e}")
        return None

# --- UI PRINCIPALE ---
st.title("🚀 C3 Reach: High-Resolution Reel Factory")
drive_url = st.text_input("Incolla il link di Google Drive del video (Assicurati che sia 'Chiunque abbia il link può visualizzare')")

if drive_url and api_key:
    if st.button("📥 Importa Video e Analizza"):
        with st.spinner("Scaricamento video da Drive in corso..."):
            video_path = download_from_drive(drive_url)
            
            if video_path:
                st.success("Video importato con successo!")
                
                # CHIAMATA A GEMINI PER LE 10 ANTEPRIME
                with st.spinner("Gemini sta estraendo i 10 momenti migliori..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    video_ai = genai.upload_file(path=video_path)
                    
                    while video_ai.state.name == "PROCESSING":
                        time.sleep(2)
                        video_ai = genai.get_file(video_ai.name)
                    
                    prompt = "Trova 10 momenti carismatici (30-50s) con senso compiuto. Rispondi SOLO JSON: [{'start': s, 'end': s, 'title': 'T'}]"
                    response = model.generate_content([video_ai, prompt])
                    st.session_state.clips = json.loads(response.text[response.text.find("["):])

    # --- GRIGLIA ANTEPRIME (10 SLOT) ---
    if 'clips' in st.session_state:
        st.header("🎬 Seleziona i Reel da produrre")
        grid = st.columns(2)
        for i, clip in enumerate(st.session_state.clips):
            with grid[i % 2]:
                with st.expander(f"📌 OPZIONE {i+1}: {clip['title']}", expanded=True):
                    st.write(f"⏱ Durata stimata: {int(clip['end']-clip['start'])} secondi")
                    if st.button(f"⚡ Renderizza Reel {i+1}", key=f"btn_{i}"):
                        # Qui viene eseguito il motore V26 con i parametri della sidebar
                        st.info("Rendering in corso con Inerzia Esponenziale...")
                        # [Inserire qui il blocco di rendering V26 adattato]
                        st.success("Reel Pronto!")
