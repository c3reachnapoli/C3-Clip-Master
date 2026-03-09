import streamlit as st
import gdown, os, json, time, numpy as np
from google import genai

# Carichiamo le librerie pesanti solo all'interno delle funzioni
# per evitare che l'app crashi all'avvio
def get_ai_clips(video_path, api_key):
    client = genai.Client(api_key=api_key)
    # Estrazione audio rapida
    import moviepy.editor as mp_edit
    video = mp_edit.VideoFileClip(video_path)
    audio_path = "temp_audio.mp3"
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()

    upload = client.files.upload(path=audio_path)
    while upload.state.name == "PROCESSING":
        time.sleep(2)
        upload = client.files.get(name=upload.name)
    
    prompt = "Trova 10 momenti carismatici (30-50s). Rispondi SOLO JSON: [{'start': s, 'end': s, 'title': 't'}]"
    resp = client.models.generate_content(model="gemini-1.5-flash", contents=[upload, prompt])
    return json.loads(resp.text[resp.text.find("["):resp.text.rfind("]")+1])

st.set_page_config(page_title="C3 Reach Napoli", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("API Key", type="password")

st.title("🎬 C3 Reach Napoli - Reel Factory")

url = st.text_input("Link Google Drive Video")

if url and API_KEY:
    if st.button("🚀 Inizia Analisi"):
        try:
            with st.spinner("Scaricamento e analisi in corso..."):
                id_drive = url.split('/')[-2] if 'view' in url else url.split('id=')[-1]
                gdown.download(id=id_drive, output="input.mp4", quiet=False)
                st.session_state.clips = get_ai_clips("input.mp4", API_KEY)
                st.success("Analisi Completata!")
        except Exception as e:
            st.error(f"Errore: {e}")

if "clips" in st.session_state:
    for i, clip in enumerate(st.session_state.clips):
        st.write(f"📍 {clip['title']} ({clip['start']}s - {clip['end']}s)")
        # Qui aggiungeremo il tasto per il rendering una volta che l'app si apre
