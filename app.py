import streamlit as st
import gdown, os, json, time
from google import genai
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="C3 Reach Napoli - Audio Optimization", layout="wide")

# --- API KEY ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Inserisci API Key", type="password")

# --- MOTORE DI TAGLIO ---
def cut_video(path, start, end):
    with VideoFileClip(path) as video:
        s, e = float(start), float(end)
        clip = video.subclip(s, e)
        w, h = clip.size
        target_w = int(h * 9/16)
        final = clip.crop(x_center=w/2, width=target_w, y_center=h/2, height=h)
        
        out_name = f"reel_{int(time.time())}.mp4"
        final.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24, logger=None)
        return out_name

# --- INTERFACCIA ---
st.title("🎙️ C3 Reach: Analizzatore Audio-Video")
video_url = st.text_input("Link Google Drive del Video")

if video_url and API_KEY:
    if st.button("🚀 Estrai e Analizza (Low Quota)"):
        try:
            client = genai.Client(api_key=API_KEY)
            
            with st.spinner("📥 1. Scaricamento video..."):
                file_id = video_url.split('/')[-2] if 'view' in video_url else video_url.split('id=')[-1]
                if os.path.exists("input.mp4"): os.remove("input.mp4")
                gdown.download(id=file_id, output="input.mp4", quiet=False)

            with st.spinner("✂️ 2. Estrazione traccia audio..."):
                # Estraiamo l'audio per risparmiare migliaia di token
                with VideoFileClip("input.mp4") as video:
                    video.audio.write_audiofile("temp_voice.mp3", logger=None)
            
            with st.spinner("🧠 3. Gemini sta ascoltando la predicazione..."):
                # Carichiamo solo l'audio
                with open("temp_voice.mp3", "rb") as f:
                    audio_file = client.files.upload(
                        file=f, 
                        config={'mime_type': 'audio/mpeg'}
                    )
                
                while audio_file.state.name == "PROCESSING":
                    time.sleep(2)
                    audio_file = client.files.get(name=audio_file.name)
                
                prompt = """Analizza l'audio di questa predicazione. 
                Trova i 5 momenti più carismatici e d'ispirazione (30-45 secondi ciascuno). 
                Restituisci ESCLUSIVAMENTE un JSON: [{"start": secondi, "end": secondi, "title": "Titolo"}]"""
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=[audio_file, prompt]
                )
                
                clean_json = response.text[response.text.find("["):response.text.rfind("]")+1]
                st.session_state.clips = json.loads(clean_json)
                st.success("Analisi completata con successo risparmiando quota!")

        except Exception as e:
            st.error(f"Errore tecnico: {e}")

# --- GRID OUTPUT ---
if "clips" in st.session_state:
    st.divider()
    for i, clip in enumerate(st.session_state.clips):
        with st.container(border=True):
            st.subheader(f"💎 {clip['title']}")
            if st.button(f"🎬 Crea Reel {i+1}", key=f"gen_{i}"):
                with st.spinner("Rendering video in corso..."):
                    path = cut_video("input.mp4", clip['start'], clip['end'])
                    with open(path, "rb") as f:
                        st.download_button("📥 Scarica", f, file_name=f"c3_reel_{i+1}.mp4")# --- GRID OUTPUT ---
if "clips" in st.session_state:
    st.divider()
    for i, clip in enumerate(st.session_state.clips):
        with st.container(border=True):
            st.subheader(f"🔥 {clip['title']}")
            if st.button(f"🎬 Crea Reel {i+1}", key=f"gen_{i}"):
                with st.spinner("Taglio..."):
                    path = cut_video("input.mp4", clip['start'], clip['end'])
                    with open(path, "rb") as f:
                        st.download_button("📥 Scarica", f, file_name=f"reel_{i+1}.mp4")
