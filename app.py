import streamlit as st
import gdown, os, json, time
from google import genai
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

st.set_page_config(page_title="C3 Reach Napoli - Pro Engine", layout="wide")

# --- API KEY ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

# --- REQUISITI MINIMI ---
# Questo codice NON usa MediaPipe (niente crash cv2)
def simple_cut(video_path, start, end, title):
    clip = VideoFileClip(video_path).subclip(start, end)
    
    # Resize verticale 9:16 semplice (crop centrale automatico)
    w, h = clip.size
    target_w = int(h * 9/16)
    x_center = w / 2
    clip_cropped = clip.crop(x1=x_center - target_w/2, y1=0, x2=x_center + target_w/2, y2=h)
    
    # Titolo semplice (se ImageMagick è presente, altrimenti solo video)
    try:
        txt = TextClip(title.upper(), fontsize=50, color='white', font='Arial-Bold', method='caption', size=(target_w*0.8, None)).set_duration(clip.duration).set_position(('center', 100))
        final = CompositeVideoClip([clip_cropped, txt])
    except:
        final = clip_cropped

    out_name = f"clip_{int(time.time())}.mp4"
    final.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24)
    return out_name

# --- INTERFACCIA ---
st.title("🎬 C3 Reach: Fast Reel Generator")
video_url = st.text_input("Incolla Link Google Drive (Accesso Pubblico)")

if video_url and API_KEY:
    if st.button("🚀 Analizza Predicazione"):
        try:
            client = genai.Client(api_key=API_KEY)
            
            with st.spinner("📥 Scaricamento video..."):
                file_id = video_url.split('/')[-2] if 'view' in video_url else video_url.split('id=')[-1]
                gdown.download(id=file_id, output="input.mp4", quiet=False)

            with st.spinner("🧠 Gemini sta analizzando il video..."):
                # Carichiamo il file su Google per l'analisi (non serve estrarre l'audio noi!)
                video_input = client.files.upload(path="input.mp4")
                while video_input.state.name == "PROCESSING":
                    time.sleep(2)
                    video_input = client.files.get(name=video_input.name)
                
                # Chiediamo i momenti migliori
                prompt = "Analizza questo video di predicazione. Trova i 5 momenti più d'impatto (30-40 secondi). Rispondi SOLO in JSON: [{'start': secondi, 'end': secondi, 'title': 'Titolo'}]"
                response = client.models.generate_content(model="gemini-1.5-flash", contents=[video_input, prompt])
                
                raw = response.text
                st.session_state.clips = json.loads(raw[raw.find("["):raw.rfind("]")+1])
                st.success("Momenti pronti!")

        except Exception as e:
            st.error(f"Errore: {e}")

if "clips" in st.session_state:
    for i, clip in enumerate(st.session_state.clips):
        with st.container(border=True):
            st.write(f"🎥 **{clip['title']}** ({clip['start']}s - {clip['end']}s)")
            if st.button(f"Esporta Clip {i+1}", key=f"btn_{i}"):
                out = simple_cut("input.mp4", clip['start'], clip['end'], clip['title'])
                with open(out, "rb") as f:
                    st.download_button("📥 Scarica MP4", f, file_name=f"reel_{i}.mp4")    if st.button("🚀 Inizia Analisi"):
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
