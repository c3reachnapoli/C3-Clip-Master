import streamlit as st
import gdown, os, json, time
from google import genai
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="C3 Reach Napoli - Pro Engine", layout="wide")

# --- API KEY ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

# --- FUNZIONE DI TAGLIO ULTRA-LIGHT ---
def simple_cut(video_path, start, end):
    with VideoFileClip(video_path) as video:
        # Taglio temporale
        clip = video.subclip(start, end)
        
        # Crop centrale 9:16 (senza MediaPipe, solo matematica)
        w, h = clip.size
        target_w = int(h * 9/16)
        x_center = w / 2
        final = clip.crop(x1=x_center - target_w/2, y1=0, x2=x_center + target_w/2, y2=h)
        
        out_name = f"reel_{int(time.time())}.mp4"
        final.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24, logger=None)
        return out_name

# --- INTERFACCIA ---
st.title("🎬 C3 Reach: Fast Reel Generator")
video_url = st.text_input("Link Google Drive (Accesso Pubblico)")

if video_url and API_KEY:
    if st.button("🚀 Analizza Video"):
        try:
            client = genai.Client(api_key=API_KEY)
            
            with st.spinner("📥 Scaricamento video da Drive..."):
                file_id = video_url.split('/')[-2] if 'view' in video_url else video_url.split('id=')[-1]
                if os.path.exists("input.mp4"): os.remove("input.mp4")
                gdown.download(id=file_id, output="input.mp4", quiet=False)

            with st.spinner("🧠 Gemini sta analizzando il contenuto..."):
                video_input = client.files.upload(path="input.mp4")
                while video_input.state.name == "PROCESSING":
                    time.sleep(2)
                    video_input = client.files.get(name=video_input.name)
                
                prompt = "Analizza questo video di predicazione. Trova i 5 momenti più d'impatto (30-40 secondi). Rispondi SOLO in formato JSON: [{'start': secondi, 'end': secondi, 'title': 'Titolo'}]"
                response = client.models.generate_content(model="gemini-1.5-flash", contents=[video_input, prompt])
                
                # Pulizia JSON
                raw = response.text
                st.session_state.clips = json.loads(raw[raw.find("["):raw.rfind("]")+1])
                st.success("Analisi completata!")

        except Exception as e:
            st.error(f"Errore: {e}")

# --- DISPLAY RISULTATI ---
if "clips" in st.session_state:
    st.divider()
    for i, clip in enumerate(st.session_state.clips):
        with st.container(border=True):
            st.write(f"🎥 **{clip['title']}**")
            st.write(f"⏱ {clip['start']}s - {clip['end']}s")
            
            if st.button(f"Genera Clip {i+1}", key=f"gen_{i}"):
                try:
                    with st.spinner("Taglio video in corso..."):
                        output_file = simple_cut("input.mp4", clip['start'], clip['end'])
                        with open(output_file, "rb") as f:
                            st.download_button("📥 Scarica MP4", f, file_name=f"reel_{i}.mp4", key=f"dl_{i}")
                except Exception as e:
                    st.error(f"Errore durante il taglio: {e}")
