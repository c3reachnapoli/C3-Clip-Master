import streamlit as st
import gdown, os, json, time
from google import genai
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="C3 Reach Napoli - Pro Engine", layout="wide")

# --- API KEY ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

# --- FUNZIONE DI TAGLIO ---
def simple_cut(video_path, start, end):
    # Usiamo un context manager per MoviePy per liberare la memoria subito dopo il taglio
    with VideoFileClip(video_path) as video:
        clip = video.subclip(float(start), float(end))
        w, h = clip.size
        target_w = int(h * 9/16)
        x_center = w / 2
        # Crop centrale matematico
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

            with st.spinner("🧠 Gemini sta analizzando..."):
                # FIX: La nuova sintassi richiede il caricamento tramite file pointer
                with open("input.mp4", "rb") as f:
                    video_input = client.files.upload(file=f, config={'mime_type': 'video/mp4'})
                
                while video_input.state.name == "PROCESSING":
                    time.sleep(3)
                    video_input = client.files.get(name=video_input.name)
                
                prompt = "Analizza questo video di predicazione. Trova i 5 momenti più d'impatto (30-40 secondi). Rispondi SOLO in formato JSON: [{'start': secondi, 'end': secondi, 'title': 'Titolo'}]"
                response = client.models.generate_content(model="gemini-1.5-flash", contents=[video_input, prompt])
                
                # Parsing del JSON pulito
                raw_text = response.text
                start_json = raw_text.find("[")
                end_json = raw_text.rfind("]") + 1
                st.session_state.clips = json.loads(raw_text[start_json:end_json])
                st.success("Analisi completata con successo!")

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")

# --- DISPLAY E DOWNLOAD ---
if "clips" in st.session_state:
    st.divider()
    for i, clip in enumerate(st.session_state.clips):
        with st.container(border=True):
            st.subheader(f"Clip {i+1}: {clip['title']}")
            st.write(f"⏱ Tempo: {clip['start']}s - {clip['end']}s")
            
            # Per evitare problemi di ricaricamento Streamlit, generiamo il download button solo dopo il taglio
            if st.button(f"🎬 Prepara Download {i+1}", key=f"btn_{i}"):
                try:
                    with st.spinner("Generazione file in corso..."):
                        output_path = simple_cut("input.mp4", clip['start'], clip['end'])
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Scarica Ora",
                                data=f,
                                file_name=f"c3_reel_{i+1}.mp4",
                                mime="video/mp4",
                                key=f"dl_{i}"
                            )
                except Exception as e:
                    st.error(f"Errore nel taglio: {e}")
