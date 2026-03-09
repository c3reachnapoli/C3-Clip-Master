import streamlit as st
import gdown, os, json, time
import numpy as np
import google.generativeai as genai
import mediapipe as mp
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

st.set_page_config(page_title="C3 Reach Napoli - Colab Edition", layout="wide")

# --- SECRETS ---
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Inserisci Gemini API Key", type="password")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://images.squarespace-cdn.com/content/v1/5f1ef45f8e53995874492379/1596794646734-7B0X8W7L8O7B7X7X7X7X/C3-Global-Logo-Black.png", width=120)
    st.header("⚙️ Regia")
    inerzia = st.slider("Fluidità", 0.01, 0.15, 0.06)
    dead_zone = st.slider("Stabilità", 0.05, 0.25, 0.12)
    sub_color = st.color_picker("Colore Testo", "#FFFFFF")
    font_size = st.slider("Grandezza", 30, 70, 48)

# --- MOTORE RENDERING ---
def render_reel(data, video_path, smooth, dz, color, f_size):
    with st.status(f"🎬 Creazione: {data['title']}...") as status:
        clip = VideoFileClip(video_path).subclipped(data['start'], data['end'])
        w_orig, h_orig = clip.size
        target_w = int(h_orig * (9/16))
        
        if not os.path.exists('detector.tflite'):
            import urllib.request
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", "detector.tflite")
        
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='detector.tflite'),
            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            raw_pos = []
            for t in np.arange(0, clip.duration, 0.1):
                frame = clip.get_frame(t)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                res = detector.detect(mp_img)
                val = (res.detections[0].bounding_box.origin_x + res.detections[0].bounding_box.width/2)/w_orig if res.detections else None
                raw_pos.append(val)
            
            final_coords = []
            cam_x = 0.5
            for p in raw_pos:
                if p is not None:
                    dist = p - cam_x
                    if abs(dist) > dz: cam_x += (p - (np.sign(dist)*dz) - cam_x) * smooth
                final_coords.append(cam_x)

            def camera_op(get_frame, t):
                idx = min(int(t / 0.1), len(final_coords) - 1)
                cx = int(final_coords[idx] * w_orig)
                x1 = max(0, min(cx - (target_w // 2), w_orig - target_w))
                return get_frame(t)[:, int(x1):int(x1+target_w)]

            tracked = clip.transform(camera_op).with_effects([FadeIn(0.5), FadeOut(1.0)])
            
            try:
                txt = (TextClip(text=data['title'].upper(), font_size=f_size, color=color, 
                               stroke_color='black', stroke_width=2, method='caption', 
                               font="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                               size=(int(target_w*0.8), None)).with_duration(clip.duration).with_position(('center', 180)))
                final_v = CompositeVideoClip([tracked, txt])
            except:
                final_v = tracked
                
            out_name = f"REEL_{int(time.time())}.mp4"
            final_v.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24, logger=None)
            return out_name

# --- INTERFACCIA ---
st.title("🚀 C3 Reach: Motore AI")
drive_url = st.text_input("Link Google Drive del video (Accesso Pubblico)")

if drive_url and API_KEY:
    if st.button("📥 Importa e Analizza"):
        try:
            with st.spinner("1. Scarico video da Drive..."):
                id_drive = drive_url.split('/')[-2] if 'view' in drive_url else drive_url.split('id=')[-1]
                if os.path.exists("input.mp4"): os.remove("input.mp4")
                gdown.download(id=id_drive, output="input.mp4", quiet=False)
            
            with st.spinner("2. Preparazione (estrazione audio per AI)..."):
                v = VideoFileClip("input.mp4")
                v.audio.write_audiofile("audio.mp3", logger=None)
                v.close()
            
            with st.status("🧠 3. Interrogo Gemini (Stile Colab)...") as status:
                genai.configure(api_key=API_KEY)
                
                # Caricamento tramite l'API ufficiale (File API)
                audio_file = genai.upload_file(path="audio.mp3")
                
                # Attendiamo che Google processi il file
                while audio_file.state.name == "PROCESSING":
                    time.sleep(3)
                    audio_file = genai.get_file(audio_file.name)
                
                if audio_file.state.name == "ACTIVE":
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = "Trova i 10 momenti più carismatici (30-50s ciascuno). Rispondi SOLO con una lista JSON nel formato: [{'start': secondi, 'end': secondi, 'title': 'Titolo'}]"
                    
                    response = model.generate_content([audio_file, prompt])
                    
                    # Parsing sicuro
                    raw_text = response.text
                    start_idx = raw_text.find("[")
                    end_idx = raw_text.rfind("]") + 1
                    
                    if start_idx != -1:
                        st.session_state.clips = json.loads(raw_text[start_idx:end_idx])
                        st.success("Trovati 10 momenti!")
                    else:
                        st.error("L'AI non ha formattato bene la risposta. Riprova.")
                else:
                    st.error(f"Errore lato Google: File in stato {audio_file.state.name}")

        except Exception as e:
            st.error(f"Errore: {e}")

    # --- GRIGLIA ANTEPRIME ---
    if 'clips' in st.session_state:
        st.divider()
        grid = st.columns(2)
        for i, clip in enumerate(st.session_state.clips):
            with grid[i % 2]:
                with st.container(border=True):
                    st.subheader(f"Opzione {i+1}: {clip.get('title', 'Senza Titolo')}")
                    st.write(f"⏱ Segmento: {clip.get('start')}s - {clip.get('end')}s")
                    if st.button(f"⚡ Renderizza Reel {i+1}", key=f"r_{i}"):
                        f_out = render_reel(clip, "input.mp4", inerzia, dead_zone, sub_color, font_size)
                        if f_out:
                            with open(f_out, "rb") as f:
                                st.download_button("📥 Scarica", f, file_name=f_out)
