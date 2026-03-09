import streamlit as st
import gdown, os, json, time, cv2
import numpy as np
import google.generativeai as genai
import mediapipe as mp
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="C3 Reach Napoli - Pro Engine", layout="wide", page_icon="🎬")

# --- GESTIONE API KEY (SECRETS) ---
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Inserisci Gemini API Key", type="password")

# --- SIDEBAR: REGIA E STILE ---
with st.sidebar:
    st.image("https://images.squarespace-cdn.com/content/v1/5f1ef45f8e53995874492379/1596794646734-7B0X8W7L8O7B7X7X7X7X/C3-Global-Logo-Black.png", width=120)
    st.header("🎮 Controlli AI")
    inerzia = st.slider("Fluidità Camera (Inerzia)", 0.01, 0.15, 0.06, help="0.04 = Cinema, 0.10 = Reattivo")
    dead_zone = st.slider("Stabilità (Dead Zone)", 0.05, 0.25, 0.12, help="Area centrale ferma")
    
    st.divider()
    st.header("🎨 Stile Sottotitoli")
    sub_color = st.color_picker("Colore Testo", "#FFFFFF")
    font_size = st.slider("Grandezza Testo", 30, 70, 48)
    st.info("I sottotitoli avranno un bordo nero per la massima leggibilità.")

# --- MOTORE DI RENDERING CINEMATICO ---
def render_reel(data, video_path, smooth, dz, color, f_size):
    with st.status(f"🎬 Elaborazione: {data['title']}...", expanded=True) as status:
        # Caricamento Clip
        clip = VideoFileClip(video_path).subclipped(data['start'], data['end'])
        w_orig, h_orig = clip.size
        target_w = int(h_orig * (9/16))
        
        # Download modello MediaPipe se mancante
        if not os.path.exists('detector.tflite'):
            import urllib.request
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", "detector.tflite")
        
        # Setup Detector
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path='detector.tflite'),
            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            raw_pos = []
            st.write("🔍 Tracking volti in corso...")
            for t in np.arange(0, clip.duration, 0.1):
                frame = clip.get_frame(t)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                res = detector.detect(mp_img)
                # Calcolo centro X
                val = (res.detections[0].bounding_box.origin_x + res.detections[0].bounding_box.width/2)/w_orig if res.detections else None
                raw_pos.append(val)
            
            # Algoritmo Cinematico (Exponential Smoothing + Dead Zone)
            final_coords = []
            cam_x = 0.5
            for p in raw_pos:
                if p is not None:
                    dist = p - cam_x
                    if abs(dist) > dz:
                        target_move = p - (np.sign(dist) * dz)
                        cam_x += (target_move - cam_x) * smooth
                final_coords.append(cam_x)

            def camera_op(get_frame, t):
                idx = min(int(t / 0.1), len(final_coords) - 1)
                cx = int(final_coords[idx] * w_orig)
                x1 = max(0, min(cx - (target_w // 2), w_orig - target_w))
                return get_frame(t)[:, int(x1):int(x1+target_w)]

            # Trasformazione e VFX
            tracked = clip.transform(camera_op).with_effects([FadeIn(0.5), FadeOut(1.0)])
            
            # Sottotitoli (Wrapped Caption)
            try:
                txt = (TextClip(text=data['title'].upper(), font_size=f_size, color=color, 
                               stroke_color='black', stroke_width=2, method='caption', 
                               font="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                               size=(int(target_w*0.8), None))
                       .with_duration(clip.duration).with_position(('center', 180)))
                final_video = CompositeVideoClip([tracked, txt])
            except:
                final_video = tracked
            
            output_filename = f"C3_REEL_{int(time.time())}.mp4"
            final_video.write_videofile(output_filename, codec="libx264", audio_codec="aac", fps=24, logger=None)
            return output_filename

# --- INTERFACCIA UTENTE ---
st.title("🚀 C3 Reach: Pro AI Reel Factory")
st.markdown("Incolla il link di **Google Drive** della predicazione per generare 10 proposte di Reel cinematografici.")

drive_url = st.text_input("Link Google Drive (Accesso pubblico necessario)", placeholder="https://drive.google.com/file/d/...")

if drive_url and API_KEY:
    if st.button("📥 Importa Video e Analizza con Gemini"):
        try:
            with st.spinner("Scaricamento video da Drive..."):
                # Estrazione ID Drive
                if 'id=' in drive_url:
                    id_drive = drive_url.split('id=')[-1]
                else:
                    id_drive = drive_url.split('/')[-2]
                
                if os.path.exists("input.mp4"): os.remove("input.mp4")
                gdown.download(id=id_drive, output="input.mp4", quiet=False)
                
            if os.path.exists("input.mp4"):
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                with st.status("🧠 Gemini sta analizzando il messaggio...") as status:
                    v_ai = genai.upload_file("input.mp4")
                    # Attesa che il file sia pronto sui server Google
                    while v_ai.state.name == "PROCESSING":
                        time.sleep(3)
                        v_ai = genai.get_file(v_ai.name)
                    
                    if v_ai.state.name == "ACTIVE":
                        prompt = "Analizza il video. Trova 10 momenti potenti (30-50s) con senso logico compiuto. Rispondi SOLO JSON: [{'start': secondi, 'end': secondi, 'title': 'Titolo Breve'}]"
                        response = model.generate_content([v_ai, prompt])
                        # Pulizia risposta JSON
                        json_clean = response.text.strip().replace('```json', '').replace('```', '')
                        st.session_state.clips = json.loads(json_clean)
                        st.success("Analisi completata!")
                    else:
                        st.error("Errore nel caricamento del file su Gemini.")
        except Exception as e:
            st.error(f"Errore durante l'importazione: {e}")

    # --- VISUALIZZAZIONE GRIGLIA 10 PROPOSTE ---
    if 'clips' in st.session_state:
        st.divider()
        st.header("🎞️ Seleziona i Reel da produrre")
        grid = st.columns(2)
        
        for i, clip in enumerate(st.session_state.clips):
            with grid[i % 2]:
                with st.container(border=True):
                    st.subheader(f"Opzione {i+1}: {clip['title']}")
                    st.info(f"⏱ Segmento: {clip['start']}s - {clip['end']}s")
                    
                    if st.button(f"⚡ Renderizza Reel {i+1}", key=f"r_{i}"):
                        file_out = render_reel(clip, "input.mp4", inerzia, dead_zone, sub_color, font_size)
                        if file_out:
                            with open(file_out, "rb") as f:
                                st.download_button(f"📥 Scarica Reel {i+1}", f, file_name=file_out, mime="video/mp4")
