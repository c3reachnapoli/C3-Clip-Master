import streamlit as st
import gdown, os, json, time, base64, requests
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="C3 Reach Napoli - Direct Engine", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Inserisci API Key", type="password")

# --- SIDEBAR REGIA ---
with st.sidebar:
    st.header("⚙️ Parametri")
    inerzia = st.slider("Fluidità", 0.01, 0.15, 0.06)
    dead_zone = st.slider("Stabilità", 0.05, 0.25, 0.12)
    sub_color = st.color_picker("Colore Testo", "#FFFFFF")
    font_size = st.slider("Grandezza", 30, 70, 45)

# --- FUNZIONE CHIAMATA DIRETTA API (No SDK) ---
def call_gemini_direct(audio_path, key):
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Endpoint STABILE v1 (Niente Beta)
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analizza l'audio e trova 10 momenti carismatici (30-50s). Rispondi SOLO JSON: [{'start': s, 'end': s, 'title': 'T'}]"},
                {"inline_data": {"mime_type": "audio/mp3", "data": audio_data}}
            ]
        }],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        res_json = response.json()
        text_content = res_json['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_content)
    else:
        raise Exception(f"Errore Google API {response.status_code}: {response.text}")

# --- MOTORE RENDERING ---
def render_reel(data, video_path, smooth, dz, color, f_size):
    with st.status(f"🎬 Creazione: {data['title']}...") as status:
        clip = VideoFileClip(video_path).subclipped(data['start'], data['end'])
        w_orig, h_orig = clip.size
        target_w = int(h_orig * (9/16))
        
        # Download MediaPipe Detector
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
st.title("🎬 C3 Reach: Pro AI (Direct Pipeline)")

drive_url = st.text_input("Link Google Drive del video")

if drive_url and API_KEY:
    if st.button("🚀 Analizza e Proponi 10 Reel"):
        try:
            with st.spinner("1. Scarico da Drive..."):
                id_drive = drive_url.split('/')[-2] if 'view' in drive_url else drive_url.split('id=')[-1]
                gdown.download(id=id_drive, output="input.mp4", quiet=False)
            
            with st.spinner("2. Estraggo Audio..."):
                v = VideoFileClip("input.mp4")
                v.audio.write_audiofile("temp_audio.mp3", logger=None)
                v.close()
            
            with st.spinner("3. Analisi AI Diretta..."):
                st.session_state.clips = call_gemini_direct("temp_audio.mp3", API_KEY)
                st.success(f"Analisi completata! Trovati {len(st.session_state.clips)} momenti.")

        except Exception as e:
            st.error(f"Errore tecnico: {e}")

    if 'clips' in st.session_state:
        grid = st.columns(2)
        for i, clip in enumerate(st.session_state.clips):
            with grid[i % 2]:
                with st.container(border=True):
                    st.subheader(f"Opzione {i+1}: {clip['title']}")
                    if st.button(f"⚡ Renderizza Reel {i+1}", key=f"btn_{i}"):
                        f_out = render_reel(clip, "input.mp4", inerzia, dead_zone, sub_color, font_size)
                        with open(f_out, "rb") as f:
                            st.download_button("📥 Scarica", f, file_name=f_out)
