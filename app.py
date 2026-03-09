import streamlit as st
import gdown, os, json, time
import numpy as np
from google import genai
import mediapipe as mp
import cv2
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import FadeIn, FadeOut

# Forza l'uso di OpenCV in modalità headless per evitare errori di libreria
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

st.set_page_config(page_title="C3 Reach Napoli", layout="wide")

# --- SECRETS ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Inserisci API Key", type="password")

# --- MOTORE RENDERING (VERSIONE SENZA DIPENDENZE ESTERNE) ---
def render_reel(data, video_path, smooth, dz, color, f_size):
    with st.status(f"🎬 Elaborazione: {data['title']}...") as status:
        clip = VideoFileClip(video_path).subclipped(data['start'], data['end'])
        w_orig, h_orig = clip.size
        target_w = int(h_orig * (9/16))
        
        # Download detector locale se manca
        if not os.path.exists('detector.tflite'):
            import urllib.request
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", "detector.tflite")
        
        base_options = mp.tasks.BaseOptions(model_asset_path='detector.tflite')
        options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE)
        
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            raw_pos = []
            for t in np.arange(0, clip.duration, 0.2): # Step più largo per velocità
                frame = clip.get_frame(t)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                res = detector.detect(mp_img)
                val = (res.detections[0].bounding_box.origin_x + res.detections[0].bounding_box.width/2)/w_orig if res.detections else 0.5
                raw_pos.append(val)
            
            # Smoothing camera
            final_coords = []
            cam_x = 0.5
            for p in raw_pos:
                dist = p - cam_x
                if abs(dist) > dz: cam_x += (p - (np.sign(dist)*dz) - cam_x) * smooth
                final_coords.append(cam_x)

            def camera_op(get_frame, t):
                idx = min(int(t / 0.2), len(final_coords) - 1)
                cx = int(final_coords[idx] * w_orig)
                x1 = max(0, min(cx - (target_w // 2), w_orig - target_w))
                return get_frame(t)[:, int(x1):int(x1+target_w)]

            # Rendering finale
            final_v = clip.transform(camera_op).with_effects([FadeIn(0.5), FadeOut(0.5)])
            out_name = f"REEL_{int(time.time())}.mp4"
            final_v.write_videofile(out_name, codec="libx264", audio_codec="aac", fps=24, logger=None)
            return out_name

# --- INTERFACCIA ---
st.title("🚀 C3 Reach: Pro AI (Safe Mode)")
drive_url = st.text_input("Link Google Drive del video")

if drive_url and API_KEY:
    if st.button("🚀 Analizza"):
        try:
            client = genai.Client(api_key=API_KEY)
            # Logica download e analisi (come prima, ma con gestione errori)
            id_drive = drive_url.split('/')[-2] if 'view' in drive_url else drive_url.split('id=')[-1]
            gdown.download(id=id_drive, output="input.mp4", quiet=False)
            
            v = VideoFileClip("input.mp4")
            v.audio.write_audiofile("audio.mp3", logger=None)
            v.close()
            
            audio_upload = client.files.upload(path="audio.mp3")
            while audio_upload.state.name == "PROCESSING":
                time.sleep(2)
                audio_upload = client.files.get(name=audio_upload.name)
            
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[audio_upload, "Trova 10 momenti (30-50s). Rispondi SOLO JSON: [{'start': s, 'end': s, 'title': 't'}]"]
            )
            
            raw = resp.text
            st.session_state.clips = json.loads(raw[raw.find("["):raw.rfind("]")+1])
            st.success("Momenti trovati!")
        except Exception as e:
            st.error(f"Errore: {e}")

if 'clips' in st.session_state:
    for i, clip in enumerate(st.session_state.clips):
        if st.button(f"⚡ Crea Reel {i+1}: {clip['title']}"):
            f_out = render_reel(clip, "input.mp4", 0.06, 0.12, "#FFFFFF", 40)
            with open(f_out, "rb") as f:
                st.download_button("📥 Scarica", f, file_name=f_out)
