# --- FUNZIONE CHIAMATA DIRETTA API CORRETTA ---
def call_gemini_direct(audio_path, key):
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Endpoint v1 (Stabile)
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}"
    
    # Payload semplificato e corretto per v1
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analizza l'audio e trova i 10 momenti più potenti (30-50s ciascuno). Rispondi ESCLUSIVAMENTE con una lista JSON valida, senza testo aggiuntivo, usando questo formato: [{'start': secondi, 'end': secondi, 'title': 'Titolo'}]"},
                {"inline_data": {"mime_type": "audio/mp3", "data": audio_data}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        res_json = response.json()
        raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
        
        # Pulizia manuale del JSON (rimuove eventuali ```json o testo extra)
        try:
            start_idx = raw_text.find("[")
            end_idx = raw_text.rfind("]") + 1
            json_data = json.loads(raw_text[start_idx:end_idx])
            return json_data
        except Exception:
            st.error(f"Errore nel formato ricevuto dall'AI: {raw_text[:200]}")
            return None
    else:
        raise Exception(f"Errore Google API {response.status_code}: {response.text}")

# --- AGGIORNAMENTO NELL'INTERFACCIA ---
# (Assicurati che nel corpo principale dell'app ci sia questo controllo)
if 'clips' in st.session_state and st.session_state.clips:
    st.divider()
    grid = st.columns(2)
    for i, clip in enumerate(st.session_state.clips):
        with grid[i % 2]:
            with st.container(border=True):
                st.subheader(f"Opzione {i+1}: {clip.get('title', 'Senza Titolo')}")
                st.write(f"⏱ {clip.get('start')}s - {clip.get('end')}s")
                if st.button(f"⚡ Renderizza Reel {i+1}", key=f"btn_{i}"):
                    f_out = render_reel(clip, "input.mp4", inerzia, dead_zone, sub_color, font_size)
                    with open(f_out, "rb") as f:
                        st.download_button("📥 Scarica", f, file_name=f_out)
