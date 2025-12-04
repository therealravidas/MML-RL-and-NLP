# app.py
import streamlit as st
import tempfile
import os
import time
from pathlib import Path
import uuid
import shutil

# ----------------------------------------
# 🔒 HIDDEN API KEY (Only here in backend)
# ----------------------------------------
ASSEMBLYAI_API_KEY = "f4eabc3180944d7f9f36134b537a0c7c"

# Import functions from your modules
import assemblyai_transcript as aai
from assemble_isl_ffmpeg import assemble_text_to_video  


# Force-set the API key inside the module
aai.API_KEY = ASSEMBLYAI_API_KEY


st.set_page_config(page_title="SLAPT - Sign Language Announcement for Public Transport", layout="wide")

st.title("SLAPT - Sign Language Announcement for Public Transport")

st.sidebar.header("Configuration")
st.sidebar.markdown("These are backend options for ISL rendering.")

clips_dir = st.sidebar.text_input("ISL clips directory", value="/home/BTECH_7TH_SEM/Desktop/isl_clips")
out_width = st.sidebar.number_input("Output width", value=1280)
out_height = st.sidebar.number_input("Output height", value=720)
fps = st.sidebar.number_input("FPS", value=30)
pause_clip = st.sidebar.text_input("Pause clip filename", value="pause.mp4")
inter_clip_pause = st.sidebar.number_input("Inter-clip pause (s)", value=0.12, format="%.3f")
font_path = st.sidebar.text_input("Font path (optional)", value="")

st.markdown("## Upload audio (wav/mp3/m4a) or provide an audio URL")

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
with col2:
    audio_url = st.text_input("Or paste audio URL")

st.markdown("### OR paste transcript manually")
manual_transcript = st.text_area("Manual transcript", height=140)

run_button = st.button("Create ISL video")


# -------------------------
# Utility: save uploaded file
# -------------------------
def save_file(file, dest):
    with open(dest, "wb") as f:
        f.write(file.getbuffer())


# -----------------------------
# Main processing
# -----------------------------
if run_button:
    if not audio_file and not audio_url and not manual_transcript:
        st.error("Upload audio, paste URL, or enter transcript.")
        st.stop()

    # -------------------------
    # STEP 1: Get Transcript
    # -------------------------
    if manual_transcript.strip():
        transcript_text = manual_transcript.strip()

    else:
        tmp = Path(tempfile.mkdtemp(prefix="aai_"))

        try:
            if audio_file:
                local_audio = tmp / f"{uuid.uuid4().hex}_{audio_file.name}"
                save_file(audio_file, local_audio)
                st.info(f"Audio saved: {local_audio}")

                with st.spinner("Uploading Audio..."):
                    upload_url = aai.upload_file(str(local_audio))
                    transcript_id = aai.start_transcription(upload_url)

            else:
                with st.spinner("Requesting transcription..."):
                    transcript_id = aai.start_transcription(audio_url)

            # Poll for completion
            import requests
            endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            headers = {"authorization": aai.API_KEY}

            prog = st.progress(0)
            status_text = st.empty()

            i = 0
            while True:
                i += 1
                r = requests.get(endpoint, headers=headers)
                r.raise_for_status()
                resp = r.json()
                status = resp.get("status")

                if status == "completed":
                    transcript_text = resp.get("text", "")
                    break
                if status == "error":
                    st.error("Transcription failed: " + resp.get("error", "unknown"))
                    st.stop()

                prog.progress((i % 20) * 5)
                status_text.text(f"Status: {status}")
                time.sleep(3)

        finally:
            pass

    # Display transcript
    st.subheader("Transcript")
    st.write(transcript_text)


    # -------------------------
    # STEP 2: Assemble ISL Video
    # -------------------------
    st.subheader("Assembling ISL video...")

    if not Path(clips_dir).exists():
        st.error(f"Clips directory not found: {clips_dir}")
        st.stop()

    out_tmp = Path(tempfile.mkdtemp(prefix="isl_out_"))
    out_mp4 = out_tmp / f"isl_{int(time.time())}.mp4"
    workdir = out_tmp / "workdir"

    with st.spinner("Building final ISL MP4 using FFmpeg..."):
        try:
            segs, wd = assemble_text_to_video(
                text=transcript_text,
                clips_dir=clips_dir,
                out_mp4=str(out_mp4),
                workdir=str(workdir),
                width=int(out_width),
                height=int(out_height),
                fps=int(fps),
                pause_clip_name=pause_clip,
                inter_pause=float(inter_clip_pause),
                fallback_spell=True,
                font_path=font_path or None
            )
        except Exception as e:
            st.error(f"ISL assembly failed: {e}")
            st.stop()

    st.success("ISL video successfully created!")
    st.video(str(out_mp4))

    with open(out_mp4, "rb") as f:
        st.download_button("Download ISL Video", f, "isl_output.mp4", "video/mp4")

    st.markdown("### Debug info")
    st.write({"segments": len(segs), "workdir": wd})
