import streamlit as st
import os
import pandas as pd
import cv2
import time

from modules.text_emotion import predict_text_emotion
from modules.audio_emotion import predict_audio_emotion
from modules.face_emotion import detect_face_emotion
from modules.fusion import fuse_emotions
from modules.session_store import save_emotion, load_history, clear_history

from audiorecorder import audiorecorder
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="HUMORA - Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("HUMORA - Multimodal Emotion Recognition System")
st.write("Emotion detection using Text, Audio and Face modalities.")

# ---------------------------------------------------
# TIMER STATE
# ---------------------------------------------------
if "last_auto_run" not in st.session_state:
    st.session_state.last_auto_run = 0

# ---------------------------------------------------
# TEXT INPUT
# ---------------------------------------------------
st.subheader("📝 Text Emotion Detection")
user_text = st.text_input("Enter a sentence")

# ---------------------------------------------------
# AUDIO INPUT
# ---------------------------------------------------
st.subheader("🎤 Audio Emotion Detection")

audio_mode = st.radio(
    "Choose Audio Input Method:",
    ["Upload Audio File", "Record from Microphone"]
)

audio_path = None

if audio_mode == "Upload Audio File":
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        audio_path = "temp.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

else:
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        audio_path = "temp.wav"
        audio.export(audio_path, format="wav")
        st.audio(audio_path)

# ---------------------------------------------------
# FACE EMOTION
# ---------------------------------------------------
st.subheader("📷 Live Face Emotion Detection")


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.latest_emotion = None
        self.latest_confidence = 0.0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (320, 240))

        self.frame_count += 1

        if self.frame_count % 30 == 0:
            emotion, conf = detect_face_emotion(small)

            if emotion:
                self.latest_emotion = emotion
                self.latest_confidence = conf

        if self.latest_emotion:
            cv2.putText(
                img,
                f"{self.latest_emotion} ({self.latest_confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        return frame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="emotion-webcam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 15},
        "audio": False,
    },
)

# AUTO REFRESH ONLY WHEN CAMERA RUNNING
if webrtc_ctx and webrtc_ctx.state.playing:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=4000, key="auto_emotion_refresh")

# STORE LIVE FACE RESULT
if webrtc_ctx and webrtc_ctx.video_processor:

    vp = webrtc_ctx.video_processor

    if vp.latest_emotion is not None:
        st.session_state["live_face_result"] = {
            "label": vp.latest_emotion,
            "score": float(vp.latest_confidence),
        }

        st.success(
            f"Live Face Emotion: {vp.latest_emotion} "
            f"(Confidence {vp.latest_confidence:.2f})"
        )

# ---------------------------------------------------
# AUTO EMOTION SAVE
# ---------------------------------------------------
face_result = st.session_state.get("live_face_result")
current_time = time.time()

if (
    face_result
    and webrtc_ctx
    and webrtc_ctx.state.playing
    and current_time - st.session_state.last_auto_run > 4
):

    st.session_state.last_auto_run = current_time

    final_emotion, _ = fuse_emotions(None, None, face_result)

    if final_emotion:
        save_emotion(final_emotion)

# ---------------------------------------------------
# MANUAL ANALYZE (TEXT + AUDIO + FACE FUSION)
# ---------------------------------------------------
st.markdown("---")
st.subheader("🔎 Manual Emotion Analysis")

if st.button("Analyze Emotion"):

    text_result = None
    audio_result = None
    face_result = st.session_state.get("live_face_result")

    # -------- TEXT --------
    if user_text.strip():
        text_result = predict_text_emotion(user_text)

        st.success(
            f"Text Emotion: {text_result['label']} "
            f"(Confidence {text_result['score']:.2f})"
        )

    # -------- AUDIO --------
    if audio_path:
        audio_result = predict_audio_emotion(audio_path)

        st.success(
            f"Audio Emotion: {audio_result['label']} "
            f"(Confidence {audio_result['score']:.2f})"
        )

        if os.path.exists(audio_path):
            os.remove(audio_path)

    # -------- FUSION --------
    final_emotion, fusion_scores = fuse_emotions(
        text_result,
        audio_result,
        face_result,
    )

    if final_emotion:

        save_emotion(final_emotion)

        st.markdown("### 🧠 Final Detected Emotion")
        st.success(final_emotion)

        df = pd.DataFrame(
            fusion_scores.items(),
            columns=["Emotion", "Score"]
        ).set_index("Emotion")

        st.bar_chart(df)


# ---------------------------------------------------
# SESSION CONTROLS
# ---------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Session Controls")

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if st.button("🗑 Clear Emotion Session"):
    st.session_state.confirm_clear = True

if st.session_state.confirm_clear:

    st.warning("Are you sure you want to delete all recorded emotions?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, Delete", key="confirm_delete"):
            clear_history()
            st.success("Emotion session cleared successfully.")
            st.session_state.clear()
            st.rerun()

    with col2:
        if st.button("❌ Cancel", key="cancel_delete"):
            st.session_state.confirm_clear = False

# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------
st.markdown("---")
st.header("📊 Emotion Analytics Dashboard")

history = load_history()

if not history.empty:

    history["timestamp"] = pd.to_datetime(history["timestamp"])

    st.subheader("Emotion Distribution")
    st.bar_chart(history["emotion"].value_counts())

    st.subheader("Emotion Timeline")
    timeline = history.set_index("timestamp")

    st.line_chart(
        timeline["emotion"].astype("category").cat.codes
    )

    dominant = history["emotion"].value_counts().idxmax()
    st.success(f"Dominant Emotion in Session: {dominant}")

else:
    st.info("No emotion data recorded yet.")

st.markdown("---")
st.caption("HUMORA © Multimodal Emotion Recognition using AI")
