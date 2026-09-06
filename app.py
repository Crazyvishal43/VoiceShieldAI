import io
import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import onnxruntime as ort


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "best_model.onnx"

MODEL_URL = (
    "https://huggingface.co/"
    "ayush2635/"
    "Dhwani-Multilingual-Deepfake-Audio-Detection-Model/"
    "resolve/main/best_model.onnx"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
    }

    .footer {
        text-align: center;
        opacity: 0.6;
        padding: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None


# =========================================================
# MODEL DOWNLOAD / LOAD
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.info(
            "Downloading AI model for the first time..."
        )

        try:

            import urllib.request

            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )

        except Exception as e:

            st.error(
                f"AI model download failed: {e}"
            )

            return None

    try:

        return ort.InferenceSession(
            MODEL_PATH
        )

    except Exception as e:

        st.error(
            f"AI model loading failed: {e}"
        )

        return None


session = load_model()


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_audio_to_wav(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".wav"):

        return uploaded_file.read()

    input_bytes = uploaded_file.read()

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(
            temp_dir,
            "input_audio"
        )

        output_path = os.path.join(
            temp_dir,
            "output.wav"
        )

        with open(input_path, "wb") as f:

            f.write(input_bytes)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        with open(output_path, "rb") as f:

            return f.read()


# =========================================================
# PREPARE AUDIO
# =========================================================

def prepare_audio(audio_bytes):

    audio_buffer = io.BytesIO(
        audio_bytes
    )

    audio, sample_rate = sf.read(
        audio_buffer
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    if sample_rate != 16000:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000
        )

    max_len = 48000

    if len(audio) > max_len:

        audio = audio[:max_len]

    elif len(audio) < max_len:

        audio = np.pad(
            audio,
            (
                0,
                max_len - len(audio)
            ),
            mode="constant"
        )

    audio = (
        audio - np.mean(audio)
    ) / np.sqrt(
        np.var(audio) + 1e-5
    )

    return audio.astype(
        np.float32
    )


# =========================================================
# AI ANALYSIS
# =========================================================

def analyze_audio(audio_bytes):

    if session is None:

        return None

    audio = prepare_audio(
        audio_bytes
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    logits = session.run(
        None,
        {
            input_name:
            audio.reshape(
                1,
                48000
            )
        }
    )[0]

    probabilities = np.exp(
        logits
    )

    probabilities = (
        probabilities
        /
        np.sum(
            probabilities,
            axis=1,
            keepdims=True
        )
    )

    fake_probability = float(
        probabilities[0][1]
    )

    risk_score = (
        fake_probability * 100
    )

    return risk_score


# =========================================================
# RISK LEVEL
# =========================================================

def get_level(risk_score):

    if risk_score < 30:

        return "LOW"

    elif risk_score < 70:

        return "MEDIUM"

    return "HIGH"


# =========================================================
# SECURITY RECOMMENDATION
# =========================================================

def show_security_action(
    risk_score
):

    level = get_level(
        risk_score
    )

    st.markdown(
        "### 🛡️ Security Assessment"
    )

    if level == "HIGH":

        st.error(
            "🚨 POTENTIAL VOICE CLONING DETECTED"
        )

        st.markdown(
            """
            **Security Recommendation**

            Do **not** approve sensitive actions
            based on voice alone.

            🔐 Recommended verification:

            - OTP verification
            - Trusted callback
            - Independent identity verification
            - Secondary authentication factor
            """
        )

        st.warning(
            "High risk means the system detected "
            "a strong AI-generated/manipulated "
            "voice signal. It is not absolute proof "
            "of identity fraud."
        )

    elif level == "MEDIUM":

        st.warning(
            "⚠️ SUSPICIOUS VOICE SIGNAL"
        )

        st.markdown(
            """
            **Security Recommendation**

            Request secondary verification
            before continuing.

            Recommended:

            - OTP
            - Callback
            - Additional authentication
            """
        )

    else:

        st.success(
            "🟢 NO STRONG AI-CLONING SIGNAL"
        )

        st.info(
            "The voice currently appears low risk. "
            "However, voice-only authentication "
            "should not be treated as absolute proof "
            "of identity."
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

def show_explanation(
    risk_score
):

    level = get_level(
        risk_score
    )

    st.markdown(
        "### 🔍 Explainable AI"
    )

    if level == "HIGH":

        st.write(
            "The acoustic characteristics of the "
            "submitted audio produced a strong "
            "synthetic/manipulated voice signal."
        )

    elif level == "MEDIUM":

        st.write(
            "The audio contains characteristics "
            "that may be consistent with synthetic "
            "or manipulated speech."
        )

    else:

        st.write(
            "No strong AI-cloning signal was "
            "detected in the analyzed audio."
        )

    st.progress(
        min(
            max(
                int(risk_score),
                0
            ),
            100
        )
    )

    st.caption(
        "Higher score = stronger detected "
        "impersonation signal."
    )


# =========================================================
# HISTORY
# =========================================================

def add_history(
    file_name,
    risk_score
):

    st.session_state.history.append(
        {
            "time":
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),

            "file":
            file_name,

            "risk":
            risk_score,

            "level":
            get_level(
                risk_score
            )
        }
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ VoiceShield AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Real-Time Voice Cloning & '
    'Impersonation Detection'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# STATUS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="status-box">
        🟢<br>
        <b>SYSTEM STATUS</b><br>
        ONLINE
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="status-box">
        🤖<br>
        <b>AI ENGINE</b><br>
        READY
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="status-box">
        🔐<br>
        <b>PRIVACY MODE</b><br>
        LOCAL ANALYSIS
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SECURITY OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 Security Overview'
    '</div>',
    unsafe_allow_html=True
)

total_scans = len(
    st.session_state.history
)

high_risk = sum(
    1
    for x in st.session_state.history
    if x["level"] == "HIGH"
)

medium_risk = sum(
    1
    for x in st.session_state.history
    if x["level"] == "MEDIUM"
)

low_risk = sum(
    1
    for x in st.session_state.history
    if x["level"] == "LOW"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Total Scans",
    total_scans
)

m2.metric(
    "🔴 High Risk",
    high_risk
)

m3.metric(
    "🟠 Medium Risk",
    medium_risk
)

m4.metric(
    "🟢 Low Risk",
    low_risk
)


# =========================================================
# MICROPHONE DETECTION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🎤 Speak & Detect'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Speak into your microphone. "
    "VoiceShield AI will analyze the recorded "
    "audio for potential AI-cloning signals."
)

recorded_audio = st.audio_input(
    "🎙️ Record your voice"
)

if recorded_audio:

    st.audio(
        recorded_audio
    )

    if st.button(
        "🤖 Analyze Recorded Voice",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing your recorded voice..."
        ):

            try:

                audio_bytes = (
                    recorded_audio.getvalue()
                )

                risk_score = analyze_audio(
                    audio_bytes
                )

                if risk_score is None:

                    st.error(
                        "AI model is not available."
                    )

                else:

                    add_history(
                        "Microphone Recording",
                        risk_score
                    )

                    st.markdown(
                        "### 📈 Microphone Analysis Result"
                    )

                    r1, r2 = st.columns(2)

                    with r1:

                        st.metric(
                            "Impersonation Risk Score",
                            f"{risk_score:.2f} / 100"
                        )

                    with r2:

                        st.metric(
                            "Risk Level",
                            get_level(
                                risk_score
                            )
                        )

                    st.progress(
                        min(
                            max(
                                int(risk_score),
                                0
                            ),
                            100
                        )
                    )

                    show_security_action(
                        risk_score
                    )

                    show_explanation(
                        risk_score
                    )

            except Exception as e:

                st.error(
                    f"Microphone analysis failed: {e}"
                )


# =========================================================
# SINGLE VOICE DETECTION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🎙️ Voice Detection'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a voice recording",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "m4a",
        "ogg",
        "flac"
    ],
    key="single_upload"
)

if uploaded_file:

    st.audio(
        uploaded_file
    )

    if st.button(
        "🔍 Analyze Voice",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing the voice..."
        ):

            try:

                audio_bytes = (
                    convert_audio_to_wav(
                        uploaded_file
                    )
                )

                risk_score = analyze_audio(
                    audio_bytes
                )

                if risk_score is None:

                    st.error(
                        "AI model not found."
                    )

                else:

                    add_history(
                        uploaded_file.name,
                        risk_score
                    )

                    st.markdown(
                        "### 📈 Analysis Result"
                    )

                    r1, r2 = st.columns(2)

                    with r1:

                        st.metric(
                            "Impersonation Risk Score",
                            f"{risk_score:.2f} / 100"
                        )

                    with r2:

                        st.metric(
                            "Risk Level",
                            get_level(
                                risk_score
                            )
                        )

                    st.progress(
                        min(
                            max(
                                int(risk_score),
                                0
                            ),
                            100
                        )
                    )

                    show_security_action(
                        risk_score
                    )

                    show_explanation(
                        risk_score
                    )

            except Exception as e:

                st.error(
                    f"Audio analysis failed: {e}"
                )


# =========================================================
# HUMAN VS AI COMPARISON
# =========================================================

st.markdown(
    '<div class="section-title">'
    '⚖️ Human vs AI Voice Comparison'
    '</div>',
    unsafe_allow_html=True
)

col_a, col_b = st.columns(2)

with col_a:

    sample_a = st.file_uploader(
        "Sample A — Human / Genuine Voice",
        type=[
            "wav",
            "mp3",
            "mpeg",
            "m4a",
            "ogg",
            "flac"
        ],
        key="sample_a"
    )

with col_b:

    sample_b = st.file_uploader(
        "Sample B — AI / Synthetic Voice",
        type=[
            "wav",
            "mp3",
            "mpeg",
            "m4a",
            "ogg",
            "flac"
        ],
        key="sample_b"
    )

if sample_a and sample_b:

    if st.button(
        "⚔️ Compare Both Voices",
        use_container_width=True
    ):

        with st.spinner(
            "Comparing both voice samples..."
        ):

            try:

                audio_a = (
                    convert_audio_to_wav(
                        sample_a
                    )
                )

                audio_b = (
                    convert_audio_to_wav(
                        sample_b
                    )
                )

                risk_a = analyze_audio(
                    audio_a
                )

                risk_b = analyze_audio(
                    audio_b
                )

                st.session_state.comparison_results = {
                    "A": {
                        "risk": risk_a,
                        "level":
                        get_level(
                            risk_a
                        )
                    },

                    "B": {
                        "risk": risk_b,
                        "level":
                        get_level(
                            risk_b
                        )
                    }
                }

            except Exception as e:

                st.error(
                    f"Comparison failed: {e}"
                )


# =========================================================
# COMPARISON RESULT
# =========================================================

comparison = (
    st.session_state.comparison_results
)

if comparison:

    st.markdown(
        "### 📊 Comparison Result"
    )

    a_risk = comparison["A"]["risk"]
    b_risk = comparison["B"]["risk"]

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "👤 Sample A Risk",
            f"{a_risk:.2f}%"
        )

        st.progress(
            min(
                max(
                    int(a_risk),
                    0
                ),
                100
            )
        )

        st.caption(
            f"Risk Level: "
            f"{comparison['A']['level']}"
        )

    with c2:

        st.metric(
            "🤖 Sample B Risk",
            f"{b_risk:.2f}%"
        )

        st.progress(
            min(
                max(
                    int(b_risk),
                    0
                ),
                100
            )
        )

        st.caption(
            f"Risk Level: "
            f"{comparison['B']['level']}"
        )

    risk_difference = abs(
        a_risk - b_risk
    )

    st.metric(
        "Risk Difference",
        f"{risk_difference:.2f}%"
    )

    if (
        comparison["A"]["level"]
        !=
        comparison["B"]["level"]
    ):

        st.success(
            "✅ The system detected a different "
            "risk profile between the two samples."
        )

    else:

        st.warning(
            "⚠️ Both samples received the same "
            "risk category. Additional testing "
            "is recommended."
        )


# =========================================================
# DETECTION HISTORY
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🕘 Detection History'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    for item in reversed(
        st.session_state.history
    ):

        st.write(
            f"**{item['time']}** | "
            f"{item['file']} | "
            f"{item['risk']:.2f}% | "
            f"{item['level']}"
        )

else:

    st.info(
        "No voice scans performed yet."
    )


# =========================================================
# ABOUT
# =========================================================

st.markdown(
    '<div class="section-title">'
    'ℹ️ About VoiceShield AI'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    VoiceShield AI is a privacy-first voice
    security prototype designed to detect
    potential AI-generated or manipulated speech.

    The system analyzes audio and generates an
    impersonation risk score to help users decide
    when additional identity verification is required.

    Voice-only detection should be treated as a
    security signal, not as absolute proof of identity.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    🛡️ VoiceShield AI<br>
    AI-Powered Voice Impersonation Protection<br>
    SIH 2026 Prototype
    </div>
    """,
    unsafe_allow_html=True
)