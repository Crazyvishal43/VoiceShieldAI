import io
import os
import subprocess
import urllib.request

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_PATH = "best_model.onnx"

MODEL_URL = (
    "https://huggingface.co/ayush2635/"
    "Dhwani-Multilingual-Deepfake-Audio-Detection-Model/"
    "resolve/main/best_model.onnx"
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "recorded_result" not in st.session_state:
    st.session_state.recorded_result = None

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if "incoming_result" not in st.session_state:
    st.session_state.incoming_result = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        text-align: center;
    }

    .risk-box {
        padding: 25px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }

    .high-risk {
        background-color: #fff0f0;
        border-color: #ff6b6b;
    }

    .medium-risk {
        background-color: #fff8e6;
        border-color: #f0ad4e;
    }

    .low-risk {
        background-color: #effbf2;
        border-color: #55a868;
    }

    .hybrid-box {
        padding: 22px;
        border-radius: 14px;
        border: 2px dashed #777777;
        margin-top: 15px;
        background-color: #f8f8f8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL DOWNLOAD
# =========================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return True

    st.info(
        "Downloading AI detection model for first-time setup..."
    )

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        return True

    except Exception as e:

        st.error(
            f"Unable to download AI model: {e}"
        )

        return False


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    if not download_model():
        return None

    try:

        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        return session

    except Exception as e:

        st.error(
            f"AI model loading failed: {e}"
        )

        return None


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_audio(audio_bytes, extension):

    input_file = "temp_input_audio"
    output_file = "temp_converted.wav"

    try:

        with open(
            f"{input_file}.{extension}",
            "wb"
        ) as f:

            f.write(audio_bytes)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            f"{input_file}.{extension}",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_file
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            return None

        with open(
            output_file,
            "rb"
        ) as f:

            converted = f.read()

        try:
            os.remove(f"{input_file}.{extension}")
        except:
            pass

        try:
            os.remove(output_file)
        except:
            pass

        return converted

    except Exception:
        return None


# =========================================================
# PREPARE AUDIO
# =========================================================

def prepare_audio(audio_bytes):

    try:

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

            sample_rate = 16000

        max_len = 48000

        if len(audio) > max_len:

            audio = audio[:max_len]

        else:

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

        audio = audio.astype(
            np.float32
        )

        return audio

    except Exception as e:

        st.error(
            f"Audio processing failed: {e}"
        )

        return None


# =========================================================
# AI ANALYSIS
# =========================================================

def analyze_audio(audio_bytes):

    session = get_model()

    if session is None:
        return None

    audio = prepare_audio(
        audio_bytes
    )

    if audio is None:
        return None

    try:

        audio_input = audio.reshape(
            1,
            48000
        )

        input_name = (
            session.get_inputs()[0].name
        )

        logits = session.run(
            None,
            {
                input_name: audio_input
            }
        )[0]

        probabilities = (
            np.exp(logits)
            /
            np.sum(
                np.exp(logits),
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

        if risk_score >= 70:

            level = "HIGH"

        elif risk_score >= 30:

            level = "MEDIUM"

        else:

            level = "LOW"

        return {
            "risk": risk_score,
            "level": level,
            "fake_probability": fake_probability
        }

    except Exception as e:

        st.error(
            f"AI analysis failed: {e}"
        )

        return None


# =========================================================
# HISTORY
# =========================================================

def save_history(
    source,
    result,
    sample_type="Unknown"
):

    if result is None:
        return

    st.session_state.history.append(
        {
            "source": source,
            "risk": result["risk"],
            "level": result["level"],
            "sample_type": sample_type
        }
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

def show_explainable_ai(
    result,
    sample_type="Unknown"
):

    if result is None:
        return

    risk = result["risk"]
    level = result["level"]

    st.markdown(
        "### 🧠 Explainable AI"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if risk >= 70:

            st.error(
                "Synthetic speech signal"
            )

        elif risk >= 30:

            st.warning(
                "Possible synthetic signal"
            )

        else:

            st.success(
                "No strong synthetic signal"
            )

    with col2:

        st.info(
            "Acoustic pattern assessment"
        )

    with col3:

        if level == "HIGH":

            st.error(
                "Low authenticity confidence"
            )

        elif level == "MEDIUM":

            st.warning(
                "Moderate authenticity confidence"
            )

        else:

            st.success(
                "Higher authenticity confidence"
            )

    with col4:

        if level == "HIGH":

            st.error(
                "Security response required"
            )

        elif level == "MEDIUM":

            st.warning(
                "Additional verification recommended"
            )

        else:

            st.success(
                "No strong AI-cloning signal"
            )

    if sample_type == "Hybrid":

        st.markdown(
            """
            <div class="hybrid-box">

            <b>🧪 Challenging Hybrid Sample</b>

            <p>
            This sample is identified as a mixed
            human + AI voice scenario. Hybrid and
            unseen voice-cloning samples can be
            difficult for an anti-spoofing model to
            classify reliably.
            </p>

            <p>
            The displayed risk score is the original
            model output. VoiceShield AI does not
            artificially change the model score.
            </p>

            <b>🔐 Recommendation:</b>
            Secondary identity verification is recommended.

            </div>
            """,
            unsafe_allow_html=True
        )

    elif level == "HIGH":

        st.error(
            "⚠️ Potential voice cloning signal detected."
        )

        st.warning(
            "Do not approve sensitive actions "
            "using voice alone. Require secondary verification."
        )

    elif level == "MEDIUM":

        st.warning(
            "⚠️ Moderate impersonation risk."
        )

        st.info(
            "Additional verification is recommended."
        )

    else:

        st.success(
            "✅ No strong AI-cloning signal detected."
        )

        st.info(
            "Voice-only authentication should still "
            "not be treated as absolute proof of identity."
        )


# =========================================================
# ANALYSIS RESULT
# =========================================================

def show_analysis_result(
    result,
    title="Analysis Result",
    sample_type="Unknown"
):

    if result is None:
        return

    risk = result["risk"]
    level = result["level"]

    st.markdown(
        f"### {title}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Impersonation Risk Score",
            f"{risk:.2f} / 100"
        )

    with col2:

        if level == "HIGH":

            st.error(
                f"🔴 {level} RISK"
            )

        elif level == "MEDIUM":

            st.warning(
                f"🟠 {level} RISK"
            )

        else:

            st.success(
                f"🟢 {level} RISK"
            )

    if risk >= 70:

        st.markdown(
            """
            <div class="risk-box high-risk">

            <h3>🚨 Potential Voice Cloning Detected</h3>

            <p>
            VoiceShield AI detected a strong
            AI-generated / manipulated voice signal.
            </p>

            <b>Security Action:</b>
            Do not approve sensitive actions using
            voice alone.

            </div>
            """,
            unsafe_allow_html=True
        )

    elif risk >= 30:

        st.markdown(
            """
            <div class="risk-box medium-risk">

            <h3>⚠️ Suspicious Voice</h3>

            <p>
            The system detected a moderate
            impersonation risk.
            </p>

            <b>Security Action:</b>
            Request additional verification.

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="risk-box low-risk">

            <h3>✅ Low Risk</h3>

            <p>
            No strong AI-cloning signal was detected.
            </p>

            <b>Security Action:</b>
            Continue normal verification policies.

            </div>
            """,
            unsafe_allow_html=True
        )

    show_explainable_ai(
        result,
        sample_type
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛡️ VoiceShield AI"
)

st.caption(
    "AI-powered voice cloning detection and impersonation fraud prevention system"
)


# =========================================================
# SYSTEM STATUS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.success(
        "🟢 SYSTEM STATUS\n\nONLINE"
    )

with col2:

    st.info(
        "🤖 AI ENGINE\n\nREADY"
    )

with col3:

    st.success(
        "🔐 PRIVACY MODE\n\nLOCAL ANALYSIS"
    )


st.divider()


# =========================================================
# SECURITY OVERVIEW
# =========================================================

st.markdown(
    "## 📊 Security Overview"
)

total_scans = len(
    st.session_state.history
)

high_count = sum(
    1
    for item in st.session_state.history
    if item["level"] == "HIGH"
)

medium_count = sum(
    1
    for item in st.session_state.history
    if item["level"] == "MEDIUM"
)

low_count = sum(
    1
    for item in st.session_state.history
    if item["level"] == "LOW"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Scans",
        total_scans
    )

with c2:
    st.metric(
        "High Risk",
        high_count
    )

with c3:
    st.metric(
        "Medium Risk",
        medium_count
    )

with c4:
    st.metric(
        "Low Risk",
        low_count
    )


st.divider()


# =========================================================
# SPEAK & DETECT
# =========================================================

st.markdown(
    "## 🎤 Speak & Detect"
)

st.write(
    "Record a voice sample directly from your microphone."
)

recorded_audio = st.audio_input(
    "🎙️ Record your voice"
)

if recorded_audio is not None:

    if st.button(
        "🔍 Analyze Recorded Voice"
    ):

        result = analyze_audio(
            recorded_audio.getvalue()
        )

        st.session_state.recorded_result = result

        if result is not None:

            save_history(
                "Microphone",
                result,
                "Human / Recorded"
            )


if st.session_state.recorded_result is not None:

    show_analysis_result(
        st.session_state.recorded_result,
        "🎤 Recorded Voice Result",
        "Human"
    )


st.divider()


# =========================================================
# VOICE DETECTION
# =========================================================

st.markdown(
    "## 🔊 Voice Detection"
)

st.write(
    "Upload an audio file for AI voice-cloning analysis."
)

uploaded_file = st.file_uploader(
    "Upload voice sample",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "ogg",
        "flac",
        "m4a"
    ]
)


# =========================================================
# NEW HYBRID SAMPLE SELECTOR
# =========================================================

sample_type = st.selectbox(
    "🧪 Select sample type for analysis",
    [
        "Unknown",
        "Human",
        "AI-generated",
        "Hybrid"
    ]
)

if sample_type == "Hybrid":

    st.info(
        "🧪 Hybrid mode: use this for a voice containing "
        "mixed human + AI characteristics. The model's "
        "original risk score will NOT be changed."
    )


if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    if st.button(
        "🤖 Analyze Uploaded Voice"
    ):

        file_bytes = uploaded_file.getvalue()

        extension = uploaded_file.name.split(
            "."
        )[-1].lower()

        if extension != "wav":

            converted_audio = convert_audio(
                file_bytes,
                extension
            )

            if converted_audio is None:

                st.error(
                    "Audio conversion failed. "
                    "Please check FFmpeg installation."
                )

                result = None

            else:

                result = analyze_audio(
                    converted_audio
                )

        else:

            result = analyze_audio(
                file_bytes
            )

        st.session_state.upload_result = result

        if result is not None:

            save_history(
                uploaded_file.name,
                result,
                sample_type
            )


if st.session_state.upload_result is not None:

    show_analysis_result(
        st.session_state.upload_result,
        "🔊 Voice Detection Result",
        sample_type
    )


st.divider()


# =========================================================
# INCOMING VOICE SECURITY GATE
# =========================================================

st.markdown(
    "## 📞 Incoming Voice Security Gate"
)

st.write(
    "Simulate a real-world incoming call where "
    "VoiceShield AI checks the caller before a "
    "sensitive action is approved."
)

incoming_file = st.file_uploader(
    "Upload incoming caller voice",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "ogg",
        "flac",
        "m4a"
    ],
    key="incoming_voice"
)

if incoming_file is not None:

    st.audio(
        incoming_file
    )

    if st.button(
        "🚨 Scan Incoming Voice"
    ):

        incoming_bytes = (
            incoming_file.getvalue()
        )

        extension = (
            incoming_file.name
            .split(".")[-1]
            .lower()
        )

        if extension != "wav":

            incoming_converted = convert_audio(
                incoming_bytes,
                extension
            )

            if incoming_converted is None:

                st.error(
                    "Incoming audio conversion failed."
                )

                incoming_result = None

            else:

                incoming_result = analyze_audio(
                    incoming_converted
                )

        else:

            incoming_result = analyze_audio(
                incoming_bytes
            )

        st.session_state.incoming_result = (
            incoming_result
        )

        if incoming_result is not None:

            save_history(
                "Incoming Call",
                incoming_result,
                "Incoming Voice"
            )


if st.session_state.incoming_result is not None:

    incoming_result = (
        st.session_state.incoming_result
    )

    show_analysis_result(
        incoming_result,
        "📞 Incoming Call Analysis",
        "Incoming Voice"
    )

    incoming_risk = incoming_result["risk"]

    st.markdown(
        "### 🔐 Security Decision"
    )

    if incoming_risk >= 70:

        st.error(
            "🛑 BLOCK / VERIFY"
        )

        st.warning(
            "Sensitive Action Automatically Stopped"
        )

        st.info(
            "🔐 Identity Verification Required"
        )

        verification_method = st.selectbox(
            "Choose secondary verification method",
            [
                "OTP Verification",
                "Trusted Callback",
                "Device Authentication",
                "In-App Confirmation",
                "Human Verification"
            ],
            key="incoming_verification"
        )

        if st.button(
            "🔐 Verify Identity"
        ):

            st.success(
                f"Prototype verification initiated using: "
                f"{verification_method}"
            )

            st.info(
                "This is a prototype simulation. "
                "A production system would connect this "
                "step to a real authentication service."
            )

    elif incoming_risk >= 30:

        st.warning(
            "⚠️ REVIEW / VERIFY"
        )

        st.info(
            "Additional identity verification is recommended "
            "before approving sensitive actions."
        )

    else:

        st.success(
            "🟢 LOW RISK — CONTINUE WITH NORMAL SECURITY POLICY"
        )


st.divider()


# =========================================================
# HUMAN VS AI COMPARISON
# =========================================================

st.markdown(
    "## 👤 Human vs 🤖 AI Comparison"
)

st.write(
    "Compare two voice samples using the same AI detection pipeline."
)

sample_a = st.file_uploader(
    "Sample A",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "ogg",
        "flac",
        "m4a"
    ],
    key="sample_a"
)

sample_b = st.file_uploader(
    "Sample B",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "ogg",
        "flac",
        "m4a"
    ],
    key="sample_b"
)

if sample_a is not None:

    st.audio(
        sample_a
    )

if sample_b is not None:

    st.audio(
        sample_b
    )


if st.button(
    "⚖️ Compare Human vs AI"
):

    if sample_a is None or sample_b is None:

        st.warning(
            "Please upload both Sample A and Sample B."
        )

    else:

        def process_uploaded_file(file):

            file_bytes = file.getvalue()

            extension = (
                file.name
                .split(".")[-1]
                .lower()
            )

            if extension != "wav":

                return convert_audio(
                    file_bytes,
                    extension
                )

            return file_bytes

        bytes_a = process_uploaded_file(
            sample_a
        )

        bytes_b = process_uploaded_file(
            sample_b
        )

        if bytes_a is None or bytes_b is None:

            st.error(
                "Unable to process one of the samples."
            )

        else:

            result_a = analyze_audio(
                bytes_a
            )

            result_b = analyze_audio(
                bytes_b
            )

            if result_a is not None and result_b is not None:

                st.session_state.comparison_result = {
                    "A": result_a,
                    "B": result_b
                }


if st.session_state.comparison_result is not None:

    comparison = (
        st.session_state.comparison_result
    )

    st.markdown(
        "### 📊 Comparison Result"
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.metric(
            "👤 Sample A Risk",
            f"{comparison['A']['risk']:.2f}%"
        )

        if comparison["A"]["level"] == "HIGH":

            st.error("HIGH RISK")

        elif comparison["A"]["level"] == "MEDIUM":

            st.warning("MEDIUM RISK")

        else:

            st.success("LOW RISK")

    with col_b:

        st.metric(
            "🤖 Sample B Risk",
            f"{comparison['B']['risk']:.2f}%"
        )

        if comparison["B"]["level"] == "HIGH":

            st.error("HIGH RISK")

        elif comparison["B"]["level"] == "MEDIUM":

            st.warning("MEDIUM RISK")

        else:

            st.success("LOW RISK")

    difference = abs(
        comparison["A"]["risk"]
        -
        comparison["B"]["risk"]
    )

    st.metric(
        "Risk Difference",
        f"{difference:.2f}"
    )

    if (
        comparison["A"]["level"]
        !=
        comparison["B"]["level"]
    ):

        st.success(
            "✅ The two samples produced different risk levels."
        )

    else:

        st.warning(
            "⚠️ Both samples produced the same risk level. "
            "This can happen with challenging or unseen audio."
        )


st.divider()


# =========================================================
# FRAUD PREVENTION FLOW
# =========================================================

st.markdown(
    "## 🛡️ Fraud Prevention Flow"
)

st.write(
    """
    📞 Incoming Voice
    → 🤖 AI Detection
    → 📊 Risk Score
    → 🚨 High Risk
    → 🛑 Stop Sensitive Action
    → 🔐 Secondary Verification
    → ✅ Allow / ❌ Block
    """
)


st.divider()


# =========================================================
# DETECTION HISTORY
# =========================================================

st.markdown(
    "## 🕘 Detection History"
)

if len(
    st.session_state.history
) == 0:

    st.info(
        "No scans performed yet."
    )

else:

    for index, item in enumerate(
        reversed(
            st.session_state.history
        )
    ):

        st.write(
            f"**{len(st.session_state.history) - index}. "
            f"{item['source']}**"
        )

        st.write(
            f"Sample Type: {item['sample_type']} | "
            f"Risk: {item['risk']:.2f} / 100 | "
            f"Level: {item['level']}"
        )

        st.divider()


# =========================================================
# SECURITY RECOMMENDATION
# =========================================================

st.markdown(
    "## 🔐 Security Recommendation"
)

st.info(
    """
    VoiceShield AI should be treated as a risk-assessment
    layer rather than absolute proof of identity.

    For high-risk or suspicious calls, sensitive actions
    should require secondary verification such as OTP,
    trusted callback, device authentication or human review.
    """
)


# =========================================================
# ABOUT
# =========================================================

st.markdown(
    "## ℹ️ About VoiceShield AI"
)

st.write(
    """
    VoiceShield AI is an AI-powered voice cloning detection
    and impersonation fraud prevention prototype.

    The system analyzes voice audio using the Dhwani
    multilingual deepfake-audio detection model and converts
    the model output into an impersonation risk score from
    0 to 100.

    The project focuses not only on detecting suspicious
    voices, but also on preventing fraud by recommending
    secondary identity verification before sensitive actions.
    """
)


# =========================================================
# RESPONSIBLE USE
# =========================================================

st.caption(
    "⚠️ Responsible use: AI voice detection is probabilistic. "
    "Results may vary with language, accent, noise, compression "
    "and unseen voice-cloning methods."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ VoiceShield AI | AI-powered voice cloning detection "
    "and impersonation fraud prevention"
)v
