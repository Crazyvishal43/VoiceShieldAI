import io
import os
import subprocess
import urllib.request

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import streamlit as st


st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide",
)


MODEL_PATH = "best_model.onnx"

MODEL_URL = (
    "https://huggingface.co/ayush2635/"
    "Dhwani-Multilingual-Deepfake-Audio-Detection-Model/"
    "resolve/main/best_model.onnx"
)


# ==============================
# SESSION STATE
# ==============================

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


# ==============================
# STYLE
# ==============================

st.markdown(
    """
    <style>
    .risk-high {
        border-left: 5px solid #d92d20;
        padding: 12px;
        border-radius: 8px;
        background: #fff5f4;
    }

    .risk-medium {
        border-left: 5px solid #f79009;
        padding: 12px;
        border-radius: 8px;
        background: #fffaeb;
    }

    .risk-low {
        border-left: 5px solid #12b76a;
        padding: 12px;
        border-radius: 8px;
        background: #f0fdf4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# MAIN TITLE
# ==============================

st.title("🛡️ VoiceShield AI")

st.caption(
    "AI-powered voice cloning detection and impersonation fraud prevention system"
)

st.divider()


# ==============================
# SYSTEM STATUS
# ==============================

c1, c2, c3 = st.columns(3)

with c1:
    st.success("🟢 SYSTEM STATUS\n\nONLINE")

with c2:
    st.info("🤖 AI ENGINE\n\nREADY")

with c3:
    st.success("🔐 PRIVACY MODE\n\nLOCAL ANALYSIS")


st.divider()


# ==============================
# MODEL
# ==============================

def download_model():

    if not os.path.exists(MODEL_PATH):

        st.info(
            "Downloading AI detection model for first-time setup..."
        )

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )


@st.cache_resource
def get_model():

    download_model()

    return ort.InferenceSession(
        MODEL_PATH
    )


# ==============================
# AUDIO CONVERSION
# ==============================

def convert_audio(uploaded_file):

    data = uploaded_file.read()

    name = uploaded_file.name.lower()


    if name.endswith(".wav"):

        audio, sr = sf.read(
            io.BytesIO(data),
            dtype="float32"
        )

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1
            )

        return audio, sr


    temp_input = "temp_input_audio"
    temp_output = "temp_converted.wav"


    with open(
        temp_input,
        "wb"
    ) as f:

        f.write(data)


    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_input,
                "-ar",
                "16000",
                "-ac",
                "1",
                temp_output,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


        if result.returncode != 0:

            raise RuntimeError(
                result.stderr[-1000:]
            )


        audio, sr = sf.read(
            temp_output,
            dtype="float32"
        )

        return audio, sr


    finally:

        for path in (
            temp_input,
            temp_output
        ):

            if os.path.exists(path):

                try:
                    os.remove(path)

                except OSError:

                    pass


# ==============================
# AUDIO PREPARATION
# ==============================

def prepare_audio(
    audio,
    sr
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )


    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )


    if sr != 16000:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )


    max_len = 48000


    if len(audio) > max_len:

        audio = audio[:max_len]

    else:

        audio = np.pad(
            audio,
            (
                0,
                max_len - len(audio)
            )
        )


    audio = (
        audio - np.mean(audio)
    ) / np.sqrt(
        np.var(audio) + 1e-5
    )


    return audio.astype(
        np.float32
    ).reshape(
        1,
        max_len
    )


# ==============================
# AI ANALYSIS
# ==============================

def analyze_audio(
    audio,
    sr
):

    session = get_model()

    prepared = prepare_audio(
        audio,
        sr
    )


    input_name = (
        session
        .get_inputs()[0]
        .name
    )


    logits = session.run(
        None,
        {
            input_name: prepared
        }
    )[0]


    logits = np.asarray(
        logits,
        dtype=np.float64
    )


    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )


    probabilities = np.exp(
        logits
    )


    probabilities = (
        probabilities
        / np.sum(
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


# ==============================
# RISK LEVEL
# ==============================

def get_level(risk):

    if risk >= 70:

        return "HIGH RISK"

    if risk >= 30:

        return "MEDIUM RISK"

    return "LOW RISK"


# ==============================
# HISTORY
# ==============================

def save_history(
    source,
    risk
):

    st.session_state.history.insert(
        0,
        {
            "source": source,
            "risk": float(risk),
            "level": get_level(risk)
        }
    )

    st.session_state.history = (
        st.session_state.history[:20]
    )


# ==============================
# EXPLAINABLE AI
# ==============================

def show_explainable_ai(risk):

    st.markdown(
        "### 🔎 Explainable AI"
    )


    if risk >= 70:

        st.error(
            "The AI model detected a strong synthetic/deepfake voice signal. "
            "Treat the speaker as untrusted until identity is verified."
        )

        st.write(
            "• Synthetic speech signal: **Strong**"
        )

        st.write(
            "• Acoustic authenticity assessment: **Suspicious**"
        )

        st.write(
            "• Security response: **Block / Verify**"
        )


    elif risk >= 30:

        st.warning(
            "The model detected an uncertain voice pattern. "
            "Use secondary verification before sensitive actions."
        )

        st.write(
            "• Synthetic speech signal: **Possible**"
        )

        st.write(
            "• Acoustic authenticity assessment: **Uncertain**"
        )

        st.write(
            "• Security response: **Verify**"
        )


    else:

        st.success(
            "No strong AI-cloning signal was detected in this sample."
        )

        st.write(
            "• Synthetic speech signal: **Low**"
        )

        st.write(
            "• Acoustic authenticity assessment: **Likely genuine**"
        )

        st.write(
            "• Security response: **Normal monitoring**"
        )


    st.caption(
        "These explanations summarize the model risk result; "
        "they are not independent forensic measurements."
    )


# ==============================
# ANALYSIS RESULT
# ==============================

def show_analysis_result(
    risk,
    title
):

    level = get_level(
        risk
    )


    st.markdown(
        f"### {title}"
    )


    if level == "HIGH RISK":

        st.markdown(
            f"""
            <div class="risk-high">
            <b>🚨 HIGH RISK</b><br>
            Potential voice cloning detected.<br>
            Risk score: <b>{risk:.2f} / 100</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    elif level == "MEDIUM RISK":

        st.markdown(
            f"""
            <div class="risk-medium">
            <b>⚠️ MEDIUM RISK</b><br>
            Voice requires additional verification.<br>
            Risk score: <b>{risk:.2f} / 100</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class="risk-low">
            <b>✅ LOW RISK</b><br>
            No strong AI-cloning signal detected.<br>
            Risk score: <b>{risk:.2f} / 100</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.progress(
        int(
            max(
                0,
                min(
                    100,
                    risk
                )
            )
        )
    )


    if level == "HIGH RISK":

        st.error(
            "🔐 Security Recommendation: Do not approve sensitive actions "
            "based on voice alone. Require secondary identity verification."
        )


    elif level == "MEDIUM RISK":

        st.warning(
            "🔐 Security Recommendation: Ask for secondary verification "
            "before approving a sensitive action."
        )


    else:

        st.info(
            "🔐 Security Recommendation: Voice appears low risk, "
            "but voice-only authentication should not be treated "
            "as absolute proof."
        )


    show_explainable_ai(
        risk
    )


# ==============================
# SECURITY OVERVIEW
# ==============================

st.markdown(
    "## 📊 Security Overview"
)


total = len(
    st.session_state.history
)

high = sum(
    x["level"] == "HIGH RISK"
    for x in st.session_state.history
)

medium = sum(
    x["level"] == "MEDIUM RISK"
    for x in st.session_state.history
)

low = sum(
    x["level"] == "LOW RISK"
    for x in st.session_state.history
)


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Total Scans",
    total
)

m2.metric(
    "High Risk",
    high
)

m3.metric(
    "Medium Risk",
    medium
)

m4.metric(
    "Low Risk",
    low
)


st.divider()


# ==============================
# SPEAK & DETECT
# ==============================

st.markdown(
    "## 🎤 Speak & Detect"
)


recorded = st.audio_input(
    "🎙️ Record your voice"
)


if recorded is not None:

    st.audio(
        recorded
    )


    if st.button(
        "🔍 Analyze Recorded Voice",
        use_container_width=True
    ):

        try:

            audio, sr = sf.read(
                recorded,
                dtype="float32"
            )


            if audio.ndim > 1:

                audio = np.mean(
                    audio,
                    axis=1
                )


            risk = analyze_audio(
                audio,
                sr
            )


            st.session_state.recorded_result = risk


            save_history(
                "Microphone",
                risk
            )


            st.rerun()


        except Exception as e:

            st.error(
                f"Audio analysis failed: {e}"
            )


if (
    st.session_state.recorded_result
    is not None
):

    show_analysis_result(
        st.session_state.recorded_result,
        "🎤 Microphone Result"
    )


st.divider()


# ==============================
# VOICE DETECTION
# ==============================

st.markdown(
    "## 🎧 Voice Detection"
)


uploaded = st.file_uploader(
    "Upload a voice sample",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "mp4",
        "m4a",
        "ogg",
        "flac"
    ]
)


if uploaded is not None:

    st.audio(
        uploaded
    )


    if st.button(
        "🤖 Analyze Voice",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "AI is analyzing the voice..."
            ):

                audio, sr = convert_audio(
                    uploaded
                )

                risk = analyze_audio(
                    audio,
                    sr
                )


            st.session_state.upload_result = risk


            save_history(
                uploaded.name,
                risk
            )


            st.rerun()


        except Exception as e:

            st.error(
                f"Audio analysis failed: {e}"
            )


if (
    st.session_state.upload_result
    is not None
):

    show_analysis_result(
        st.session_state.upload_result,
        "🤖 Voice Detection Result"
    )


st.divider()


# ==============================
# INCOMING VOICE SECURITY GATE
# ==============================

st.markdown(
    "## 📞 Incoming Voice Security Gate"
)


incoming = st.file_uploader(
    "Upload an incoming-call voice sample",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "mp4",
        "m4a",
        "ogg",
        "flac"
    ],
    key="incoming_voice"
)


if incoming is not None:

    st.audio(
        incoming
    )


    if st.button(
        "🚨 Scan Incoming Voice",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Scanning incoming voice for impersonation risk..."
            ):

                audio, sr = convert_audio(
                    incoming
                )

                risk = analyze_audio(
                    audio,
                    sr
                )


            st.session_state.incoming_result = risk


            save_history(
                "Incoming Call",
                risk
            )


            st.rerun()


        except Exception as e:

            st.error(
                f"Incoming voice analysis failed: {e}"
            )


if (
    st.session_state.incoming_result
    is not None
):

    risk = (
        st.session_state.incoming_result
    )


    show_analysis_result(
        risk,
        "📞 Incoming Call Security Decision"
    )


    if risk >= 70:

        st.error(
            "🛑 Sensitive Action Automatically Stopped"
        )

        st.warning(
            "🔐 Identity Verification Required"
        )


        method = st.selectbox(
            "Choose secondary verification method",
            [
                "OTP Verification",
                "Trusted Callback",
                "Device Authentication",
                "In-App Confirmation",
                "Human Verification"
            ],
            key="verification_method"
        )


        if st.button(
            "🔐 Verify Identity",
            use_container_width=True
        ):

            st.success(
                f"Prototype verification step initiated using: {method}"
            )

            st.caption(
                "Demo note: this is a simulated verification step, "
                "not a live OTP or identity-provider integration."
            )


st.divider()


# ==============================
# HUMAN VS AI
# ==============================

st.markdown(
    "## 👤 Human vs 🤖 AI Comparison"
)


col1, col2 = st.columns(2)


with col1:

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


with col2:

    sample_b = st.file_uploader(
        "Sample B — AI / Suspected Cloned Voice",
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


if st.button(
    "⚖️ Compare Samples",
    use_container_width=True
):

    if (
        sample_a is None
        or sample_b is None
    ):

        st.warning(
            "Please upload both Sample A and Sample B."
        )

    else:

        try:

            with st.spinner(
                "Comparing both voice samples..."
            ):

                a_audio, a_sr = convert_audio(
                    sample_a
                )

                b_audio, b_sr = convert_audio(
                    sample_b
                )

                a_risk = analyze_audio(
                    a_audio,
                    a_sr
                )

                b_risk = analyze_audio(
                    b_audio,
                    b_sr
                )


            st.session_state.comparison_result = {
                "A": {
                    "risk": a_risk,
                    "level": get_level(a_risk)
                },
                "B": {
                    "risk": b_risk,
                    "level": get_level(b_risk)
                }
            }


        except Exception as e:

            st.error(
                f"Comparison failed: {e}"
            )


if (
    st.session_state.comparison_result
):

    comparison = (
        st.session_state.comparison_result
    )


    a = comparison["A"]["risk"]
    b = comparison["B"]["risk"]


    st.markdown(
        "### 📊 Comparison Result"
    )


    ca, cb = st.columns(2)


    with ca:

        st.metric(
            "👤 Sample A Risk",
            f"{a:.2f}%"
        )


    with cb:

        st.metric(
            "🤖 Sample B Risk",
            f"{b:.2f}%"
        )


    st.markdown(
        "**Sample A**"
    )

    st.progress(
        int(
            max(
                0,
                min(
                    100,
                    a
                )
            )
        )
    )

    st.caption(
        f"Risk Score: {a:.2f}%"
    )


    st.markdown(
        "**Sample B**"
    )

    st.progress(
        int(
            max(
                0,
                min(
                    100,
                    b
                )
            )
        )
    )

    st.caption(
        f"Risk Score: {b:.2f}%"
    )


    difference = abs(
        a - b
    )


    st.metric(
        "Risk Difference",
        f"{difference:.2f} points"
    )


    if difference >= 20:

        st.success(
            "The two samples show a clear difference "
            "in model risk scores."
        )

    else:

        st.warning(
            "The two samples have similar model risk scores. "
            "Use additional verification."
        )


st.divider()


# ==============================
# FRAUD PREVENTION FLOW
# ==============================

st.markdown(
    "## 🛡️ Fraud Prevention Flow"
)


f1, f2, f3, f4 = st.columns(4)


f1.info(
    "🎙️ Detect\n\nIncoming Voice"
)

f2.info(
    "🤖 Score\n\nAI Risk Analysis"
)

f3.warning(
    "🔐 Verify\n\nIdentity Check"
)

f4.success(
    "✅ Decide\n\nAllow / Block"
)


st.divider()


# ==============================
# DETECTION HISTORY
# ==============================

st.markdown(
    "## 📜 Detection History"
)


if st.session_state.history:

    for item in st.session_state.history:

        st.write(
            f"**{item['source']}** — "
            f"{item['risk']:.2f}/100 — "
            f"{item['level']}"
        )

else:

    st.caption(
        "No scans yet."
    )


st.divider()


# ==============================
# SECURITY RECOMMENDATION
# ==============================

st.markdown(
    "## 🔐 Security Recommendation"
)


st.write(
    "Use VoiceShield AI as a security screening layer. "
    "High-risk voice detections should trigger a secondary "
    "identity check before financial, account, password-reset, "
    "or other sensitive actions are approved."
)


st.divider()


# ==============================
# ABOUT
# ==============================

st.markdown(
    "## ℹ️ About VoiceShield AI"
)


st.write(
    "VoiceShield AI is a prototype designed to detect potential "
    "AI-generated or cloned speech and convert the detection result "
    "into an actionable fraud-prevention decision."
)


st.caption(
    "Responsible-use note: voice deepfake detection is probabilistic. "
    "Performance can vary with unseen generators, languages, accents, "
    "recording conditions, compression, and background noise."
)


st.caption(
    "VoiceShield AI • Hackathon Prototype"
)
