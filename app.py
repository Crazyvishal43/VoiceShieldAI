import io
import os
import subprocess
import tempfile
from datetime import datetime

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import streamlit as st


# ============================================================
# VOICESHIELD AI
# AI-Powered Voice Cloning Detection & Fraud Prevention
# ============================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "best_model.onnx"

MODEL_URL = (
    "https://huggingface.co/ayush2635/"
    "Dhwani-Multilingual-Deepfake-Audio-Detection-Model/"
    "resolve/main/best_model.onnx"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .security-card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "incoming_result" not in st.session_state:
    st.session_state.incoming_result = None


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.info(
            "Downloading AI detection model for first-time setup..."
        )

        import urllib.request

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    return session


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio_to_wav(audio_bytes, filename):

    extension = os.path.splitext(filename)[1].lower()

    if extension == ".wav":
        return audio_bytes

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    )

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    input_path = input_file.name
    output_path = output_file.name

    input_file.write(audio_bytes)
    input_file.close()
    output_file.close()

    try:

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        with open(output_path, "rb") as f:
            result = f.read()

        return result

    except Exception:

        return None

    finally:

        try:
            os.remove(input_path)
        except:
            pass

        try:
            os.remove(output_path)
        except:
            pass


# ============================================================
# AUDIO PREPARATION
# ============================================================

def prepare_audio(audio_bytes):

    audio_array, sample_rate = sf.read(
        io.BytesIO(audio_bytes)
    )

    if len(audio_array.shape) > 1:

        audio_array = np.mean(
            audio_array,
            axis=1
        )

    audio_array = audio_array.astype(
        np.float32
    )

    if sample_rate != 16000:

        audio_array = librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=16000
        )

    max_len = 48000

    if len(audio_array) > max_len:

        audio_array = audio_array[:max_len]

    else:

        audio_array = np.pad(
            audio_array,
            (
                0,
                max_len - len(audio_array)
            ),
            mode="constant"
        )

    audio_array = (
        audio_array - np.mean(audio_array)
    ) / (
        np.sqrt(
            np.var(audio_array) + 1e-5
        )
    )

    return audio_array.astype(
        np.float32
    )


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_audio(audio_bytes):

    session = load_model()

    audio = prepare_audio(
        audio_bytes
    )

    input_name = session.get_inputs()[0].name

    audio_input = audio.reshape(
        1,
        48000
    )

    logits = session.run(
        None,
        {
            input_name: audio_input
        }
    )[0]

    logits = logits.astype(
        np.float64
    )

    exp_logits = np.exp(
        logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        exp_logits /
        np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )

    fake_probability = float(
        probabilities[0][1]
    )

    return fake_probability * 100


# ============================================================
# RISK LEVEL
# ============================================================

def get_level(risk):

    if risk >= 70:
        return "HIGH RISK"

    elif risk >= 30:
        return "MEDIUM RISK"

    return "LOW RISK"


# ============================================================
# SECURITY DECISION
# ============================================================

def security_decision(risk):

    level = get_level(risk)

    if level == "HIGH RISK":
        return "BLOCK / VERIFY"

    elif level == "MEDIUM RISK":
        return "PAUSE / VERIFY"

    return "ALLOW WITH CAUTION"


# ============================================================
# HISTORY
# ============================================================

def add_history(source, risk):

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Source": source,
            "Risk Score": round(
                risk,
                2
            ),
            "Level": get_level(risk),
            "Decision": security_decision(risk)
        }
    )


# ============================================================
# SECURITY ACTION
# ============================================================

def show_security_action(risk):

    level = get_level(risk)

    st.markdown(
        "### 🛡️ Security Recommendation"
    )

    if level == "HIGH RISK":

        st.error(
            """
            🚨 **Potential voice cloning detected**

            **Security Gate: BLOCK / VERIFY**

            Do NOT approve sensitive actions using voice alone.

            Recommended:

            **STOP → VERIFY IDENTITY → SECONDARY AUTHENTICATION → CONTINUE**
            """
        )

        st.warning(
            """
            🔐 Secondary verification options:

            • OTP verification  
            • Trusted callback  
            • In-app confirmation  
            • Device authentication  
            • Human verification
            """
        )

    elif level == "MEDIUM RISK":

        st.warning(
            """
            ⚠️ **Suspicious voice signal detected**

            **Security Gate: PAUSE / VERIFY**

            Recommended:

            **PAUSE → VERIFY IDENTITY → THEN CONTINUE**
            """
        )

    else:

        st.success(
            """
            ✅ **Low-risk voice signal**

            **Security Gate: ALLOW WITH CAUTION**

            No strong AI-cloning signal was detected.

            Sensitive actions should still use secondary
            authentication whenever possible.
            """
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

def show_explanation(risk):

    st.markdown(
        "### 🔍 Explainable AI"
    )

    if risk >= 70:

        st.write(
            """
            The AI engine detected acoustic characteristics that
            are consistent with a potentially synthetic or manipulated
            voice.

            Possible indicators include unusual spectral patterns,
            speech characteristics and synthetic-generation artifacts.

            This result is a **security warning**, not absolute
            forensic proof.
            """
        )

    elif risk >= 30:

        st.write(
            """
            The AI engine detected characteristics requiring
            additional verification.

            The signal is not clearly safe or clearly synthetic,
            so secondary authentication is recommended.
            """
        )

    else:

        st.write(
            """
            The analyzed voice did not show a strong AI-cloning
            signal according to the current model.

            A low risk score does not prove the speaker's identity.
            """
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🛡️ VoiceShield AI</h1>

        <p>
        AI-powered voice cloning detection and impersonation
        fraud prevention system
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STATUS
# ============================================================

status1, status2, status3 = st.columns(3)

with status1:

    st.success(
        "🟢 SYSTEM STATUS\n\nONLINE"
    )

with status2:

    st.info(
        "🤖 AI ENGINE\n\nREADY"
    )

with status3:

    st.success(
        "🔐 PRIVACY MODE\n\nLOCAL ANALYSIS"
    )


# ============================================================
# SECURITY OVERVIEW
# ============================================================

st.markdown(
    "## 📊 Security Overview"
)

total_scans = len(
    st.session_state.history
)

high_risk = sum(
    1
    for item in st.session_state.history
    if item["Level"] == "HIGH RISK"
)

medium_risk = sum(
    1
    for item in st.session_state.history
    if item["Level"] == "MEDIUM RISK"
)

low_risk = sum(
    1
    for item in st.session_state.history
    if item["Level"] == "LOW RISK"
)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Total Scans",
        total_scans
    )

with m2:
    st.metric(
        "High Risk",
        high_risk
    )

with m3:
    st.metric(
        "Medium Risk",
        medium_risk
    )

with m4:
    st.metric(
        "Low Risk",
        low_risk
    )


# ============================================================
# INCOMING VOICE SECURITY GATE
# ============================================================

st.markdown("---")

st.header(
    "📞 Incoming Voice Security Gate"
)

st.write(
    """
    Simulate an incoming voice interaction before allowing
    a sensitive action such as financial approval, account
    recovery or confidential access.
    """
)


incoming_file = st.file_uploader(
    "Upload incoming voice sample",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "m4a",
        "ogg",
        "flac"
    ],
    key="incoming_voice"
)


if incoming_file is not None:

    st.audio(
        incoming_file
    )

    if st.button(
        "🚨 Scan Incoming Voice",
        use_container_width=True
    ):

        with st.spinner(
            "Scanning incoming voice for impersonation risk..."
        ):

            try:

                original_bytes = (
                    incoming_file.getvalue()
                )

                wav_bytes = convert_audio_to_wav(
                    original_bytes,
                    incoming_file.name
                )

                if wav_bytes is None:

                    st.error(
                        "Audio conversion failed."
                    )

                    st.stop()

                risk = analyze_audio(
                    wav_bytes
                )

                level = get_level(
                    risk
                )

                decision = security_decision(
                    risk
                )

                st.session_state.incoming_result = {
                    "risk": risk,
                    "level": level,
                    "decision": decision,
                    "source": incoming_file.name
                }

                add_history(
                    "Incoming: " + incoming_file.name,
                    risk
                )

            except Exception as e:

                st.error(
                    f"Incoming voice analysis failed: {e}"
                )


# ============================================================
# INCOMING RESULT
# ============================================================

if st.session_state.incoming_result:

    result = (
        st.session_state.incoming_result
    )

    st.markdown(
        "## 🚦 Incoming Voice Decision"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Impersonation Risk",
            f"{result['risk']:.2f} / 100"
        )

    with r2:

        st.metric(
            "Risk Level",
            result["level"]
        )

    with r3:

        st.metric(
            "Security Decision",
            result["decision"]
        )

    show_security_action(
        result["risk"]
    )

    if result["level"] == "HIGH RISK":

        st.error(
            """
            🛑 **Sensitive Action Automatically Stopped**

            VoiceShield AI has detected a high impersonation risk.

            The requested sensitive action should remain blocked
            until identity verification is completed.
            """
        )

        st.markdown(
            "### 🔐 Identity Verification Required"
        )

        verification = st.selectbox(
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
                f"""
                Verification step initiated using:

                **{verification}**

                This prototype simulates the verification gate.
                In a production system, this step would connect
                to the organization's real authentication service.
                """
            )

            st.info(
                """
                🛡️ **Action Status: WAITING FOR VERIFIED IDENTITY**

                The sensitive action remains protected until
                successful secondary authentication.
                """
            )

    elif result["level"] == "MEDIUM RISK":

        st.warning(
            """
            ⚠️ **Sensitive Action Paused**

            Additional identity verification is required before
            continuing the requested action.
            """
        )

    else:

        st.success(
            """
            🟢 **Security Gate Passed with Caution**

            No strong AI-cloning signal detected.

            For high-value actions, secondary authentication
            is still recommended.
            """
        )

    show_explanation(
        result["risk"]
    )


# ============================================================
# MICROPHONE DETECTION
# ============================================================

st.markdown("---")

st.header(
    "🎤 Speak & Detect"
)

st.write(
    "Record a short voice sample and analyze it using the AI engine."
)


recorded_audio = st.audio_input(
    "🎙️ Record your voice"
)


if recorded_audio is not None:

    st.audio(
        recorded_audio
    )

    if st.button(
        "🤖 Analyze Recorded Voice",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing your voice..."
        ):

            try:

                audio_bytes = (
                    recorded_audio.getvalue()
                )

                risk = analyze_audio(
                    audio_bytes
                )

                add_history(
                    "Microphone",
                    risk
                )

                st.markdown(
                    "## 🎯 Analysis Result"
                )

                st.metric(
                    "Impersonation Risk Score",
                    f"{risk:.2f} / 100"
                )

                level = get_level(
                    risk
                )

                if level == "HIGH RISK":

                    st.error(
                        f"🚨 {level}"
                    )

                elif level == "MEDIUM RISK":

                    st.warning(
                        f"⚠️ {level}"
                    )

                else:

                    st.success(
                        f"✅ {level}"
                    )

                show_security_action(
                    risk
                )

                show_explanation(
                    risk
                )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown("---")

st.header(
    "📁 Voice Detection"
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
    key="general_upload"
)


if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    if st.button(
        "🔎 Analyze Uploaded Voice",
        use_container_width=True
    ):

        with st.spinner(
            "Processing audio..."
        ):

            try:

                original_bytes = (
                    uploaded_file.getvalue()
                )

                wav_bytes = convert_audio_to_wav(
                    original_bytes,
                    uploaded_file.name
                )

                if wav_bytes is None:

                    st.error(
                        "Audio conversion failed."
                    )

                    st.stop()

                risk = analyze_audio(
                    wav_bytes
                )

                add_history(
                    uploaded_file.name,
                    risk
                )

                st.markdown(
                    "## 🎯 Analysis Result"
                )

                st.metric(
                    "Impersonation Risk Score",
                    f"{risk:.2f} / 100"
                )

                level = get_level(
                    risk
                )

                if level == "HIGH RISK":

                    st.error(
                        f"🚨 {level}"
                    )

                elif level == "MEDIUM RISK":

                    st.warning(
                        f"⚠️ {level}"
                    )

                else:

                    st.success(
                        f"✅ {level}"
                    )

                show_security_action(
                    risk
                )

                show_explanation(
                    risk
                )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )


# ============================================================
# HUMAN VS AI
# ============================================================

st.markdown("---")

st.header(
    "👤 Human vs 🤖 AI Voice Comparison"
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "👤 Sample A — Human"
    )

    human_file = st.file_uploader(
        "Upload human voice",
        type=[
            "wav",
            "mp3",
            "mpeg",
            "m4a",
            "ogg",
            "flac"
        ],
        key="human"
    )


with col2:

    st.subheader(
        "🤖 Sample B — AI"
    )

    ai_file = st.file_uploader(
        "Upload AI voice",
        type=[
            "wav",
            "mp3",
            "mpeg",
            "m4a",
            "ogg",
            "flac"
        ],
        key="ai"
    )


if st.button(
    "⚔️ Compare Human vs AI",
    use_container_width=True
):

    if human_file is None or ai_file is None:

        st.warning(
            "Please upload both Human and AI voice samples."
        )

    else:

        with st.spinner(
            "Comparing both voice samples..."
        ):

            try:

                human_wav = convert_audio_to_wav(
                    human_file.getvalue(),
                    human_file.name
                )

                ai_wav = convert_audio_to_wav(
                    ai_file.getvalue(),
                    ai_file.name
                )

                if human_wav is None or ai_wav is None:

                    st.error(
                        "One or both audio files could not be converted."
                    )

                    st.stop()

                human_risk = analyze_audio(
                    human_wav
                )

                ai_risk = analyze_audio(
                    ai_wav
                )

                st.session_state.comparison_results = {
                    "A": {
                        "risk": human_risk,
                        "level": get_level(human_risk)
                    },
                    "B": {
                        "risk": ai_risk,
                        "level": get_level(ai_risk)
                    }
                }

            except Exception as e:

                st.error(
                    f"Comparison failed: {e}"
                )


# ============================================================
# COMPARISON RESULT
# ============================================================

if st.session_state.comparison_results:

    comparison = (
        st.session_state.comparison_results
    )

    st.markdown(
        "## 📊 Comparison Result"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "👤 Sample A Risk",
            f"{comparison['A']['risk']:.2f}%"
        )

        st.progress(
            min(
                max(
                    int(
                        comparison["A"]["risk"]
                    ),
                    0
                ),
                100
            )
        )

        st.caption(
            f"Sample A — {comparison['A']['level']}"
        )

    with c2:

        st.metric(
            "🤖 Sample B Risk",
            f"{comparison['B']['risk']:.2f}%"
        )

        st.progress(
            min(
                max(
                    int(
                        comparison["B"]["risk"]
                    ),
                    0
                ),
                100
            )
        )

        st.caption(
            f"Sample B — {comparison['B']['level']}"
        )

    difference = abs(
        comparison["A"]["risk"]
        -
        comparison["B"]["risk"]
    )

    st.metric(
        "Risk Difference",
        f"{difference:.2f} points"
    )

    if (
        comparison["A"]["level"]
        !=
        comparison["B"]["level"]
    ):

        st.success(
            "✅ Different risk levels detected between the two samples."
        )

    else:

        st.warning(
            "⚠️ Both samples received the same risk category. "
            "This prototype result should not be treated as definitive."
        )


# ============================================================
# FRAUD PREVENTION WORKFLOW
# ============================================================

st.markdown("---")

st.header(
    "🚨 Fraud Prevention Workflow"
)

st.write(
    "VoiceShield AI acts as a security gate before a sensitive action."
)


f1, f2, f3, f4 = st.columns(4)


with f1:

    st.markdown(
        """
        ### 1️⃣ DETECT

        🎤

        Analyze incoming voice.
        """
    )


with f2:

    st.markdown(
        """
        ### 2️⃣ SCORE

        📊

        Calculate impersonation risk.
        """
    )


with f3:

    st.markdown(
        """
        ### 3️⃣ VERIFY

        🔐

        Trigger secondary authentication.
        """
    )


with f4:

    st.markdown(
        """
        ### 4️⃣ DECIDE

        🛡️

        Allow or block sensitive action.
        """
    )


st.info(
    """
    **Example security flow**

    📞 Incoming Voice
    →
    🤖 AI Detection
    →
    📊 Risk Score
    →
    🚨 High Risk
    →
    🛑 Stop Sensitive Action
    →
    🔐 Secondary Verification
    →
    ✅ Allow / ❌ Block
    """
)


# ============================================================
# DETECTION HISTORY
# ============================================================

st.markdown("---")

st.header(
    "📜 Detection History"
)


if len(st.session_state.history) == 0:

    st.info(
        "No detection history yet."
    )

else:

    for item in reversed(
        st.session_state.history
    ):

        st.write(
            f"🕒 {item['Time']} | "
            f"🎤 {item['Source']} | "
            f"📊 {item['Risk Score']} / 100 | "
            f"🚦 {item['Level']} | "
            f"🛡️ {item['Decision']}"
        )


# ============================================================
# ABOUT
# ============================================================

st.markdown("---")

st.header(
    "ℹ️ About VoiceShield AI"
)

st.write(
    """
    VoiceShield AI is a prototype cybersecurity system designed to
    detect potentially AI-generated or manipulated speech and reduce
    the risk of voice-cloning impersonation attacks.

    The system analyzes voice, produces an impersonation risk score,
    and acts as a security gate before sensitive actions.

    Suspicious calls can be paused or blocked until secondary
    identity verification is completed.

    Voice-only authentication should never be treated as absolute
    proof of identity.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🛡️ VoiceShield AI — AI-powered voice cloning detection & fraud prevention"
)