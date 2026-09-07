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
# CUSTOM STYLE
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
        color: #777;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 14px;
        margin: 10px 0;
    }

    .section-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .small-note {
        color: #777;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL DOWNLOAD
# =========================================================

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI detection model for first-time setup..."):
            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )

    return MODEL_PATH


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():
    model_file = download_model()

    session = ort.InferenceSession(
        model_file,
        providers=["CPUExecutionProvider"]
    )

    return session


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_audio(uploaded_file):
    """
    Converts MP3/M4A/etc. to WAV using FFmpeg.
    """

    file_bytes = uploaded_file.getvalue()

    input_path = "temp_input_audio"
    output_path = "temp_converted.wav"

    try:
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            return None

        if not os.path.exists(output_path):
            return None

        audio, sr = sf.read(output_path)

        try:
            os.remove(input_path)
            os.remove(output_path)
        except Exception:
            pass

        return audio.astype(np.float32), sr

    except Exception:
        return None


# =========================================================
# AUDIO PREPARATION
# =========================================================

def prepare_audio(uploaded_file):
    """
    Reads uploaded audio and converts it to:
    16 kHz
    mono
    normalized
    maximum 3 seconds for Dhwani
    """

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".wav"):
            uploaded_file.seek(0)

            audio_bytes = io.BytesIO(
                uploaded_file.getvalue()
            )

            audio, sr = sf.read(
                audio_bytes
            )

            audio = np.asarray(
                audio,
                dtype=np.float32
            )

        else:
            converted = convert_audio(
                uploaded_file
            )

            if converted is None:
                return None

            audio, sr = converted

        # Stereo → mono
        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        # Resample
        if sr != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000
            )
            sr = 16000

        # Remove NaN/Inf
        audio = np.nan_to_num(
            audio
        )

        # Normalize
        audio = audio.astype(
            np.float32
        )

        mean = np.mean(audio)
        variance = np.var(audio)

        audio = (
            audio - mean
        ) / np.sqrt(
            variance + 1e-5
        )

        # Dhwani expects 48000 samples
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

        return audio, sr

    except Exception as e:
        st.error(
            f"Audio processing error: {e}"
        )
        return None


# =========================================================
# MODEL ANALYSIS
# =========================================================

def analyze_audio(audio):
    """
    Dhwani model inference.

    According to the model usage:
    probabilities[0][1] is treated as fake probability.
    """

    session = get_model()

    input_name = session.get_inputs()[0].name

    model_input = (
        audio
        .astype(np.float32)
        .reshape(1, 48000)
    )

    logits = session.run(
        None,
        {
            input_name: model_input
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
        exp_logits
        / np.sum(
            exp_logits,
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

    risk_score = max(
        0,
        min(
            100,
            risk_score
        )
    )

    return risk_score


# =========================================================
# SEGMENT ANALYSIS
# =========================================================

def segment_analysis(original_audio):
    """
    Additional security analysis.

    This does NOT replace the AI model score.

    It checks whether different portions of the recording
    produce very different AI scores.

    Large variation can indicate an uncertain/challenging
    recording and can trigger secondary verification.
    """

    try:
        audio = np.asarray(
            original_audio,
            dtype=np.float32
        )

        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        # Resample
        # Original audio is normally 16kHz here.
        sample_rate = 16000

        window_size = 48000
        hop_size = 24000

        scores = []

        session = get_model()

        input_name = session.get_inputs()[0].name

        if len(audio) < 8000:
            return {
                "scores": [],
                "mean": None,
                "std": None,
                "range": None,
                "challenging": False
            }

        start = 0

        while (
            start < len(audio)
            and len(scores) < 8
        ):

            segment = audio[
                start:
                start + window_size
            ]

            if len(segment) < 8000:
                break

            if len(segment) > window_size:
                segment = segment[
                    :window_size
                ]

            if len(segment) < window_size:
                segment = np.pad(
                    segment,
                    (
                        0,
                        window_size - len(segment)
                    ),
                    mode="constant"
                )

            segment = (
                segment
                - np.mean(segment)
            ) / np.sqrt(
                np.var(segment) + 1e-5
            )

            model_input = (
                segment
                .astype(np.float32)
                .reshape(1, 48000)
            )

            logits = session.run(
                None,
                {
                    input_name: model_input
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

            probs = (
                exp_logits
                / np.sum(
                    exp_logits,
                    axis=1,
                    keepdims=True
                )
            )

            score = (
                float(probs[0][1])
                * 100
            )

            scores.append(
                score
            )

            start += hop_size

        if not scores:
            return {
                "scores": [],
                "mean": None,
                "std": None,
                "range": None,
                "challenging": False
            }

        scores_array = np.array(
            scores
        )

        mean_score = float(
            np.mean(scores_array)
        )

        std_score = float(
            np.std(scores_array)
        )

        score_range = float(
            np.max(scores_array)
            - np.min(scores_array)
        )

        # This is an uncertainty indicator,
        # NOT a claim of automatic hybrid detection.
        challenging = (
            len(scores) >= 2
            and std_score >= 15
            and score_range >= 30
        )

        return {
            "scores": scores,
            "mean": mean_score,
            "std": std_score,
            "range": score_range,
            "challenging": challenging
        }

    except Exception:
        return {
            "scores": [],
            "mean": None,
            "std": None,
            "range": None,
            "challenging": False
        }


# =========================================================
# RISK LEVEL
# =========================================================

def get_level(score):
    if score >= 70:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


# =========================================================
# SECURITY DECISION
# =========================================================

def get_security_decision(
    model_score,
    segment_result=None,
    sample_type="Unknown"
):
    """
    Security decision.

    The model score is preserved.

    For Hybrid samples, the UI can classify the case as
    challenging/verification-required when segment analysis
    shows strong uncertainty.
    """

    level = get_level(
        model_score
    )

    if level == "HIGH":
        return {
            "level": "HIGH",
            "decision": "BLOCK / VERIFY",
            "reason": "Strong AI-cloning signal detected."
        }

    if (
        segment_result
        and segment_result.get("challenging")
    ):
        return {
            "level": "MEDIUM",
            "decision": "VERIFY",
            "reason": (
                "Different audio segments produced "
                "inconsistent AI-risk signals. "
                "Secondary identity verification is recommended."
            )
        }

    if sample_type == "Hybrid":
        return {
            "level": "MEDIUM",
            "decision": "VERIFY",
            "reason": (
                "This sample is marked as a challenging "
                "human + synthetic case. The current model "
                "score is preserved, but voice-only identity "
                "should not be trusted without secondary verification."
            )
        }

    if level == "MEDIUM":
        return {
            "level": "MEDIUM",
            "decision": "VERIFY",
            "reason": (
                "The model produced an intermediate risk signal. "
                "Additional identity verification is recommended."
            )
        }

    return {
        "level": "LOW",
        "decision": "ALLOW WITH CAUTION",
        "reason": (
            "No strong AI-cloning signal detected. "
            "Voice alone should not be treated as absolute proof of identity."
        )
    }


# =========================================================
# HISTORY
# =========================================================

def save_history(
    source,
    score,
    level,
    decision
):
    st.session_state.history.insert(
        0,
        {
            "Source": source,
            "Risk": round(score, 2),
            "Level": level,
            "Decision": decision
        }
    )

    st.session_state.history = (
        st.session_state.history[:20]
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

def show_explainable_ai(
    score,
    level,
    segment_result=None
):

    st.markdown(
        "### 🔎 Explainable AI"
    )

    col1, col2 = st.columns(2)

    with col1:

        if score >= 70:
            st.error(
                "🤖 Synthetic speech signal: STRONG"
            )

        elif score >= 30:
            st.warning(
                "⚠️ Synthetic speech signal: UNCERTAIN"
            )

        else:
            st.success(
                "👤 Synthetic speech signal: LOW"
            )

    with col2:

        if segment_result and segment_result.get(
            "challenging"
        ):
            st.warning(
                "🧬 Segment consistency: CHALLENGING"
            )

        else:
            st.info(
                "🎧 Segment consistency: NO STRONG WARNING"
            )

    if score >= 70:

        st.write(
            "The AI model produced a strong "
            "voice-cloning risk signal."
        )

        st.write(
            "Security response: sensitive actions "
            "should be blocked until identity is verified."
        )

    elif (
        segment_result
        and segment_result.get("challenging")
    ):

        st.write(
            "Different portions of the recording "
            "show inconsistent AI-risk signals."
        )

        st.write(
            "This can represent a challenging or "
            "uncertain audio case."
        )

        st.write(
            "Security response: require secondary verification."
        )

    elif score >= 30:

        st.write(
            "The model produced an intermediate "
            "risk signal."
        )

        st.write(
            "Security response: require secondary verification."
        )

    else:

        st.write(
            "No strong AI-cloning signal was detected "
            "by the current model."
        )

        st.write(
            "Voice-only authentication should still "
            "not be treated as absolute proof of identity."
        )


# =========================================================
# RESULT DISPLAY
# =========================================================

def show_analysis_result(
    score,
    source,
    sample_type="Unknown",
    original_audio=None,
    show_segments=True
):

    segment_result = None

    if (
        original_audio is not None
        and show_segments
    ):
        segment_result = segment_analysis(
            original_audio
        )

    security = get_security_decision(
        score,
        segment_result,
        sample_type
    )

    level = security["level"]

    st.markdown(
        "### 📊 Analysis Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "AI Risk Score",
            f"{score:.2f} / 100"
        )

    with col2:

        if level == "HIGH":
            st.error(
                f"🔴 {level} RISK"
            )

        elif level == "MEDIUM":
            st.warning(
                f"🟡 {level} RISK"
            )

        else:
            st.success(
                f"🟢 {level} RISK"
            )

    with col3:
        st.metric(
            "Security Decision",
            security["decision"]
        )

    st.progress(
        int(
            max(
                0,
                min(
                    100,
                    score
                )
            )
        )
    )

    if level == "HIGH":

        st.error(
            "🚨 Potential voice cloning detected."
        )

    elif level == "MEDIUM":

        st.warning(
            "⚠️ Verification recommended."
        )

    else:

        st.success(
            "✅ No strong AI-cloning signal detected."
        )

    st.info(
        f"Security Assessment: {security['reason']}"
    )

    if sample_type == "Hybrid":

        st.warning(
            "🧬 CHALLENGING HYBRID SAMPLE\n\n"
            "This sample is marked as Human + AI / "
            "synthetically modified audio for demonstration. "
            "The original model score is preserved. "
            "Because hybrid and unseen manipulation can be difficult "
            "for a detector, VoiceShield recommends secondary verification."
        )

    if segment_result:

        with st.expander(
            "🎧 Segment Analysis"
        ):

            scores = segment_result.get(
                "scores",
                []
            )

            if scores:

                st.write(
                    "Individual segment AI-risk scores:"
                )

                for i, segment_score in enumerate(
                    scores,
                    start=1
                ):
                    st.write(
                        f"Segment {i}: "
                        f"{segment_score:.2f}"
                    )

                st.write(
                    f"Average segment score: "
                    f"{segment_result['mean']:.2f}"
                )

                st.write(
                    f"Segment variation: "
                    f"{segment_result['std']:.2f}"
                )

                st.write(
                    f"Score range: "
                    f"{segment_result['range']:.2f}"
                )

                if segment_result[
                    "challenging"
                ]:

                    st.warning(
                        "The segments show substantial "
                        "risk variation. Treat this as an "
                        "uncertain/challenging case."
                    )

                else:

                    st.success(
                        "No strong segment-level "
                        "inconsistency warning."
                    )

    show_explainable_ai(
        score,
        level,
        segment_result
    )

    save_history(
        source,
        score,
        level,
        security["decision"]
    )

    return {
        "score": score,
        "level": level,
        "decision": security["decision"],
        "reason": security["reason"],
        "segment": segment_result
    }


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
# STATUS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        "🟢 SYSTEM STATUS\n\nONLINE"
    )

with col2:
    st.success(
        "🤖 AI ENGINE\n\nREADY"
    )

with col3:
    st.info(
        "🔐 PRIVACY MODE\n\nLOCAL ANALYSIS"
    )


st.divider()


# =========================================================
# SECURITY OVERVIEW
# =========================================================

st.header(
    "📈 Security Overview"
)

total_scans = len(
    st.session_state.history
)

high_count = sum(
    1
    for x in st.session_state.history
    if x["Level"] == "HIGH"
)

medium_count = sum(
    1
    for x in st.session_state.history
    if x["Level"] == "MEDIUM"
)

low_count = sum(
    1
    for x in st.session_state.history
    if x["Level"] == "LOW"
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
# MICROPHONE
# =========================================================

st.header(
    "🎤 Speak & Detect"
)

st.write(
    "Record a voice sample directly from your microphone."
)

recorded_audio = st.audio_input(
    "🎙️ Record your voice"
)

if recorded_audio is not None:

    if st.button(
        "🔍 Analyze Recorded Voice",
        key="analyze_recorded"
    ):

        prepared = prepare_audio(
            recorded_audio
        )

        if prepared is not None:

            audio, sr = prepared

            with st.spinner(
                "AI is analyzing your recorded voice..."
            ):

                score = analyze_audio(
                    audio
                )

            result = show_analysis_result(
                score,
                "Microphone",
                "Unknown",
                audio,
                True
            )

            st.session_state.recorded_result = (
                result
            )


st.divider()


# =========================================================
# VOICE DETECTION
# =========================================================

st.header(
    "📁 Voice Detection"
)

sample_type = st.selectbox(
    "Select sample type for analysis",
    [
        "Unknown",
        "Human",
        "AI-generated",
        "Hybrid"
    ]
)

uploaded_audio = st.file_uploader(
    "Upload voice sample",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac"
    ]
)

if uploaded_audio is not None:

    st.audio(
        uploaded_audio
    )

    if st.button(
        "🔍 Analyze Uploaded Voice",
        key="analyze_uploaded"
    ):

        prepared = prepare_audio(
            uploaded_audio
        )

        if prepared is not None:

            audio, sr = prepared

            with st.spinner(
                "AI is analyzing uploaded audio..."
            ):

                score = analyze_audio(
                    audio
                )

            result = show_analysis_result(
                score,
                uploaded_audio.name,
                sample_type,
                audio,
                True
            )

            st.session_state.upload_result = (
                result
            )


st.divider()


# =========================================================
# INCOMING VOICE SECURITY GATE
# =========================================================

st.header(
    "📞 Incoming Voice Security Gate"
)

st.write(
    "Prototype fraud-prevention workflow for an incoming voice interaction."
)

incoming_audio = st.file_uploader(
    "Upload incoming-call voice sample",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac"
    ],
    key="incoming_audio"
)

if incoming_audio is not None:

    st.audio(
        incoming_audio
    )

    if st.button(
        "🚨 Scan Incoming Voice",
        key="scan_incoming"
    ):

        prepared = prepare_audio(
            incoming_audio
        )

        if prepared is not None:

            audio, sr = prepared

            with st.spinner(
                "Scanning incoming voice for impersonation risk..."
            ):

                score = analyze_audio(
                    audio
                )

            result = show_analysis_result(
                score,
                "Incoming Voice",
                "Unknown",
                audio,
                True
            )

            st.session_state.incoming_result = (
                result
            )


# =========================================================
# INCOMING SECURITY RESPONSE
# =========================================================

if st.session_state.incoming_result:

    incoming = (
        st.session_state.incoming_result
    )

    st.markdown(
        "### 🔐 Security Response"
    )

    if incoming["level"] == "HIGH":

        st.error(
            "🚨 SENSITIVE ACTION AUTOMATICALLY STOPPED"
        )

        st.warning(
            "Identity verification is required before "
            "continuing the sensitive action."
        )

        verification_method = st.selectbox(
            "Select secondary verification method",
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
            "🔐 Verify Identity",
            key="verify_identity"
        ):

            st.success(
                f"Prototype verification initiated using: "
                f"{verification_method}"
            )

            st.info(
                "Demo note: this verification step is simulated. "
                "A production system would connect this to a real "
                "OTP, trusted device, callback or identity service."
            )

    elif incoming["level"] == "MEDIUM":

        st.warning(
            "🟡 VOICE IS UNCERTAIN — SECONDARY VERIFICATION REQUIRED"
        )

        st.info(
            "The system does not automatically trust the voice. "
            "Verify the caller using another trusted method."
        )

    else:

        st.success(
            "🟢 NO STRONG CLONING SIGNAL — CONTINUE WITH CAUTION"
        )

        st.info(
            "Voice-only authentication should not be treated "
            "as absolute proof of identity."
        )


st.divider()


# =========================================================
# HUMAN VS AI COMPARISON
# =========================================================

st.header(
    "👤 Human vs 🤖 AI Comparison"
)

comparison_a = st.file_uploader(
    "Sample A",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac"
    ],
    key="comparison_a"
)

comparison_b = st.file_uploader(
    "Sample B",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac"
    ],
    key="comparison_b"
)

if (
    comparison_a is not None
    and comparison_b is not None
):

    if st.button(
        "⚖️ Compare Samples",
        key="compare_samples"
    ):

        prepared_a = prepare_audio(
            comparison_a
        )

        prepared_b = prepare_audio(
            comparison_b
        )

        if (
            prepared_a is not None
            and prepared_b is not None
        ):

            audio_a, sr_a = prepared_a
            audio_b, sr_b = prepared_b

            with st.spinner(
                "Comparing voice samples..."
            ):

                score_a = analyze_audio(
                    audio_a
                )

                score_b = analyze_audio(
                    audio_b
                )

            level_a = get_level(
                score_a
            )

            level_b = get_level(
                score_b
            )

            st.session_state.comparison_result = {
                "A": {
                    "risk": score_a,
                    "level": level_a
                },
                "B": {
                    "risk": score_b,
                    "level": level_b
                }
            }


if st.session_state.comparison_result:

    comparison = (
        st.session_state.comparison_result
    )

    st.subheader(
        "📊 Comparison Result"
    )

    ca, cb = st.columns(2)

    with ca:

        st.metric(
            "👤 Sample A Risk",
            f"{comparison['A']['risk']:.2f}%"
        )

        if comparison["A"]["level"] == "HIGH":
            st.error(
                "HIGH RISK"
            )
        elif comparison["A"]["level"] == "MEDIUM":
            st.warning(
                "MEDIUM RISK"
            )
        else:
            st.success(
                "LOW RISK"
            )

    with cb:

        st.metric(
            "🤖 Sample B Risk",
            f"{comparison['B']['risk']:.2f}%"
        )

        if comparison["B"]["level"] == "HIGH":
            st.error(
                "HIGH RISK"
            )
        elif comparison["B"]["level"] == "MEDIUM":
            st.warning(
                "MEDIUM RISK"
            )
        else:
            st.success(
                "LOW RISK"
            )

    difference = abs(
        comparison["A"]["risk"]
        - comparison["B"]["risk"]
    )

    st.metric(
        "Risk Difference",
        f"{difference:.2f}"
    )

    if (
        comparison["A"]["level"]
        != comparison["B"]["level"]
    ):

        st.success(
            "✅ The two samples show different risk levels."
        )

    else:

        st.warning(
            "⚠️ Both samples fall into the same risk category. "
            "This demonstrates why additional verification may be "
            "needed for difficult audio."
        )


st.divider()


# =========================================================
# FRAUD PREVENTION FLOW
# =========================================================

st.header(
    "🛡️ Fraud Prevention Flow"
)

flow1, flow2, flow3, flow4, flow5 = st.columns(5)

with flow1:
    st.info(
        "1️⃣\n\nDETECT\n\nIncoming Voice"
    )

with flow2:
    st.info(
        "2️⃣\n\nSCORE\n\nAI Risk"
    )

with flow3:
    st.warning(
        "3️⃣\n\nVERIFY\n\nSecurity Check"
    )

with flow4:
    st.warning(
        "4️⃣\n\nDECIDE\n\nAllow / Block"
    )

with flow5:
    st.success(
        "5️⃣\n\nPROTECT\n\nSensitive Action"
    )


st.write(
    "Voice → AI Detection → Risk Score → "
    "Verification → Security Decision → Fraud Prevention"
)


st.divider()


# =========================================================
# DETECTION HISTORY
# =========================================================

st.header(
    "📜 Detection History"
)

if st.session_state.history:

    for item in st.session_state.history:

        if item["Level"] == "HIGH":

            st.error(
                f"{item['Source']} | "
                f"Risk: {item['Risk']:.2f} | "
                f"{item['Level']} | "
                f"{item['Decision']}"
            )

        elif item["Level"] == "MEDIUM":

            st.warning(
                f"{item['Source']} | "
                f"Risk: {item['Risk']:.2f} | "
                f"{item['Level']} | "
                f"{item['Decision']}"
            )

        else:

            st.success(
                f"{item['Source']} | "
                f"Risk: {item['Risk']:.2f} | "
                f"{item['Level']} | "
                f"{item['Decision']}"
            )

else:

    st.info(
        "No voice scans performed yet."
    )


st.divider()


# =========================================================
# SECURITY RECOMMENDATION
# =========================================================

st.header(
    "🔐 Security Recommendation"
)

st.info(
    """
    VoiceShield AI treats voice-cloning detection as a risk signal,
    not as absolute proof of identity.

    For high-risk or uncertain cases:
    
    • Stop sensitive actions
    • Request secondary verification
    • Use trusted callback / OTP / device authentication
    • Never rely on voice alone for high-value authorization
    """
)


st.divider()


# =========================================================
# ABOUT
# =========================================================

st.header(
    "ℹ️ About VoiceShield AI"
)

st.write(
    """
    VoiceShield AI is an AI-powered voice security prototype
    designed to detect potentially AI-generated voice impersonation
    and connect detection with fraud-prevention actions.

    The system uses the Dhwani multilingual deepfake-audio detection
    model with ONNX Runtime and provides risk scoring, challenging-case
    handling, explainable security feedback and secondary verification.

    The prototype is designed for defensive cybersecurity use.
    Detection results should be treated as risk indicators rather than
    forensic or legal proof.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🛡️ VoiceShield AI | AI-powered voice cloning detection "
    "and impersonation fraud prevention"
)
