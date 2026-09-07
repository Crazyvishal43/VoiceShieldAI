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
    layout="wide",
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
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .risk-high {
        padding: 18px;
        border-radius: 12px;
        background: #ffe5e5;
        border: 1px solid #ff9999;
    }

    .risk-medium {
        padding: 18px;
        border-radius: 12px;
        background: #fff4d6;
        border: 1px solid #e6c15a;
    }

    .risk-low {
        padding: 18px;
        border-radius: 12px;
        background: #e5f7e8;
        border: 1px solid #8bd49a;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL DOWNLOAD
# =========================================================

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):

        st.info(
            "Downloading AI detection model for first-time setup..."
        )

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH,
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
        providers=["CPUExecutionProvider"],
    )

    return session


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_audio(uploaded_file):

    input_name = "input_audio"
    output_name = "converted.wav"

    try:

        input_bytes = uploaded_file.getvalue()

        with open(input_name, "wb") as f:
            f.write(input_bytes)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_name,
            "-ac",
            "1",
            "-ar",
            "16000",
            output_name,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return None

        if not os.path.exists(output_name):
            return None

        data, sample_rate = sf.read(
            output_name
        )

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        return data, sample_rate

    except Exception:
        return None

    finally:

        try:

            if os.path.exists(input_name):
                os.remove(input_name)

            if os.path.exists(output_name):
                os.remove(output_name)

        except Exception:
            pass


# =========================================================
# PREPARE AUDIO
# =========================================================

def prepare_audio(
    audio,
    sample_rate,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    if sample_rate != 16000:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )

    max_len = 48000

    if len(audio) > max_len:

        audio = audio[:max_len]

    elif len(audio) < max_len:

        audio = np.pad(
            audio,
            (
                0,
                max_len - len(audio),
            ),
            mode="constant",
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
# AI MODEL ANALYSIS
# =========================================================

def analyze_audio(
    audio,
    sample_rate,
):

    session = get_model()

    processed = prepare_audio(
        audio,
        sample_rate,
    )

    input_name = (
        session.get_inputs()[0].name
    )

    logits = session.run(
        None,
        {
            input_name:
            processed.reshape(
                1,
                48000,
            )
        },
    )[0]

    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
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
            keepdims=True,
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
# SEGMENT ANALYSIS
# =========================================================

def segment_analysis(
    audio,
    sample_rate=16000,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    if sample_rate != 16000:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )

        sample_rate = 16000

    segment_length = 3 * 16000
    hop_length = int(
        1.5 * 16000
    )

    scores = []

    start = 0

    while (
        start + segment_length
        <= len(audio)
    ):

        segment = audio[
            start:
            start + segment_length
        ]

        score = analyze_audio(
            segment,
            16000,
        )

        scores.append(
            float(score)
        )

        start += hop_length

    if not scores:

        scores.append(
            float(
                analyze_audio(
                    audio,
                    16000,
                )
            )
        )

    values = np.asarray(
        scores,
        dtype=np.float32,
    )

    return {
        "scores": values.tolist(),
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(values)
        ),
        "minimum": float(
            np.min(values)
        ),
        "maximum": float(
            np.max(values)
        ),
        "range": float(
            np.max(values)
            - np.min(values)
        ),
    }


# =========================================================
# NEW STRICT VOICE DETECTION POLICY
# =========================================================

def get_level(score):

    # Very small AI signal
    if score < 10:

        return "LOW RISK"

    # Any noticeable AI/deepfake signal
    elif score < 70:

        return "MEDIUM RISK"

    # Strong AI/deepfake signal
    else:

        return "HIGH RISK"


def get_decision(level):

    if level == "HIGH RISK":

        return "BLOCK / VERIFY"

    elif level == "MEDIUM RISK":

        return "VERIFY"

    return "ALLOW WITH CAUTION"


# =========================================================
# MICROPHONE SECURITY DECISION
# =========================================================

def microphone_security_decision(
    segment_result,
):

    mean_score = (
        segment_result["mean"]
    )

    median_score = (
        segment_result["median"]
    )

    maximum_score = (
        segment_result["maximum"]
    )

    # Strong AI signal
    if (
        mean_score >= 70
        or maximum_score >= 80
    ):

        return (
            "HIGH RISK",
            "BLOCK / VERIFY",
            "Strong synthetic-voice signal detected "
            "in the microphone recording."
        )

    # Any noticeable AI signal
    if (
        mean_score >= 10
        or median_score >= 10
        or maximum_score >= 20
    ):

        return (
            "MEDIUM RISK",
            "VERIFY",
            "A noticeable synthetic-voice signal was "
            "detected. Additional identity verification "
            "is recommended."
        )

    # Very low AI signal
    return (
        "LOW RISK",
        "ALLOW WITH CAUTION",
        "No strong AI-cloning signal was detected."
    )


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    sample_name,
    score,
    level,
    decision,
):

    st.session_state.history.append(
        {
            "Sample": sample_name,
            "Risk Score": round(
                float(score),
                2,
            ),
            "Risk Level": level,
            "Decision": decision,
        }
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

def show_explainable_ai(
    score,
    level,
):

    st.markdown(
        "### 🔎 Explainable AI"
    )

    if level == "HIGH RISK":

        st.error(
            "🚨 Strong synthetic-voice signal detected."
        )

        st.write(
            "• Synthetic speech signal: HIGH concern"
        )

        st.write(
            "• Acoustic authenticity: LOW confidence"
        )

        st.write(
            "• Speech authenticity: SUSPICIOUS"
        )

        st.write(
            "• Identity confidence: LOW"
        )

        st.write(
            "• Security response: BLOCK / VERIFY"
        )

    elif level == "MEDIUM RISK":

        st.warning(
            "⚠️ Noticeable AI/deepfake signal detected."
        )

        st.write(
            "• Synthetic speech signal: PRESENT"
        )

        st.write(
            "• Acoustic authenticity: NEEDS VERIFICATION"
        )

        st.write(
            "• Speech authenticity: UNCERTAIN"
        )

        st.write(
            "• Identity confidence: MEDIUM"
        )

        st.write(
            "• Security response: VERIFY"
        )

    else:

        st.success(
            "✅ No strong AI-cloning signal detected."
        )

        st.write(
            "• Synthetic speech signal: LOW"
        )

        st.write(
            "• Acoustic authenticity: CONSISTENT"
        )

        st.write(
            "• Speech authenticity: LIKELY GENUINE"
        )

        st.write(
            "• Identity confidence: HIGHER"
        )

        st.write(
            "• Security response: ALLOW WITH CAUTION"
        )

    st.caption(
        "AI percentage represents the model's "
        "deepfake-risk probability, not a literal "
        "percentage of AI content in the recording."
    )


# =========================================================
# RESULT DISPLAY
# =========================================================

def show_analysis_result(
    score,
    level,
    decision,
    title,
):

    st.markdown(
        f"### {title}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "🤖 AI / Deepfake Risk",
            f"{score:.2f}%",
        )

    with c2:

        st.metric(
            "Risk Level",
            level,
        )

    with c3:

        st.metric(
            "Security Decision",
            decision,
        )

    if level == "HIGH RISK":

        st.error(
            "🚨 Potential voice cloning detected."
        )

    elif level == "MEDIUM RISK":

        st.warning(
            "⚠️ AI/deepfake signal detected. "
            "Additional verification recommended."
        )

    else:

        st.success(
            "✅ No strong AI-cloning signal detected."
        )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛡️ VoiceShield AI"
)

st.caption(
    "AI-powered voice cloning detection and "
    "impersonation fraud prevention system"
)


# =========================================================
# SYSTEM STATUS
# =========================================================

s1, s2, s3 = st.columns(3)

with s1:

    st.success(
        "🟢 SYSTEM STATUS\n\nONLINE"
    )

with s2:

    st.info(
        "🤖 AI ENGINE\n\nREADY"
    )

with s3:

    st.success(
        "🔐 PRIVACY MODE\n\nLOCAL ANALYSIS"
    )


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
    for x in st.session_state.history
    if x["Risk Level"] == "HIGH RISK"
)

medium_count = sum(
    1
    for x in st.session_state.history
    if x["Risk Level"] == "MEDIUM RISK"
)

low_count = sum(
    1
    for x in st.session_state.history
    if x["Risk Level"] == "LOW RISK"
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "📊 Total Scans",
        total_scans,
    )

with m2:

    st.metric(
        "🔴 High Risk",
        high_count,
    )

with m3:

    st.metric(
        "🟡 Medium Risk",
        medium_count,
    )

with m4:

    st.metric(
        "🟢 Low Risk",
        low_count,
    )


# =========================================================
# SPEAK & DETECT
# =========================================================

st.markdown(
    "## 🎤 Speak & Detect"
)

st.write(
    "Record a voice sample using your microphone "
    "and analyze it for potential AI-generated speech."
)

recorded_audio = st.audio_input(
    "🎙️ Record your voice"
)

if recorded_audio is not None:

    st.audio(
        recorded_audio
    )

    if st.button(
        "🔍 Analyze Recorded Voice",
        key="analyze_recorded_voice",
        use_container_width=True,
    ):

        try:

            audio_bytes = (
                recorded_audio.getvalue()
            )

            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes)
            )

            audio_data = np.asarray(
                audio_data,
                dtype=np.float32,
            )

            if audio_data.ndim > 1:

                audio_data = np.mean(
                    audio_data,
                    axis=1,
                )

            if sample_rate != 16000:

                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000,
                )

                sample_rate = 16000

            with st.spinner(
                "AI is analyzing multiple voice segments..."
            ):

                segment_result = (
                    segment_analysis(
                        audio_data,
                        sample_rate,
                    )
                )

            score = (
                segment_result["mean"]
            )

            level, decision, reason = (
                microphone_security_decision(
                    segment_result
                )
            )

            st.session_state.recorded_result = {
                "score": score,
                "level": level,
                "decision": decision,
                "segments": segment_result,
            }

            save_history(
                "Microphone Recording",
                score,
                level,
                decision,
            )

            show_analysis_result(
                score,
                level,
                decision,
                "🎤 Recorded Voice Analysis",
            )

            st.info(
                reason
            )

            with st.expander(
                "📡 Detection Evidence"
            ):

                st.write(
                    f"Segments analyzed: "
                    f"{len(segment_result['scores'])}"
                )

                st.write(
                    f"Average AI risk: "
                    f"{segment_result['mean']:.2f}%"
                )

                st.write(
                    f"Median AI risk: "
                    f"{segment_result['median']:.2f}%"
                )

                st.write(
                    f"Minimum segment risk: "
                    f"{segment_result['minimum']:.2f}%"
                )

                st.write(
                    f"Maximum segment risk: "
                    f"{segment_result['maximum']:.2f}%"
                )

                st.write(
                    f"Segment variation: "
                    f"{segment_result['std']:.2f}"
                )

            show_explainable_ai(
                score,
                level,
            )

        except Exception as e:

            st.error(
                f"Microphone analysis failed: {e}"
            )


# =========================================================
# VOICE DETECTION
# =========================================================

st.markdown(
    "## 🔊 Voice Detection"
)

st.write(
    "Upload an audio file for AI voice-cloning analysis."
)

sample_type = st.selectbox(
    "Sample Type",
    [
        "Unknown",
        "Human",
        "AI-generated",
        "Hybrid",
    ],
)

uploaded_file = st.file_uploader(
    "Upload voice sample",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac",
    ],
    key="voice_upload",
)

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    if st.button(
        "🔍 Analyze Uploaded Voice",
        key="analyze_uploaded_voice",
        use_container_width=True,
    ):

        with st.spinner(
            "AI is analyzing the uploaded voice..."
        ):

            converted = convert_audio(
                uploaded_file
            )

        if converted is None:

            st.error(
                "Unable to process the audio file. "
                "Please check FFmpeg."
            )

        else:

            audio_data, sample_rate = converted

            try:

                score = analyze_audio(
                    audio_data,
                    sample_rate,
                )

                level = get_level(
                    score
                )

                decision = get_decision(
                    level
                )

                st.session_state.upload_result = {
                    "score": score,
                    "level": level,
                    "decision": decision,
                    "sample_type": sample_type,
                }

                save_history(
                    uploaded_file.name,
                    score,
                    level,
                    decision,
                )

                show_analysis_result(
                    score,
                    level,
                    decision,
                    "🔊 Voice Detection Result",
                )

                # -------------------------------------------------
                # HUMAN / AI / HYBRID INFORMATION
                # -------------------------------------------------

                human_probability = max(
                    0,
                    100 - score,
                )

                st.markdown(
                    "### 📊 Voice Composition Indicator"
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "👤 Human-like Probability",
                        f"{human_probability:.2f}%",
                    )

                with p2:

                    st.metric(
                        "🤖 AI / Deepfake Risk",
                        f"{score:.2f}%",
                    )

                if sample_type == "Hybrid":

                    st.warning(
                        "🟡 CHALLENGING HYBRID SAMPLE"
                    )

                    st.write(
                        "This sample is marked as Hybrid "
                        "for demonstration purposes."
                    )

                    st.write(
                        "The original model score is preserved. "
                        "Because mixed human-AI audio can be difficult "
                        "to classify reliably, additional verification "
                        "is recommended."
                    )

                show_explainable_ai(
                    score,
                    level,
                )

            except Exception as e:

                st.error(
                    f"Voice analysis failed: {e}"
                )


# =========================================================
# INCOMING VOICE SECURITY GATE
# =========================================================

st.markdown(
    "## 📞 Incoming Voice Security Gate"
)

st.write(
    "Scan an incoming caller voice before approving "
    "a sensitive action."
)

incoming_file = st.file_uploader(
    "Upload incoming caller voice",
    type=[
        "wav",
        "mp3",
        "m4a",
        "mpeg",
        "ogg",
        "flac",
    ],
    key="incoming_voice",
)

if incoming_file is not None:

    st.audio(
        incoming_file
    )

    if st.button(
        "🚨 Scan Incoming Voice",
        key="scan_incoming_voice",
        use_container_width=True,
    ):

        with st.spinner(
            "Scanning incoming voice..."
        ):

            converted = convert_audio(
                incoming_file
            )

        if converted is None:

            st.error(
                "Unable to process incoming voice."
            )

        else:

            audio_data, sample_rate = converted

            try:

                score = analyze_audio(
                    audio_data,
                    sample_rate,
                )

                level = get_level(
                    score
                )

                decision = get_decision(
                    level
                )

                st.session_state.incoming_result = {
                    "score": score,
                    "level": level,
                    "decision": decision,
                }

                save_history(
                    "Incoming Voice",
                    score,
                    level,
                    decision,
                )

                show_analysis_result(
                    score,
                    level,
                    decision,
                    "📞 Incoming Voice Analysis",
                )

                if level == "HIGH RISK":

                    st.error(
                        "🛑 Sensitive Action Automatically Stopped"
                    )

                    st.warning(
                        "🔐 Identity Verification Required"
                    )

                    verification_method = st.selectbox(
                        "Choose Secondary Verification",
                        [
                            "OTP Verification",
                            "Trusted Callback",
                            "Device Authentication",
                            "In-App Confirmation",
                            "Human Verification",
                        ],
                        key="verification_method",
                    )

                    if st.button(
                        "🔐 Verify Identity",
                        key="verify_identity",
                    ):

                        st.success(
                            f"Prototype verification initiated "
                            f"using: {verification_method}"
                        )

                        st.info(
                            "This is a prototype simulation. "
                            "No real OTP or authentication service "
                            "is connected."
                        )

                elif level == "MEDIUM RISK":

                    st.warning(
                        "⚠️ Sensitive action requires "
                        "additional identity verification."
                    )

                else:

                    st.success(
                        "✅ No strong AI-cloning signal detected. "
                        "Continue with caution."
                    )

                show_explainable_ai(
                    score,
                    level,
                )

            except Exception as e:

                st.error(
                    f"Incoming voice analysis failed: {e}"
                )


# =========================================================
# HUMAN VS AI COMPARISON
# =========================================================

st.markdown(
    "## 👤 Human vs 🤖 AI Comparison"
)

ca, cb = st.columns(2)

with ca:

    st.markdown(
        "### 👤 Sample A — Human"
    )

    human_file = st.file_uploader(
        "Upload human sample",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mpeg",
            "ogg",
            "flac",
        ],
        key="human_sample",
    )

with cb:

    st.markdown(
        "### 🤖 Sample B — AI"
    )

    ai_file = st.file_uploader(
        "Upload AI sample",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mpeg",
            "ogg",
            "flac",
        ],
        key="ai_sample",
    )


if st.button(
    "⚖️ Compare Human vs AI",
    key="compare_samples",
    use_container_width=True,
):

    if (
        human_file is None
        or ai_file is None
    ):

        st.warning(
            "Please upload both Human and AI samples."
        )

    else:

        with st.spinner(
            "Comparing voice samples..."
        ):

            human_converted = convert_audio(
                human_file
            )

            ai_converted = convert_audio(
                ai_file
            )

        if (
            human_converted is None
            or ai_converted is None
        ):

            st.error(
                "Unable to process one or both files."
            )

        else:

            try:

                human_audio, human_sr = (
                    human_converted
                )

                ai_audio, ai_sr = (
                    ai_converted
                )

                human_score = analyze_audio(
                    human_audio,
                    human_sr,
                )

                ai_score = analyze_audio(
                    ai_audio,
                    ai_sr,
                )

                human_level = get_level(
                    human_score
                )

                ai_level = get_level(
                    ai_score
                )

                difference = abs(
                    ai_score
                    - human_score
                )

                st.session_state.comparison_result = {
                    "A": {
                        "risk": human_score,
                        "level": human_level,
                    },
                    "B": {
                        "risk": ai_score,
                        "level": ai_level,
                    },
                    "difference": difference,
                }

                st.markdown(
                    "### 📊 Risk Score Comparison"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "👤 Human Risk",
                        f"{human_score:.2f}%",
                    )

                    st.progress(
                        min(
                            max(
                                int(
                                    human_score
                                ),
                                0,
                            ),
                            100,
                        )
                    )

                with c2:

                    st.metric(
                        "🤖 AI Risk",
                        f"{ai_score:.2f}%",
                    )

                    st.progress(
                        min(
                            max(
                                int(
                                    ai_score
                                ),
                                0,
                            ),
                            100,
                        )
                    )

                st.metric(
                    "Risk Difference",
                    f"{difference:.2f}",
                )

                if (
                    human_level
                    != ai_level
                ):

                    st.success(
                        "✅ The two samples received "
                        "different risk classifications."
                    )

                else:

                    st.warning(
                        "⚠️ Both samples received the same "
                        "classification. This demonstrates "
                        "that recording conditions can affect "
                        "deepfake detection."
                    )

            except Exception as e:

                st.error(
                    f"Comparison failed: {e}"
                )


# =========================================================
# FRAUD PREVENTION FLOW
# =========================================================

st.markdown(
    "## 🛡️ Fraud Prevention Flow"
)

f1, f2, f3, f4, f5 = st.columns(5)

with f1:

    st.info(
        "📞\n\nIncoming Voice"
    )

with f2:

    st.info(
        "🤖\n\nAI Detection"
    )

with f3:

    st.info(
        "📊\n\nRisk Score"
    )

with f4:

    st.warning(
        "🔐\n\nVerify"
    )

with f5:

    st.success(
        "✅ / 🛑\n\nAllow / Block"
    )

st.write(
    "**Workflow:** Incoming Voice → AI Detection → "
    "Risk Score → Security Decision → Secondary "
    "Verification → Allow / Block"
)


# =========================================================
# DETECTION HISTORY
# =========================================================

st.markdown(
    "## 📜 Detection History"
)

if st.session_state.history:

    for index, item in enumerate(
        reversed(
            st.session_state.history
        ),
        start=1,
    ):

        st.write(
            f"**{index}. {item['Sample']}** — "
            f"{item['Risk Score']:.2f}/100 — "
            f"{item['Risk Level']} — "
            f"{item['Decision']}"
        )

else:

    st.caption(
        "No scans recorded yet."
    )


# =========================================================
# SECURITY RECOMMENDATION
# =========================================================

st.markdown(
    "## 🔐 Security Recommendation"
)

st.write(
    "VoiceShield AI treats voice detection as a "
    "security risk signal rather than absolute proof "
    "of identity."
)

st.write(
    "Any noticeable AI/deepfake signal should trigger "
    "additional verification before sensitive actions."
)

st.write(
    "High-risk voice activity should automatically "
    "stop the sensitive action and require secondary "
    "identity verification."
)


# =========================================================
# ABOUT
# =========================================================

st.markdown(
    "## ℹ️ About VoiceShield AI"
)

st.write(
    "VoiceShield AI is an AI-powered voice-cloning "
    "detection and impersonation-fraud prevention prototype."
)

st.write(
    "The system analyzes voice input, generates a "
    "0–100 deepfake-risk score and converts that signal "
    "into a security decision."
)

st.write(
    "The prototype uses the Dhwani multilingual "
    "deepfake-audio detection model with ONNX Runtime."
)

st.write(
    "Detection performance can vary with recording "
    "conditions, speaker-to-microphone audio, noise, "
    "compression, accents, unseen cloning methods and "
    "mixed human-AI recordings."
)


# =========================================================
# RESPONSIBLE USE
# =========================================================

st.markdown(
    "## ⚠️ Responsible Use"
)

st.caption(
    "VoiceShield AI is a prototype security-assistance "
    "system. Results should not be treated as definitive "
    "forensic proof or the sole basis for high-impact decisions."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🛡️ VoiceShield AI | AI-powered voice cloning detection "
    "and impersonation fraud prevention"
)
