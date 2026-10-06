"""
Cocktail Party Problem — Streamlit Demo App

Orchestrates the full pipeline:
  1. Upload/select mixture → Module 1 (Separation)
  2. Speaker analysis → Module 2 (speaker embeddings, per-stream timeline, diarization)
  3. Convert voice + pitch modulation → Module 3 (Voice Conversion)

Run with: streamlit run app.py
"""

import io
import os
import sys

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.separation import SAMPLE_RATE, separate_waveform
from models.diarization import diarize, get_embedding, speech_activity
from models.voice_conversion import convert, modulate

SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'samples')
RESULT_KEYS = ['stream1', 'stream2', 'stream_sim', 'activity', 'diar_input',
               'converted', 'conv_metrics', 'pitched']

# ═══════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="Cocktail Party Problem — Speech Pipeline Demo",
                   page_icon="🎤", layout="wide")
st.title("🎤 Cocktail Party Problem")
st.markdown("**Speech Separation → Speaker Diarization → Voice Conversion**")
st.markdown("---")


# ═══════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════

def plot_waveform(audio, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(np.arange(len(audio)) / sr, audio, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(audio) / sr)
    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sr, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 3))
    S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80), ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_timeline(rows, duration, title):
    """rows: dict label -> list of (start, end)."""
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.5 * len(rows)))
    colors = plt.cm.tab10.colors
    for i, (label, spans) in enumerate(rows.items()):
        ax.broken_barh([(s, e - s) for s, e in spans], (i - 0.35, 0.7), color=colors[i % 10])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows.keys()))
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def activity_spans(audio, sr, min_len=0.1):
    """VAD frames -> list of (start, end) speech spans."""
    vad = speech_activity(audio, sr)
    spans, start = [], None
    for i, v in enumerate(np.append(vad, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) * 0.01 >= min_len:
                spans.append((start * 0.01, i * 0.01))
            start = None
    return spans


def audio_to_bytes(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sr, format='WAV')
    buffer.seek(0)
    return buffer


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_step(label, fn, *args, **kwargs):
    """Run a model call, turning exceptions (e.g. missing checkpoints) into UI errors."""
    try:
        with st.spinner(label):
            return fn(*args, **kwargs)
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    return None


# ═══════════════════════════════════════════════════════════
# Sidebar: File Selection
# ═══════════════════════════════════════════════════════════

st.sidebar.header("📁 Input Audio")
input_method = st.sidebar.radio("Select input method:", ["Use sample", "Upload file"])

mixture, input_id = None, None
if input_method == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload a mixture audio file", type=['wav', 'flac', 'mp3'])
    if uploaded:
        mixture, _ = librosa.load(io.BytesIO(uploaded.getvalue()), sr=SAMPLE_RATE)
        input_id = f"upload:{uploaded.name}:{uploaded.size}"
else:
    sample_files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.wav', '.flac'))) \
        if os.path.isdir(SAMPLES_DIR) else []
    if sample_files:
        selected = st.sidebar.selectbox("Choose a sample:", sample_files)
        mixture, _ = librosa.load(os.path.join(SAMPLES_DIR, selected), sr=SAMPLE_RATE)
        input_id = f"sample:{selected}"
    else:
        st.sidebar.warning("No sample files found in samples/ — run scripts/prepare_data.py")

st.sidebar.markdown("---")
st.sidebar.caption("`sample_mixture_*` — two overlapping speakers (for separation).  \n"
                   "`sample_conversation_*` — two speakers taking turns (for diarization).")

# New input → clear every downstream result
if input_id != st.session_state.get('input_id'):
    for k in RESULT_KEYS:
        st.session_state.pop(k, None)
    st.session_state['input_id'] = input_id

state = st.session_state


# ═══════════════════════════════════════════════════════════
# Step 1: Speech Separation
# ═══════════════════════════════════════════════════════════

st.header("Step 1: Speech Separation (Conv-TasNet)")

if mixture is None:
    st.info("👈 Upload or select an audio file to begin.")
    st.stop()

duration = len(mixture) / SAMPLE_RATE
st.subheader(f"Input ({duration:.1f} s)")
st.audio(audio_to_bytes(mixture, SAMPLE_RATE), format='audio/wav')
col1, col2 = st.columns(2)
with col1:
    show_figure(plot_waveform(mixture, SAMPLE_RATE, "Input Waveform"))
with col2:
    show_figure(plot_spectrogram(mixture, SAMPLE_RATE, "Input Spectrogram"))

if st.button("🔀 Separate Speakers", key="separate_btn"):
    result = run_step("Running Conv-TasNet separation...", separate_waveform, mixture)
    if result is not None:
        for k in RESULT_KEYS:
            state.pop(k, None)
        state['stream1'], state['stream2'] = result

if 'stream1' in state:
    st.subheader("Separated Streams")
    col1, col2 = st.columns(2)
    for col, key, name in ((col1, 'stream1', "Stream 1"), (col2, 'stream2', "Stream 2")):
        with col:
            st.markdown(f"**{name}**")
            st.audio(audio_to_bytes(state[key], SAMPLE_RATE), format='audio/wav')
            show_figure(plot_spectrogram(state[key], SAMPLE_RATE, name))


# ═══════════════════════════════════════════════════════════
# Step 2: Speaker Analysis / Diarization
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.header("Step 2: Speaker Diarization (ECAPA-TDNN)")

tab_streams, tab_input = st.tabs(["Separated streams (overlapping speech)",
                                  "Diarize input (turn-taking speech)"])

with tab_streams:
    st.caption("After separation each stream holds one speaker, so *who spoke when* is each "
               "stream's speech activity. Speaker embeddings check that the two streams are "
               "really different voices.")
    if 'stream1' not in state:
        st.info("Run separation first (Step 1).")
    else:
        if st.button("🏷️ Analyse Speakers", key="analyse_btn"):
            def analyse():
                e1 = get_embedding(state['stream1'], SAMPLE_RATE)
                e2 = get_embedding(state['stream2'], SAMPLE_RATE)
                return cosine(e1, e2), {
                    "Speaker A (stream 1)": activity_spans(state['stream1'], SAMPLE_RATE),
                    "Speaker B (stream 2)": activity_spans(state['stream2'], SAMPLE_RATE)}
            result = run_step("Extracting speaker embeddings...", analyse)
            if result is not None:
                state['stream_sim'], state['activity'] = result

        if 'activity' in state:
            show_figure(plot_timeline(state['activity'], duration, "Speech activity per speaker"))
            sim = state['stream_sim']
            st.metric("Cosine similarity between stream speakers", f"{sim:.3f}")
            if sim > 0.6:
                st.warning("The streams sound like the same speaker — separation may have failed "
                           "or the input contains only one voice.")
            else:
                st.success("The two streams are distinct speakers.")

with tab_input:
    st.caption("Clusters sliding-window embeddings of the original input. Assumes one speaker "
               "at a time — try `sample_conversation_0000.wav`.")
    num_spk = st.number_input("Number of speakers", min_value=1, max_value=6, value=2)
    if st.button("🏷️ Diarize Input", key="diarize_btn"):
        result = run_step("Running diarization...", diarize, mixture, SAMPLE_RATE, int(num_spk))
        if result is not None:
            state['diar_input'] = result

    if 'diar_input' in state:
        rows = {}
        for spk, start, end in state['diar_input']:
            rows.setdefault(spk, []).append((start, end))
        if rows:
            show_figure(plot_timeline(dict(sorted(rows.items())), duration, "Diarization of input"))
            with st.expander("Segments"):
                for spk, start, end in state['diar_input']:
                    st.write(f"{spk}: {start:.2f}s — {end:.2f}s")
        else:
            st.warning("No speech detected.")


# ═══════════════════════════════════════════════════════════
# Step 3: Voice Conversion + Pitch Modulation
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.header("Step 3: Voice Conversion (AutoVC + HiFi-GAN)")

if 'stream1' not in state:
    st.info("Run separation first (Step 1).")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    source_stream = st.selectbox("Source stream:", ["Stream 1", "Stream 2"])
with col2:
    target_uploaded = st.file_uploader("Target voice reference (optional; defaults to the "
                                       "other stream)", type=['wav', 'flac'], key="target_upload")

if st.button("🎭 Convert Voice", key="convert_btn"):
    source = state['stream1'] if source_stream == "Stream 1" else state['stream2']
    if target_uploaded:
        target_ref, _ = librosa.load(io.BytesIO(target_uploaded.getvalue()), sr=SAMPLE_RATE)
    else:
        target_ref = state['stream2'] if source_stream == "Stream 1" else state['stream1']

    converted = run_step("Running voice conversion...", convert, source, target_ref, SAMPLE_RATE)
    if converted is not None:
        e_conv, e_tgt, e_src = (get_embedding(x, SAMPLE_RATE) for x in (converted, target_ref, source))
        state['converted'] = converted
        state['conv_metrics'] = (cosine(e_conv, e_tgt), cosine(e_conv, e_src), cosine(e_src, e_tgt))
        state.pop('pitched', None)

if 'converted' in state:
    st.subheader("Converted Audio")
    st.audio(audio_to_bytes(state['converted'], SAMPLE_RATE), format='audio/wav')
    to_tgt, to_src, src_tgt = state['conv_metrics']
    c1, c2, c3 = st.columns(3)
    c1.metric("Similarity → target", f"{to_tgt:.3f}", f"{to_tgt - src_tgt:+.3f} vs. source voice")
    c2.metric("Similarity → source", f"{to_src:.3f}")
    c3.metric("Source ↔ target (before)", f"{src_tgt:.3f}")
    show_figure(plot_spectrogram(state['converted'], SAMPLE_RATE, "Converted"))

# ─── Pitch Modulation ───
st.subheader("🎵 Pitch Modulation")
pitch_options = (["Converted output"] if 'converted' in state else []) + ["Stream 1", "Stream 2"]
pitch_source = st.selectbox("Audio to modulate:", pitch_options)
pitch_factor = st.slider("Pitch shift (semitones):", -12, 12, 0)

if st.button("🎶 Apply Pitch Shift", key="pitch_btn"):
    audio_to_shift = {"Converted output": state.get('converted'),
                      "Stream 1": state['stream1'], "Stream 2": state['stream2']}[pitch_source]
    shifted = run_step("Applying pitch shift...", modulate, audio_to_shift, SAMPLE_RATE, float(pitch_factor))
    if shifted is not None:
        state['pitched'] = (pitch_source, pitch_factor, shifted)

if 'pitched' in state:
    src_name, factor, shifted = state['pitched']
    st.caption(f"{src_name}, shifted {factor:+d} semitones")
    st.audio(audio_to_bytes(shifted, SAMPLE_RATE), format='audio/wav')


# ═══════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("**Cocktail Party Problem — Phase 1** | Trained on LibriSpeech dev-clean")
