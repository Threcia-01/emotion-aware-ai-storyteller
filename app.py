"""
app.py — Emotion-Aware Story Friend

Updated fixes in this version:
- Wake word is now polled independently of normal user interaction using
  @st.fragment(run_every=0.8), so wake events surface even when the page is idle.
- Wake word trigger records audio directly through wake_listener.record_audio().
- Sidebar now correctly calls get_last_detected_emotion().
- All other behaviour (typed input, manual st.audio_input, face emotion,
  sidebar controls) remains unchanged.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from rag import (
    index_documents,
    generate_story,
    conversation_history,
    reset_story,
    story_to_audio_file,
    get_last_detected_emotion,
)
from webcam import detect_user_emotion_from_webcam
from voice_input import process_voice_bytes
from wake_word import get_listener

# How many seconds to record after a wake-word event.
WAKE_RECORD_SECS = 5.0

st.set_page_config(
    page_title="Emotion-Aware Story Friend",
    page_icon="🎧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_js_string(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
    )

def announce(msg: str) -> None:
    safe = _safe_js_string(msg)
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            const parentDoc = window.parent.document;
            let region = parentDoc.getElementById("story-friend-live-region");
            if (!region) {{
              region = parentDoc.createElement("div");
              region.id = "story-friend-live-region";
              region.setAttribute("aria-live", "assertive");
              region.setAttribute("aria-atomic", "true");
              region.style.position = "fixed";
              region.style.width = "1px";
              region.style.height = "1px";
              region.style.overflow = "hidden";
              region.style.clipPath = "inset(50%)";
              region.style.whiteSpace = "nowrap";
              parentDoc.body.appendChild(region);
            }}
            region.textContent = `{safe}`;
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )

def autoplay_audio(audio_bytes: bytes) -> None:
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            const audio = new Audio("data:audio/mp3;base64,{b64}");
            audio.autoplay = true;
            audio.play().catch(() => {{}});
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )

def effective_emotion() -> str:
    for key in ("voice_emotion", "user_emotion"):
        value = st.session_state.get(key, "neutral")
        if value and value != "neutral":
            return value
    return "neutral"

def ensure_docs_folder() -> None:
    Path("docs").mkdir(exist_ok=True)

def generate_and_play(user_text: str, voice_emotion: str = "neutral"):
    story, final_emotion = generate_story(
        user_text,
        voice_emotion=voice_emotion,
        camera_emotion=st.session_state.get("user_emotion", "neutral"),
    )

    audio_path = story_to_audio_file(story, final_emotion)
    audio_bytes = Path(audio_path).read_bytes()

    with st.chat_message("assistant"):
        st.markdown(story)

    autoplay_audio(audio_bytes)
    announce(f"Story ready. Emotion detected as {final_emotion}.")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "user_emotion": "neutral",
    "voice_emotion": "neutral",
    "emotion_context": "No emotion detected yet",
    "pending_audio": None,
    "pending_text": None,
    "wake_triggered": False,
    "_last_audio_hash": None,
    "indexed": False,
    "chunk_count": 0,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Startup indexing
# ---------------------------------------------------------------------------

ensure_docs_folder()

if not st.session_state.indexed:
    with st.spinner("Preparing story memory..."):
        count = index_documents("docs")
    st.session_state.indexed = True
    st.session_state.chunk_count = count

# ---------------------------------------------------------------------------
# Wake word listener
# ---------------------------------------------------------------------------

wake_listener = get_listener()
wake_backend = getattr(wake_listener, "_backend", "unavailable")

@st.fragment(run_every=0.8)
def wake_word_poller():
    """
    Poll the wake-word listener independently of normal page interaction.
    If the listener fires, promote it to a full app rerun.
    """
    if wake_listener.triggered() and not st.session_state.get("wake_triggered", False):
        wake_listener.reset()
        st.session_state.wake_triggered = True
        st.rerun()

wake_word_poller()

# ---------------------------------------------------------------------------
# Emoji map
# ---------------------------------------------------------------------------

EMOJI_MAP = {
    "happy": "😊",
    "joy": "😊",
    "amusement": "😄",
    "love": "🥰",
    "fear": "😱",
    "surprise": "😮",
    "neutral": "😐",
    "sad": "😢",
    "sadness": "😢",
    "angry": "😠",
    "anger": "😠",
    "disgust": "🤢",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎧 Emotion-Aware Story Friend")
st.caption('Say "Hey Mycroft" to speak hands-free, or hold **Space** to record')

if wake_backend == "unavailable":
    st.info(
        "👂 **Hands-free tip:** `pip install openwakeword pyaudio` to enable "
        "wake word so the child never needs to touch anything."
    )

# ---------------------------------------------------------------------------
# Emotion controls
# ---------------------------------------------------------------------------

_, col_face, col_voice = st.columns([3, 1, 1])

with col_face:
    if st.button("📷 Read Face", use_container_width=True):
        with st.spinner("Analysing face..."):
            detected = detect_user_emotion_from_webcam(max_frames=15) or "neutral"
            st.session_state.user_emotion = detected
            st.session_state.emotion_context = f"Face emotion: {detected}"
        announce(f"Face emotion detected: {detected}")
        st.rerun()

    fe = st.session_state.user_emotion
    st.caption(f"{EMOJI_MAP.get(fe, '🤔')} Face: **{fe}**")

with col_voice:
    ve = st.session_state.voice_emotion
    st.caption(
        f"{EMOJI_MAP.get(ve, '🎙️')} Voice: **{ve}**"
        if ve and ve != "neutral"
        else "🎙️ Voice: —"
    )

eff = effective_emotion()
if eff != "neutral":
    st.info(f"Active emotion → **{eff}** {EMOJI_MAP.get(eff, '')}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Controls")
    st.write(f"Chunks: {st.session_state.get('chunk_count', 0)}")
    st.write(f"Last emotion: {get_last_detected_emotion()}")
    st.write(f"Wake word: {wake_backend}")

    if st.button("🔄 New story"):
        reset_story()
        st.session_state.user_emotion = "neutral"
        st.session_state.voice_emotion = "neutral"
        st.session_state.pending_audio = None
        st.session_state.pending_text = None
        st.session_state.emotion_context = "No emotion detected yet"
        announce("New story started.")
        st.rerun()

    if st.button("🔨 Rebuild index"):
        with st.spinner("Rebuilding..."):
            st.session_state.chunk_count = index_documents("docs", force_rebuild=True)
        st.rerun()

    st.divider()
    st.caption("Quick stories")

    if st.button("🐰 Sad bunny"):
        st.session_state.pending_text = "I feel sad. Tell me a soft bunny story."
        st.rerun()

    if st.button("🐉 Happy dragon"):
        st.session_state.pending_text = "I feel happy! Tell me a playful dragon story."
        st.rerun()

    if st.button("🌲 Safe bedtime"):
        st.session_state.pending_text = "I feel scared. Tell me a safe forest story."
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Wake-triggered recording block
# ---------------------------------------------------------------------------

if st.session_state.wake_triggered:
    st.session_state.wake_triggered = False

    st.success("🔔 Wake word heard — listening now!")
    announce("Wake word heard. Speak now.")

    components.html(
        """
        <script>
        (function() {
          try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.value = 880;
            gain.gain.value = 0.02;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.18);
          } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )

    with st.spinner(f"Recording for {int(WAKE_RECORD_SECS)} seconds..."):
        raw_bytes = wake_listener.record_audio(duration=WAKE_RECORD_SECS)

    if raw_bytes:
        st.session_state.pending_audio = raw_bytes
    else:
        st.warning("Wake recording failed — microphone unavailable.")

    st.rerun()

# ---------------------------------------------------------------------------
# Voice input
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🎙️ Speak your request")
st.caption("Hold **Space bar** to record • Or say **'Hey Mycroft'** for hands-free")

if hasattr(st, "audio_input"):
    audio_value = st.audio_input(
        "Hold & Speak",
        key="ptt_audio",
        label_visibility="collapsed",
    )

    if audio_value is not None:
        raw_bytes = audio_value.read()
        current_hash = hash(raw_bytes) if raw_bytes else None

        if raw_bytes and current_hash != st.session_state.get("_last_audio_hash"):
            st.session_state._last_audio_hash = current_hash
            st.session_state.pending_audio = raw_bytes
            announce("Recording received. Processing your voice now.")
            st.rerun()
else:
    st.warning("Your Streamlit version does not support st.audio_input. Please upgrade Streamlit.")

# ---------------------------------------------------------------------------
# Typed input
# ---------------------------------------------------------------------------

typed = st.chat_input("Tell me a story about a rabbit, dragon, forest, moon, or anything you like...")
if typed:
    st.session_state.pending_text = typed
    st.rerun()

# ---------------------------------------------------------------------------
# Process pending audio
# ---------------------------------------------------------------------------

if st.session_state.pending_audio is not None:
    audio_bytes = st.session_state.pending_audio
    st.session_state.pending_audio = None

    with st.spinner("Listening carefully..."):
        text, voice_emotion = process_voice_bytes(audio_bytes)

    st.session_state.voice_emotion = voice_emotion or "neutral"

    if not text.strip():
        st.warning("I heard your voice, but I could not understand the words clearly. Please try again.")
        announce("I heard your voice, but I could not understand the words clearly. Please try again.")
    else:
        with st.chat_message("user"):
            st.markdown(text)

        announce(f"You said: {text}")
        generate_and_play(text, voice_emotion=st.session_state.voice_emotion)

    st.stop()

# ---------------------------------------------------------------------------
# Process pending text
# ---------------------------------------------------------------------------

if st.session_state.pending_text:
    user_text = st.session_state.pending_text
    st.session_state.pending_text = None

    if user_text.strip():
        with st.chat_message("user"):
            st.markdown(user_text)

        announce(f"You asked: {user_text}")
        generate_and_play(user_text, voice_emotion=st.session_state.voice_emotion)

    st.stop()