"""
voice_input.py — robust voice processing with Azure STT + Whisper fallback.

Fix in this version:
- SpeechBrain WinError 1314 on Windows: foreign_class tries to create a symlink
  for custom_interface.py which requires Developer Mode or admin rights.
  Workaround: manually copy the file from the HuggingFace cache to the savedir
  before calling foreign_class, so no symlink is needed.
"""

import io
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
SILENCE_THRESH = 0.008
SILENCE_PAD = 0.25
MIN_AUDIO_SECS = 0.5

_sb_classifier = None
_sb_lock = threading.Lock()

_SB_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "sad": "sadness",
    "ang": "anger",
    "neutral": "neutral",
    "happy": "happy",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
}

# Where SpeechBrain will store the model locally
_SB_SAVEDIR = "pretrained_models/emotion-wav2vec2"
_SB_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"


def _fix_symlink_on_windows() -> None:
    """
    On Windows without Developer Mode, os.symlink raises WinError 1314.
    SpeechBrain's foreign_class symlinks custom_interface.py from the HF cache
    into the savedir. We copy it manually so no symlink is needed.
    """
    if os.name != "nt":
        return  # Only needed on Windows

    savedir = Path(_SB_SAVEDIR)
    savedir.mkdir(parents=True, exist_ok=True)
    dest = savedir / "custom_interface.py"
    if dest.exists():
        return  # Already there, nothing to do

    # Find the file in the HuggingFace hub cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = _SB_SOURCE.replace("/", "--")
    candidates = list(hf_cache.glob(f"models--{model_slug}/snapshots/*/custom_interface.py"))

    if candidates:
        src = candidates[0]
        print(f"📋 Copying custom_interface.py from HF cache to savedir (Windows symlink workaround)")
        shutil.copy2(src, dest)
    else:
        print("⚠️ custom_interface.py not found in HF cache — SpeechBrain will download it first run")


def _load_speechbrain():
    global _sb_classifier
    if _sb_classifier is not None:
        return _sb_classifier

    with _sb_lock:
        if _sb_classifier is not None:
            return _sb_classifier
        try:
            from speechbrain.inference.interfaces import foreign_class

            _fix_symlink_on_windows()

            print("⏳ Loading SpeechBrain emotion model (~380 MB first run)...")
            _sb_classifier = foreign_class(
                source=_SB_SOURCE,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=_SB_SAVEDIR,
                run_opts={"device": "cpu"},
            )
            print("✅ SpeechBrain ready")
        except Exception as e:
            print(f"⚠️ SpeechBrain load failed: {e}")
            _sb_classifier = None
        return _sb_classifier


def detect_emotion_from_audio(wav_path: str) -> str:
    classifier = _load_speechbrain()
    if classifier is None:
        return "neutral"

    try:
        _, score, _, text_lab = classifier.classify_file(wav_path)
        raw = text_lab[0].lower() if text_lab else "neu"
        label = _SB_LABEL_MAP.get(raw, "neutral")
        try:
            conf = float(score)
            print(f"🎙️ Voice emotion: {label} (raw={raw}, conf={conf:.3f})")
        except Exception:
            print(f"🎙️ Voice emotion: {label} (raw={raw})")
        return label
    except Exception as e:
        print(f"⚠️ Voice emotion failed: {e}")
        return "neutral"


def _bytes_to_numpy(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    buf = io.BytesIO(audio_bytes)

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
        return data, sr
    except Exception:
        pass

    try:
        from pydub import AudioSegment

        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
        arr /= 2 ** (seg.sample_width * 8 - 1)
        if seg.channels > 1:
            arr = arr.reshape(-1, seg.channels).mean(axis=1)
        return arr, seg.frame_rate
    except Exception as e:
        raise RuntimeError(f"Could not decode audio bytes: {e}") from e


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    length = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, length),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if getattr(audio, "ndim", 1) == 2:
        return audio.mean(axis=1)
    return audio


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio
    window = max(1, int(0.1 * SAMPLE_RATE))
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(window) / window, "same"))
    active = np.where(rms > SILENCE_THRESH)[0]
    if len(active) == 0:
        return audio
    pad = int(SILENCE_PAD * SAMPLE_RATE)
    start = max(0, active[0] - pad)
    end = min(len(audio), active[-1] + pad)
    return audio[start:end]


def prepare_audio(audio_bytes: bytes) -> Tuple[np.ndarray, str]:
    audio, sr = _bytes_to_numpy(audio_bytes)
    audio = _to_mono(audio)
    audio = _resample(audio, sr)
    audio = _trim_silence(audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
    tmp.close()
    return audio, tmp.name


def transcribe_azure(wav_path: str) -> str:
    key = os.getenv("AZURE_SPEECH_KEY", "")
    region = os.getenv("AZURE_SPEECH_REGION", "centralindia")

    if not key:
        print("⚠️ AZURE_SPEECH_KEY not set — Azure STT unavailable")
        return ""

    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        print("⚠️ azure-cognitiveservices-speech not installed")
        return ""

    try:
        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
        cfg.speech_recognition_language = "en-IN"
        cfg.set_service_property(
            "punctuation",
            "explicit",
            speechsdk.ServicePropertyChannel.UriQueryParameter,
        )

        audio_cfg = speechsdk.audio.AudioConfig(filename=wav_path)
        recogniser = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
        result = recogniser.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text.strip()
            print(f"📝 Azure STT [{region}]: {text!r}")
            return text

        if result.reason == speechsdk.ResultReason.NoMatch:
            print("⚠️ Azure STT: no speech recognised")
        elif result.reason == speechsdk.ResultReason.Canceled:
            d = speechsdk.CancellationDetails(result)
            print(f"❌ Azure STT cancelled: {d.reason} — {d.error_details}")
        return ""
    except Exception as e:
        print(f"❌ Azure STT crashed: {e}")
        return ""


def transcribe_whisper(wav_path: str) -> str:
    try:
        import whisper
    except ImportError:
        print("⚠️ Whisper not installed — fallback unavailable")
        return ""

    model_name = os.getenv("WHISPER_MODEL", "base")
    try:
        print(f"⏳ Whisper STT loading model: {model_name}")
        model = whisper.load_model(model_name)
        result = model.transcribe(wav_path, language="en", fp16=False)
        text = (result.get("text") or "").strip()
        if text:
            print(f"📝 Whisper STT: {text!r}")
        else:
            print("⚠️ Whisper STT: empty transcription")
        return text
    except Exception as e:
        print(f"❌ Whisper STT failed: {e}")
        return ""


def transcribe_speech(wav_path: str) -> str:
    text = transcribe_azure(wav_path)
    if text:
        return text
    print("⚠️ Falling back to Whisper STT")
    return transcribe_whisper(wav_path)


def process_voice_bytes(audio_bytes: bytes) -> Tuple[str, str]:
    if not audio_bytes:
        return "", "neutral"

    try:
        audio, wav_path = prepare_audio(audio_bytes)
    except Exception as e:
        print(f"❌ Audio preparation failed: {e}")
        return "", "neutral"

    if len(audio) < SAMPLE_RATE * MIN_AUDIO_SECS:
        print(f"⚠️ Recording too short ({len(audio)/SAMPLE_RATE:.2f}s) — skipping")
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        return "", "neutral"

    text_result = ""
    emotion_result = "neutral"

    def _stt():
        nonlocal text_result
        text_result = transcribe_speech(wav_path)

    def _ser():
        nonlocal emotion_result
        emotion_result = detect_emotion_from_audio(wav_path)

    t1 = threading.Thread(target=_stt, daemon=True)
    t2 = threading.Thread(target=_ser, daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()

    try:
        os.unlink(wav_path)
    except OSError:
        pass

    print(f"🎯 Voice pipeline → text={text_result!r} emotion={emotion_result}")
    return text_result, emotion_result