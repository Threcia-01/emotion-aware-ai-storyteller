import asyncio
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI
from transformers import pipeline

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment or .env file")

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.3-70b-instruct"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "storyteller_docs"
SKIP_FILES = {"dialogs.txt"}
MAX_HISTORY = 6

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=60.0,
    max_retries=1,
)

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

conversation_history: List[Dict[str, str]] = []
_last_detected_emotion = "neutral"

VOICE_PROFILES: Dict[str, dict] = {
    "happy": {"voice": "en-US-JennyNeural", "style": None, "rate": "+15%", "pitch": "+10%", "volume": "loud", "sentence_break": "200ms"},
    "joy": {"voice": "en-US-JennyNeural", "style": None, "rate": "+15%", "pitch": "+10%", "volume": "loud", "sentence_break": "200ms"},
    "amusement": {"voice": "en-US-JennyNeural", "style": None, "rate": "+18%", "pitch": "+12%", "volume": "x-loud", "sentence_break": "180ms"},
    "love": {"voice": "en-US-AvaNeural", "style": "empathetic", "style_degree": 1.3, "rate": "-8%", "pitch": "+4%", "volume": "soft", "sentence_break": "450ms"},
    "neutral": {"voice": "en-IN-NeerjaNeural", "style": None, "rate": "-5%", "pitch": "0%", "volume": "medium", "sentence_break": "380ms"},
    "surprise": {"voice": "en-US-JennyNeural", "style": None, "rate": "+8%", "pitch": "+8%", "volume": "medium", "sentence_break": "320ms"},
    "sad": {"voice": "en-US-AvaNeural", "style": "sad", "style_degree": 1.5, "rate": "-22%", "pitch": "-6%", "volume": "soft", "sentence_break": "650ms"},
    "sadness": {"voice": "en-US-AvaNeural", "style": "sad", "style_degree": 1.5, "rate": "-22%", "pitch": "-6%", "volume": "soft", "sentence_break": "650ms"},
    "fear": {"voice": "en-US-AvaNeural", "style": "empathetic", "style_degree": 1.8, "rate": "-28%", "pitch": "-8%", "volume": "x-soft", "sentence_break": "750ms"},
    "anger": {"voice": "en-US-AvaNeural", "style": "calm", "style_degree": 1.5, "rate": "-18%", "pitch": "-5%", "volume": "medium", "sentence_break": "520ms"},
    "angry": {"voice": "en-US-AvaNeural", "style": "calm", "style_degree": 1.5, "rate": "-18%", "pitch": "-5%", "volume": "medium", "sentence_break": "520ms"},
    "disgust": {"voice": "en-IN-NeerjaNeural", "style": None, "rate": "-12%", "pitch": "0%", "volume": "medium", "sentence_break": "420ms"},
}

STYLE_MAP = {
    "happy": "Use a bright, playful, cheerful tone.",
    "joy": "Use a bright, playful, cheerful tone.",
    "amusement": "Use a light, bouncy, funny tone.",
    "love": "Use a warm, affectionate, cozy tone.",
    "neutral": "Use a balanced, imaginative storytelling tone.",
    "surprise": "Use wonder and curiosity without making it scary.",
    "sad": "Use a comforting, reassuring, hopeful tone.",
    "sadness": "Use a comforting, reassuring, hopeful tone.",
    "fear": "Use a calm, safe, soothing tone and reassure often.",
    "anger": "Use a calming, steady, patient tone.",
    "angry": "Use a calming, steady, patient tone.",
    "disgust": "Use a gentle tone and shift toward pleasant imagery.",
}


def get_last_detected_emotion() -> str:
    return _last_detected_emotion


def _normalise_emotion(label: Optional[str]) -> str:
    return (label or "neutral").strip().lower()


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def build_ssml(text: str, emotion: str) -> str:
    profile = VOICE_PROFILES.get(_normalise_emotion(emotion), VOICE_PROFILES["neutral"])
    voice = profile["voice"]
    style = profile.get("style")
    degree = profile.get("style_degree")
    rate = profile["rate"]
    pitch = profile["pitch"]
    volume = profile["volume"]
    pause = profile["sentence_break"]

    sentence_blocks = []
    for index, sentence in enumerate(_split_sentences(text)):
        sentence_blocks.append(_xml_escape(sentence))
        if index < len(_split_sentences(text)) - 1:
            sentence_blocks.append(f'<break time="{pause}"/>')
    body = "\n".join(sentence_blocks)

    prosody = f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{body}</prosody>'
    if style:
        degree_attr = f' styledegree="{degree}"' if degree is not None else ""
        prosody = f'<mstts:express-as style="{style}"{degree_attr}>{prosody}</mstts:express-as>'

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">'
        f'<voice name="{voice}">{prosody}</voice>'
        '</speak>'
    )


def _tts_azure(ssml: str, output_path: str) -> bool:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        return False

    config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(ssml).get()
    return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted


def _tts_edge_fallback(text: str, emotion: str, output_path: str) -> None:
    import edge_tts

    voice, rate, pitch = {
        "happy": ("en-US-JennyNeural", "+12%", "+4Hz"),
        "joy": ("en-US-JennyNeural", "+12%", "+4Hz"),
        "amusement": ("en-US-JennyNeural", "+15%", "+6Hz"),
        "love": ("en-US-AriaNeural", "-8%", "+2Hz"),
        "neutral": ("en-IN-NeerjaNeural", "-5%", "+0Hz"),
        "surprise": ("en-US-JennyNeural", "+5%", "+4Hz"),
        "sad": ("en-US-AriaNeural", "-20%", "-4Hz"),
        "sadness": ("en-US-AriaNeural", "-20%", "-4Hz"),
        "fear": ("en-US-AriaNeural", "-25%", "-6Hz"),
        "anger": ("en-US-AriaNeural", "-15%", "-2Hz"),
        "angry": ("en-US-AriaNeural", "-15%", "-2Hz"),
        "disgust": ("en-IN-NeerjaNeural", "-10%", "+0Hz"),
    }.get(_normalise_emotion(emotion), ("en-IN-NeerjaNeural", "-5%", "+0Hz"))

    async def _run() -> None:
        await edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch).save(output_path)

    asyncio.run(_run())


def story_to_audio_file(text: str, emotion: str = "neutral") -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    output_path = handle.name
    handle.close()

    if AZURE_SPEECH_KEY and _tts_azure(build_ssml(text, emotion), output_path):
        return output_path

    _tts_edge_fallback(text, emotion, output_path)
    return output_path


def detect_emotion(text: str) -> str:
    global _last_detected_emotion
    try:
        results = _emotion_classifier(text[:512])[0]
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        label = _normalise_emotion(best.get("label"))
    except Exception:
        label = "neutral"
    _last_detected_emotion = label
    return label


def load_documents(folder_path: str) -> List[Dict[str, str]]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []

    documents: List[Dict[str, str]] = []
    for path in folder.rglob("*.txt"):
        if path.name.lower() in SKIP_FILES:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if content:
            documents.append({"filename": path.name, "filepath": str(path), "content": content})
    return documents


def chunk_document(content: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    words = content.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size]).strip()
        if len(chunk) >= 60:
            chunks.append(chunk)
    return chunks


def get_embeddings(text: str, input_type: str = "passage") -> List[float]:
    for attempt in range(2):
        try:
            response = _client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
                extra_body={"input_type": input_type},
            )
            return response.data[0].embedding
        except Exception:
            if attempt == 1:
                raise
            time.sleep(2)
    return []


def index_needed(folder_path: str) -> bool:
    docs = load_documents(folder_path)
    try:
        return _collection.count() < max(3, len(docs))
    except Exception:
        return True


def reset_index() -> None:
    global _collection
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def index_documents(folder_path: str, force_rebuild: bool = False) -> int:
    if not force_rebuild and not index_needed(folder_path):
        return _collection.count()

    reset_index()
    total_chunks = 0

    seed_texts = [
        "Children's stories should feel warm, vivid, safe, and easy to follow.",
        "Describe sounds, textures, warmth, wind, footsteps, rustling leaves, soft fur, and gentle light.",
        "When a child seems sad or scared, the story should become more comforting and reassuring.",
    ]

    for index, text in enumerate(seed_texts):
        _collection.add(
            ids=[f"seed_{index}"],
            documents=[text],
            embeddings=[get_embeddings(text)],
            metadatas=[{"source": "story_seed.txt", "kind": "seed"}],
        )
        total_chunks += 1

    for document in load_documents(folder_path):
        for index, chunk in enumerate(chunk_document(document["content"])[:6]):
            _collection.add(
                ids=[f"{document['filename']}_chunk_{index}"],
                documents=[chunk],
                embeddings=[get_embeddings(chunk)],
                metadatas=[{"source": document["filename"], "kind": "document"}],
            )
            total_chunks += 1

    return total_chunks


def retrieve(query: str, n_results: int = 2) -> List[Dict[str, str]]:
    try:
        if _collection.count() == 0:
            return []
        results = _collection.query(
            query_embeddings=[get_embeddings(query, input_type="query")],
            n_results=n_results,
        )
    except Exception:
        return []

    return [
        {
            "text": document,
            "source": metadata.get("source", "unknown"),
            "kind": metadata.get("kind", "document"),
        }
        for document, metadata in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0])
    ]


def fallback_story(emotion: str) -> str:
    emotion = _normalise_emotion(emotion)
    if emotion in {"sadness", "sad", "fear"}:
        return (
            "A little rabbit found a warm patch of sunlight beside a quiet tree. "
            "It listened to the soft wind in the leaves, took a slow breath, and felt safe. "
            "Soon, a friendly bird sat nearby, and the world felt gentle again."
        )
    if emotion in {"happy", "joy", "amusement", "love"}:
        return (
            "A bright little dragon skipped across a meadow of glowing flowers. "
            "Every step made the grass sparkle, and every laugh made the butterflies dance. "
            "It was the kind of day that felt like a song."
        )
    return (
        "A small cloud floated slowly over a quiet forest while the evening breeze hummed a soft tune. "
        "Below, a sleepy fox curled into warm moss and watched the stars appear one by one."
    )


def choose_final_emotion(
    voice_emotion: Optional[str],
    camera_emotion: Optional[str],
    text_emotion: Optional[str],
) -> Tuple[str, str]:
    voice = _normalise_emotion(voice_emotion)
    camera = _normalise_emotion(camera_emotion)
    text = _normalise_emotion(text_emotion)

    if voice != "neutral":
        return voice, "voice"
    if camera != "neutral":
        return camera, "camera"
    return text, "text"


# 3) fix generate_story()
def generate_story(
    user_message: str,
    voice_emotion: Optional[str] = None,
    camera_emotion: Optional[str] = None,
) -> Tuple[str, str]:
    global _last_detected_emotion

    voice_e = _normalise_emotion(voice_emotion) if voice_emotion else "neutral"
    camera_e = _normalise_emotion(camera_emotion) if camera_emotion else "neutral"
    text_emotion = detect_emotion(user_message)

    if voice_e != "neutral":
        emotion = voice_e
        source = "voice"
    elif camera_e != "neutral":
        emotion = camera_e
        source = "camera"
    else:
        emotion = text_emotion
        source = "text"

    _last_detected_emotion = emotion

    chunks = retrieve(user_message, n_results=2)
    context_text = (
        "\n\n".join(f"[Source: {c['source']}] {c['text']}" for c in chunks)
        if chunks else "No extra context."
    )

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_history[-MAX_HISTORY:]
    )

    style_instr = STYLE_MAP.get(emotion, STYLE_MAP["neutral"])

    system_prompt = f"""
You are a warm storyteller for children, especially blind children who rely
entirely on your voice to experience the story.

The child's emotion right now: {emotion} (detected from their {source}).
Style guidance: {style_instr}

Rules:
- Keep the story short (3-5 sentences).
- Every sentence must end with a period, question mark, or exclamation mark.
- Use rich sensory language: sounds, textures, smells, warmth, temperature.
- Avoid visual-only descriptions like "he saw" or "the blue sky".
- The story must directly incorporate the child's request: {user_message}
- Do not repeat previous stories.
- Keep the story safe, vivid, and easy to follow aloud.

Context: {context_text}
Conversation: {history_text}
""".strip()

    conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": system_prompt}] + conversation_history[-MAX_HISTORY:]

    try:
        response = _client.with_options(timeout=120.0).chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.9,
            max_tokens=220,
        )
        assistant_reply = response.choices[0].message.content.strip()
    except APITimeoutError:
        assistant_reply = fallback_story(emotion)
    except Exception as e:
        assistant_reply = fallback_story(emotion)

    conversation_history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply, emotion

def reset_story() -> None:
    global _last_detected_emotion
    _last_detected_emotion = "neutral"
    conversation_history.clear()
