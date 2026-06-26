"""
api_server.py — FastAPI wrapper around existing Python modules.
Run with: uvicorn api_server:app --reload --port 8000
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

app = FastAPI(title="Emotion-Aware Story API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track state server-side
_state = {
    "indexed": False,
    "chunk_count": 0,
    "user_emotion": "neutral",
    "voice_emotion": "neutral",
}


@app.on_event("startup")
async def startup():
    Path("docs").mkdir(exist_ok=True)
    if not _state["indexed"]:
        count = index_documents("docs")
        _state["indexed"] = True
        _state["chunk_count"] = count


# ── Status ──────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "chunk_count": _state["chunk_count"],
        "last_emotion": get_last_detected_emotion(),
        "user_emotion": _state["user_emotion"],
        "voice_emotion": _state["voice_emotion"],
        "indexed": _state["indexed"],
    }


# ── Story generation ─────────────────────────────────────────────────────────

class StoryRequest(BaseModel):
    text: str


@app.post("/api/story/text")
def story_from_text(req: StoryRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    story, emotion = generate_story(
        req.text,
        voice_emotion=_state["voice_emotion"],
        camera_emotion=_state["user_emotion"],
    )

    audio_path = story_to_audio_file(story, emotion)
    audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()

    return {
        "story": story,
        "emotion": emotion,
        "audio_b64": audio_b64,
        "history": conversation_history[-10:],
    }


@app.post("/api/story/voice")
async def story_from_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio received")

    text, voice_emotion = process_voice_bytes(audio_bytes)
    _state["voice_emotion"] = voice_emotion or "neutral"

    if not text.strip():
        return {"error": "Could not understand speech", "text": "", "emotion": "neutral"}

    story, final_emotion = generate_story(
        text,
        voice_emotion=_state["voice_emotion"],
        camera_emotion=_state["user_emotion"],
    )

    audio_path = story_to_audio_file(story, final_emotion)
    audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()

    return {
        "text": text,
        "story": story,
        "emotion": final_emotion,
        "voice_emotion": voice_emotion,
        "audio_b64": audio_b64,
        "history": conversation_history[-10:],
    }


# ── Emotion ──────────────────────────────────────────────────────────────────

@app.post("/api/emotion/face")
def detect_face_emotion():
    detected = detect_user_emotion_from_webcam(max_frames=15) or "neutral"
    _state["user_emotion"] = detected
    return {"emotion": detected}


# ── Controls ─────────────────────────────────────────────────────────────────

@app.post("/api/story/reset")
def reset():
    reset_story()
    _state["user_emotion"] = "neutral"
    _state["voice_emotion"] = "neutral"
    return {"ok": True}


@app.post("/api/index/rebuild")
def rebuild_index():
    count = index_documents("docs", force_rebuild=True)
    _state["chunk_count"] = count
    return {"chunk_count": count}
