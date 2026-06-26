# Emotion-Aware Story Friend

An AI-powered storytelling app that detects emotion from your voice and face, then generates and narrates a short comforting story adapted to how you feel.

Built as a final year B.Tech capstone project at Babu Banarasi Das University.

---

## Demo

> Speak a request or type it — the app reads your emotion, generates a story using a RAG pipeline powered by LLaMA 3.3 70B, and reads it back to you in an emotion-matched voice.

---

## Architecture

```
Next.js Frontend (localhost:3000)
        ↕ REST API
FastAPI Backend (localhost:8000)
        ↕
Python ML Modules
├── rag.py          — RAG story generation + TTS
├── voice_input.py  — Speech-to-text + voice emotion
├── webcam.py       — Face emotion detection
└── wake_word.py    — Hands-free wake word listener
```

---

## Features

- **Emotion-aware stories** — detects emotion from voice (SpeechBrain) and face (DeepFace + OpenCV), fuses signals with priority `voice > camera > text`
- **RAG story generation** — retrieves context from ChromaDB, generates stories using NVIDIA-hosted LLaMA 3.3 70B
- **Emotion-matched TTS** — Azure Neural TTS with Edge TTS fallback, different voice profile per emotion
- **Hands-free mode** — say "Hey Mycroft" to trigger recording via OpenWakeWord
- **Voice input** — Azure Speech-to-Text with OpenAI Whisper fallback
- **Conversation memory** — multi-turn storytelling sessions
- **Next.js frontend** — responsive React UI with Tailwind CSS, replaces original Streamlit interface

---

## Project Structure

```
.
├── app.py              # Original Streamlit UI (kept for reference)
├── api_server.py       # FastAPI backend — wraps all Python ML modules
├── rag.py              # RAG indexing, story generation, emotion fusion, TTS
├── voice_input.py      # Audio decoding, STT, voice emotion detection
├── webcam.py           # Webcam capture, face emotion detection
├── wake_word.py        # Background wake-word listener
├── requirements.txt    # Python dependencies
├── .env                # Local secrets (create manually, never commit)
├── docs/               # Optional .txt story source files for RAG
├── chroma_db/          # Persistent ChromaDB vector store (created at runtime)
└── nextjs-frontend/    # Next.js + Tailwind CSS frontend
    ├── app/
    │   ├── page.jsx    # Main UI — chat, mic, emotion display
    │   ├── layout.jsx
    │   └── globals.css
    ├── package.json
    └── ...
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, Tailwind CSS |
| Backend API | FastAPI, Uvicorn |
| Story Generation | LLaMA 3.3 70B via NVIDIA API, ChromaDB (RAG) |
| Voice Emotion | SpeechBrain |
| Face Emotion | DeepFace, OpenCV |
| Speech-to-Text | Azure Speech Services, OpenAI Whisper (fallback) |
| Text-to-Speech | Azure Neural TTS, Edge TTS (fallback) |
| Wake Word | OpenWakeWord, Porcupine (optional) |

---

## Setup

### Prerequisites

- Python 3.11
- Node.js 18+
- A microphone and webcam

### 1. Clone the repo

```bash
git clone https://github.com/Threcia-01/emotion-aware-ai-storyteller.git
cd emotion-aware-ai-storyteller
```

### 2. Create Python virtual environment

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 4. Create your .env file

```env
# Required
NVIDIA_API_KEY=your_nvidia_api_key

# Optional — Azure Speech (falls back to Whisper STT + Edge TTS without this)
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=centralindia

# Optional — Porcupine wake word (falls back to OpenWakeWord without this)
PORCUPINE_ACCESS_KEY=your_porcupine_access_key

# Optional — Whisper model size
WHISPER_MODEL=base
```

Get your free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com)

### 5. Add story documents (optional)

```
docs/
├── animals.txt
├── bedtime.txt
└── forest_adventure.txt
```

Plain `.txt` files placed here are chunked and indexed into ChromaDB for retrieval.

### 6. Start the FastAPI backend

```bash
uvicorn api_server:app --reload --port 8000
```

Wait for `Application startup complete.`

### 7. Start the Next.js frontend

Open a second terminal:

```bash
cd nextjs-frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Current emotion state and index info |
| POST | `/api/story/text` | Generate story from text input |
| POST | `/api/story/voice` | Generate story from voice recording |
| POST | `/api/emotion/face` | Detect emotion from webcam |
| POST | `/api/story/reset` | Reset conversation history |
| POST | `/api/index/rebuild` | Rebuild ChromaDB index from docs/ |

---

## How It Works

1. User speaks or types a request in the Next.js frontend
2. Frontend sends audio or text to the FastAPI backend
3. `voice_input.py` transcribes audio and detects voice emotion via SpeechBrain
4. `webcam.py` optionally detects facial emotion via DeepFace
5. `rag.py` fuses emotion signals, retrieves story context from ChromaDB, generates a story using LLaMA 3.3 70B, and synthesizes speech with emotion-matched TTS
6. Frontend receives the story text and base64 audio, displays and plays it back

---

## My Contribution

This was a group final year project. My specific contributions:

- Built the entire Next.js frontend and FastAPI integration layer
- Replaced the original Streamlit UI with a decoupled React + FastAPI architecture
- Integrated all Python ML modules (`rag.py`, `webcam.py`, `voice_input.py`) via REST APIs
- Handled data cleaning and preprocessing pipeline for RAG story documents
- Debugged and fixed dependency issues across Python 3.11, TensorFlow 2.21, and edge-tts

---

## Notes

- First run downloads model files for SpeechBrain, DeepFace, and Whisper — may take a few minutes
- Microphone and camera permissions required in browser
- On Windows, PyAudio is included in requirements.txt as a prebuilt wheel