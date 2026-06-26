# Emotion-Aware Story Friend

Emotion-Aware Story Friend is a Streamlit app that listens to a child’s request, detects emotion from voice and optionally from the webcam, retrieves supporting story context, generates a short comforting story, and reads it aloud. The current project is organized around a Streamlit UI layer, a RAG and TTS engine, a voice pipeline, webcam emotion detection, and a background wake-word listener.

## Features

- Streamlit interface for typed and voice-first storytelling interactions.
- Wake-word support through a background listener with direct post-wake recording.
- Voice transcription using Azure Speech-to-Text with Whisper fallback.
- Voice emotion detection using SpeechBrain.
- Face emotion detection using DeepFace and OpenCV.
- Retrieval-augmented story generation with ChromaDB and NVIDIA-hosted models.
- Emotion-aware speech synthesis using Azure Speech, with Edge TTS fallback.
- Conversation memory for short multi-turn storytelling sessions.

## Current project structure

```text
.
├── app.py              # Streamlit UI, session flow, wake polling, playback
├── rag.py              # RAG indexing/retrieval, emotion fusion, story generation, TTS
├── voice_input.py      # Audio decoding, trimming, STT, speech emotion recognition
├── webcam.py           # Webcam capture and face emotion detection
├── wake_word.py        # Background wake-word listener and direct microphone recording
├── requirements.txt    # Python dependencies for the current codebase
├── README.md           # Project documentation
├── .env                # Local secrets and configuration (create manually)
├── docs/               # Optional .txt knowledge/story source files
└── chroma_db/          # Persistent Chroma vector store (created at runtime)
```

## How the app works

1. `app.py` starts the Streamlit interface and indexes documents from `docs/` if available.
2. User input can come from typed text, manual audio input, or a wake-word triggered recording flow.
3. `voice_input.py` prepares audio, runs speech-to-text, and predicts voice emotion.
4. `webcam.py` can detect facial emotion when the user asks the app to read the face.
5. `rag.py` fuses emotion signals with priority `voice > camera > text`, retrieves context from ChromaDB, generates a short story, and synthesizes speech output.
6. The generated MP3 is played back in the Streamlit app.

## Environment variables

Create a `.env` file in the project root.

Required:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Optional but recommended:

```env
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=centralindia
PORCUPINE_ACCESS_KEY=your_porcupine_access_key
WHISPER_MODEL=base
```

Notes:

- `NVIDIA_API_KEY` is required because `rag.py` initializes the NVIDIA embedding and chat client at import time.
- `AZURE_SPEECH_KEY` enables Azure STT and Azure TTS; without it, STT falls back to Whisper and TTS falls back to Edge TTS.
- `PORCUPINE_ACCESS_KEY` is only needed when using the Porcupine backend for wake-word detection.
- If no wake-word backend is installed or configured, the app can still run with typed input and manual audio input.

## Installation

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add optional story documents

Create a `docs/` folder beside `app.py` and place `.txt` files inside it. These are chunked and indexed into ChromaDB for retrieval. `dialogs.txt` is skipped by the current indexing logic.

Example:

```text
docs/
├── animals.txt
├── bedtime.txt
└── forest_adventure.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## Dependency notes

- `streamlit` powers the UI.
- `chromadb`, `openai`, and `python-dotenv` are used by the RAG pipeline.
- `transformers`, `torch`, and `torchaudio` support text and audio emotion models.
- `speechbrain`, `soundfile`, `numpy`, and `pydub` support voice preprocessing and emotion detection.
- `deepface` and `opencv-python` support webcam emotion detection.
- `azure-cognitiveservices-speech` and `edge-tts` provide speech services.
- `openwakeword`, `pvporcupine`, and `pyaudio` support wake-word detection and microphone capture.
- `openai-whisper` is required for the offline transcription fallback used by `voice_input.py`.

## Run checklist

- Make sure a microphone is connected and allowed by the OS.
- For webcam emotion detection, ensure a working camera is available.
- For Windows, `PyAudio` may require a wheel installation if normal pip install fails.
- The first run may download model files for SpeechBrain, Transformers, DeepFace, or Whisper.

## Current behavior reflected here

This README matches the present code structure in `app.py`, `rag.py`, `voice_input.py`, `webcam.py`, and `wake_word.py`, including wake-word polling, Azure STT with Whisper fallback, emotion-aware TTS, and runtime-created `docs/` and `chroma_db/` folders.