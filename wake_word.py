import io
import os
import threading
import time
from typing import Callable, Optional
import numpy as np

SAMPLE_RATE = 16000
FRAME_LENGTH = 1280  # 80 ms, recommended for openwakeword

# How long (seconds) to suppress repeated wake detections after one fires.
# Must be longer than your typical record_audio duration.
WAKE_COOLDOWN = 8.0

# How many seconds to wait before restarting a crashed listen loop.
_RESTART_DELAY = 2.0


class WakeWordListener:
    def __init__(self, on_wake: Optional[Callable] = None, sensitivity: float = 0.35):
        self._event = threading.Event()
        self._stop_event = threading.Event()
        # Pause flag: when set, the listen loop yields the mic
        self._pause_event = threading.Event()
        # Signals back to record_audio that the loop has fully released the mic
        self._paused_ack = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_wake = on_wake
        self._sensitivity = sensitivity
        self._backend = None
        # Timestamp of last trigger — used to suppress repeated firings
        self._last_trigger_time: float = 0.0
        # True while record_audio() is running — suppresses re-trigger
        self._recording = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> str:
        self._stop_event.clear()
        backend = self._load_backend()
        self._backend = backend
        if backend == "unavailable":
            print("⚠️ Wake word: no backend available — install openwakeword or pvporcupine")
            return backend
        self._ensure_thread_running(backend)
        print(f"👂 Wake word listener started ({backend})")
        return backend

    def _ensure_thread_running(self, backend: str) -> None:
        """Start the listener thread if it is not currently alive."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._supervised_loop,
                args=(backend,),
                daemon=True,
                name="wake-word-listener",
            )
            self._thread.start()

    def _supervised_loop(self, backend: str) -> None:
        """
        Wraps the real listen loop so that if it crashes, it restarts
        automatically after a short delay — as long as stop() has not been called.
        """
        while not self._stop_event.is_set():
            try:
                if backend == "porcupine":
                    self._porcupine_loop()
                elif backend == "openwakeword":
                    self._openwakeword_loop()
            except Exception as e:
                print(f"❌ Wake word listener crashed: {e}  — restarting in {_RESTART_DELAY}s")

            if not self._stop_event.is_set():
                time.sleep(_RESTART_DELAY)
                print(f"🔄 Restarting wake word listener ({backend})")

    def stop(self) -> None:
        self._stop_event.set()

    def triggered(self) -> bool:
        # Also make sure the thread is still alive; resurrect if needed.
        if self._backend and self._backend != "unavailable":
            self._ensure_thread_running(self._backend)
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()

    def record_audio(self, duration: float = 5.0) -> Optional[bytes]:
        """
        Pause the background wake-word mic stream, record `duration` seconds
        of audio on the same mic, then resume listening.
        Returns raw WAV bytes (16-bit PCM, 16 kHz mono), or None on failure.
        """
        import pyaudio
        import wave

        self._recording.set()  # block re-triggers while we record

        # 1. Ask the listen loop to release the mic.
        #    Clear paused_ack BEFORE setting pause_event so we never read
        #    a stale signal from a previous call.
        self._paused_ack.clear()
        self._pause_event.set()
        if self._thread and self._thread.is_alive():
            got_ack = self._paused_ack.wait(timeout=3.0)
            if not got_ack:
                print("⚠️ record_audio: listen loop did not acknowledge pause — proceeding anyway")

        pa = pyaudio.PyAudio()
        frames = []
        stream = None
        try:
            stream = pa.open(
                rate=SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=FRAME_LENGTH,
            )
            total_frames = int(SAMPLE_RATE / FRAME_LENGTH * duration)
            print(f"🎙️ Recording {duration}s of audio ({total_frames} frames)…")
            for _ in range(total_frames):
                data = stream.read(FRAME_LENGTH, exception_on_overflow=False)
                frames.append(data)
            print("✅ Recording complete")
        except Exception as e:
            print(f"❌ record_audio failed: {e}")
            frames = []
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()
            # 2. Resume the background listener.
            self._pause_event.clear()
            self._recording.clear()

        if not frames:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_backend(self) -> str:
        key = os.getenv("PORCUPINE_ACCESS_KEY", "")
        if key:
            try:
                import pvporcupine  # noqa: F401
                return "porcupine"
            except ImportError:
                print("⚠️ pvporcupine not installed — trying openwakeword")
        try:
            import openwakeword  # noqa: F401
            from openwakeword.model import Model  # noqa: F401
            return "openwakeword"
        except Exception as e:
            print(f"⚠️ openwakeword import failed: {e}")
            return "unavailable"

    def _listen_loop(self, backend: str) -> None:
        # Kept for compatibility; _supervised_loop now calls the specific methods directly.
        if backend == "porcupine":
            self._porcupine_loop()
        elif backend == "openwakeword":
            self._openwakeword_loop()

    def _should_trigger(self) -> bool:
        """Return True only if we are outside the cooldown window and not recording."""
        if self._recording.is_set():
            return False
        elapsed = time.monotonic() - self._last_trigger_time
        return elapsed >= WAKE_COOLDOWN

    def _porcupine_loop(self) -> None:
        import pvporcupine
        import pyaudio

        key = os.getenv("PORCUPINE_ACCESS_KEY", "")
        porcupine = pvporcupine.create(
            access_key=key,
            keywords=["porcupine"],
            sensitivities=[self._sensitivity],
        )
        pa = pyaudio.PyAudio()

        def _open_stream():
            return pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
            )

        stream = _open_stream()
        print("👂 Porcupine listening…")
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    stream.stop_stream()
                    stream.close()
                    stream = None
                    self._paused_ack.set()
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.05)
                    if not self._stop_event.is_set():
                        stream = _open_stream()
                    continue

                try:
                    pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                except OSError as e:
                    print(f"⚠️ Porcupine stream read error: {e} — reopening")
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    stream = _open_stream()
                    continue

                frame = list(memoryview(pcm).cast("h"))
                idx = porcupine.process(frame)
                if idx >= 0 and self._should_trigger():
                    print("🔔 Wake word detected (Porcupine)")
                    self._trigger()
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()
            porcupine.delete()

    def _openwakeword_loop(self) -> None:
        import pyaudio
        import openwakeword
        from openwakeword.model import Model

        if hasattr(openwakeword, "utils") and hasattr(openwakeword.utils, "download_models"):
            try:
                openwakeword.utils.download_models()
            except Exception as e:
                print(f"⚠️ Model download skipped/failed: {e}")

        model = Model(wakeword_models=[], inference_framework="onnx")
        pa = pyaudio.PyAudio()

        def _open_stream():
            return pa.open(
                rate=SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=FRAME_LENGTH,
            )

        stream = _open_stream()
        print("👂 openwakeword listening (say 'hey mycroft')…")
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    stream.stop_stream()
                    stream.close()
                    stream = None
                    self._paused_ack.set()
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.05)
                    if not self._stop_event.is_set():
                        stream = _open_stream()
                    continue

                try:
                    raw = stream.read(FRAME_LENGTH, exception_on_overflow=False)
                except OSError as e:
                    print(f"⚠️ openwakeword stream read error: {e} — reopening")
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    stream = _open_stream()
                    continue

                audio = np.frombuffer(raw, dtype=np.int16)
                scores = model.predict(audio)
                for model_name, score in scores.items():
                    if score > self._sensitivity and self._should_trigger():
                        print(
                            f"🔔 Wake word detected (openwakeword: {model_name}, score={score:.3f})"
                        )
                        self._trigger()
                        break  # one trigger per frame
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    def _trigger(self) -> None:
        self._last_trigger_time = time.monotonic()  # start cooldown
        self._event.set()
        if self._on_wake:
            try:
                self._on_wake()
            except Exception as e:
                print(f"⚠️ Wake word callback error: {e}")


_listener: Optional[WakeWordListener] = None


def get_listener() -> WakeWordListener:
    global _listener
    if _listener is None:
        _listener = WakeWordListener(sensitivity=0.35)
        _listener.start()
    elif _listener._backend and _listener._backend != "unavailable":
        # Resurrect the thread if it died (e.g. after a crash)
        _listener._ensure_thread_running(_listener._backend)
    return _listener