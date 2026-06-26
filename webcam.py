import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Dict, Optional
import cv2

try:
    from deepface import DeepFace
except ImportError as exc:
    raise ImportError("DeepFace not found. Install it with: pip install deepface") from exc

_DEEPFACE_MAP = {
    "happy": "happy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
}


def _analyse_frame(frame) -> Optional[Dict]:
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
    except Exception as e:
        print(f"DeepFace error: {e}")
        return None

    result = results[0] if isinstance(results, list) and results else results
    if not result:
        return None

    emotion_scores = result.get("emotion", {})
    dominant = str(result.get("dominant_emotion", "neutral")).lower()

    return {
        "emotion": _DEEPFACE_MAP.get(dominant, dominant),
        "score": float(emotion_scores.get(dominant, 0.0)),
        "all_scores": {
            _DEEPFACE_MAP.get(k, k): float(v)
            for k, v in emotion_scores.items()
        },
    }


def choose_emotion(emotions: Dict[str, float], confidence_threshold: float = 25.0) -> str:
    non_neutral = {k: v for k, v in emotions.items() if k != "neutral"}
    if not non_neutral:
        return "neutral"
    best = max(non_neutral, key=non_neutral.get)
    return best if non_neutral[best] >= confidence_threshold else "neutral"


def _open_camera() -> Optional[cv2.VideoCapture]:
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "Default"),
    ]

    for backend_id, backend_name in backends:
        for index in range(4):
            print(f"Trying camera index={index} backend={backend_name}")
            cap = cv2.VideoCapture(index, backend_id)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"Opened camera index={index} backend={backend_name}")
                return cap

            cap.release()

    print("No working camera found.")
    return None


def detect_user_emotion_from_webcam(max_frames: int = 20) -> str:
    cap = _open_camera()
    if cap is None:
        return "neutral"

    emotion_votes: Dict[str, int] = {}

    try:
        for _ in range(10):
            cap.read()

        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            result = _analyse_frame(frame)
            if not result:
                continue

            emotion = choose_emotion(result["all_scores"])
            if emotion != "neutral":
                emotion_votes[emotion] = emotion_votes.get(emotion, 0) + 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not emotion_votes:
        return "neutral"

    winner = max(emotion_votes, key=emotion_votes.get)
    return winner if emotion_votes[winner] >= 2 else "neutral"


def live_preview() -> None:
    cap = _open_camera()
    if cap is None:
        print("Could not open webcam.")
        return

    print("Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            result = _analyse_frame(frame)
            label = "neutral"

            if result:
                label = choose_emotion(result["all_scores"])

            cv2.putText(
                frame,
                f"Emotion: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Emotion Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    live_preview()