import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.emotion_detection.services._shared import (
    build_service_output,
    label_to_name,
    normalize_emotions,
)

load_dotenv()

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "smartclick.csv"

# SmartClick "Face Emotion Recognition" API. The published contract uses a
# multipart POST with the image file in the "image" field and returns a
# top-level "emotions" object with per-emotion probabilities. Endpoint and
# auth header are overridable via env so the script can adapt if SmartClick
# changes their gateway (they have rotated URLs in the past).
_DEFAULT_ENDPOINT = "https://api.smartclick.ai/v1/face-emotion-recognition"

# SmartClick reports the standard Ekman set plus neutral; no contempt.
_EMOTION_MAPPING = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


def _build_headers(api_key: str) -> dict[str, str]:
    header_name = os.getenv("SMARTCLICK_API_KEY_HEADER", "Authorization")
    prefix = os.getenv("SMARTCLICK_API_KEY_PREFIX", "Bearer")
    value = f"{prefix} {api_key}" if prefix else api_key
    return {header_name: value}


def _extract_emotions(payload: dict) -> dict[str, float | None]:
    if "emotions" in payload and isinstance(payload["emotions"], dict):
        return payload["emotions"]
    faces = payload.get("faces") or payload.get("results")
    if isinstance(faces, list) and faces:
        face = faces[0]
        if isinstance(face, dict):
            return face.get("emotions", face.get("emotion", {})) or {}
    return {}


def run_smartclick(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("SMARTCLICK_API_KEY")
    if not api_key:
        raise ValueError("SMARTCLICK_API_KEY must be set in environment.")

    endpoint = os.getenv("SMARTCLICK_EMOTION_URL", _DEFAULT_ENDPOINT)
    image_field = os.getenv("SMARTCLICK_IMAGE_FIELD", "image")
    headers = _build_headers(api_key)

    data = {
        "id": [],
        "label": [],
        "label_name": [],
        "latency_ms": [],
        "service_output": [],
    }

    for _, row in vea_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"SmartClick: {image_file}")

        latency_ms: float | None = None
        error: str | None = None
        normalized: dict[str, float | None] = {}

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = time.perf_counter()
            response = requests.post(
                endpoint,
                headers=headers,
                files={image_field: image_bytes},
                timeout=30,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            response.raise_for_status()
            raw_scores = _extract_emotions(response.json())
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        data["id"].append(f"smartclick_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(vea_data):
    return run_smartclick(vea_data)
