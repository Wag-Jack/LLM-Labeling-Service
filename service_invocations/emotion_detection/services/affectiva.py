import os
import time as perf_time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.emotion_detection.services._shared import (
    format_service_output,
    label_to_name,
    normalize_emotions,
    pick_top_emotion,
)

load_dotenv()

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "affectiva.csv"

_EMOTION_MAPPING = {
    "anger": "anger",
    "contempt": "contempt",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}


def _build_headers() -> dict:
    headers: dict[str, str] = {}
    api_key = os.getenv("AFFECTIVA_API_KEY")
    if api_key:
        header_name = os.getenv("AFFECTIVA_API_KEY_HEADER", "Authorization")
        prefix = os.getenv("AFFECTIVA_API_KEY_PREFIX", "Bearer")
        value = f"{prefix} {api_key}" if prefix else api_key
        headers[header_name] = value
    return headers


def _extract_emotions(payload: dict) -> dict[str, float | None]:
    if "emotions" in payload:
        return payload.get("emotions", {})
    if "faces" in payload and payload["faces"]:
        face = payload["faces"][0]
        return face.get("emotions", face.get("emotion", {}))
    return {}


def run_affectiva(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_url = os.getenv("AFFECTIVA_API_URL")
    if not api_url:
        raise ValueError(
            "AFFECTIVA_API_URL is required. Set it to your Affectiva endpoint or "
            "SDK wrapper URL."
        )

    image_field = os.getenv("AFFECTIVA_IMAGE_FIELD", "image")

    data = {
        "id": [],
        "image_file": [],
        "label": [],
        "label_name": [],
        "service_output": [],
        "top_emotion": [],
        "latency_ms": [],
    }

    for _, row in vea_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"Affectiva: {image_file}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = perf_time.perf_counter()
            response = requests.post(
                api_url,
                headers=_build_headers(),
                files={image_field: image_bytes},
                timeout=60,
            )
            latency_ms = (perf_time.perf_counter() - start_time) * 1000.0
            response.raise_for_status()
            payload = response.json()

            raw_scores = _extract_emotions(payload)
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            top_emotion = pick_top_emotion(normalized)
            output = format_service_output(payload, normalized)
        except Exception as exc:  # noqa: BLE001
            latency_ms = None
            normalized = {}
            top_emotion = None
            output = format_service_output({"error": str(exc)}, normalized)

        data["id"].append(f"affectiva_{sample_id:04d}")
        data["image_file"].append(image_file)
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["service_output"].append(output)
        data["top_emotion"].append(top_emotion)
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))

    data["llm_judge_score"] = [0.0 for _ in data["id"]]

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(vea_data):
    return run_affectiva(vea_data)
