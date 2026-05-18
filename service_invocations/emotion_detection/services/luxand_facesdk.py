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
RESULTS_FILE = "luxand_facesdk.csv"

_EMOTION_MAPPING = {
    "anger": "anger",
    "angry": "anger",
    "contempt": "contempt",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}


def _extract_emotions(payload: dict) -> dict[str, float | None]:
    if "faces" in payload and payload["faces"]:
        face = payload["faces"][0]
        return face.get("emotions", face.get("emotion", {}))
    return payload.get("emotions", {})


def run_luxand_facesdk(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_token = os.getenv("LUXAND_API_TOKEN")
    if not api_token:
        raise ValueError("LUXAND_API_TOKEN is required.")

    url = os.getenv("LUXAND_EMOTION_URL", "https://api.luxand.cloud/photo/emotions")

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
        print(f"Luxand FaceSDK Cloud: {image_file}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = perf_time.perf_counter()
            response = requests.post(
                url,
                headers={"token": api_token},
                files={"photo": image_bytes},
                timeout=30,
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

        data["id"].append(f"luxand_facesdk_{sample_id:04d}")
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
    return run_luxand_facesdk(vea_data)
