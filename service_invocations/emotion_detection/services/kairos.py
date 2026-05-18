import os
import time
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
RESULTS_FILE = "kairos.csv"

_EMOTION_MAPPING = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
}


def _poll_media(media_id: str, headers: dict, timeout_seconds: int) -> dict:
    url = f"https://api.kairos.com/v2/media/{media_id}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("frames"):
            return payload
        time.sleep(1)
    return payload


def _extract_emotions(payload: dict) -> dict[str, float | None]:
    frames = payload.get("frames", [])
    if not frames:
        return {}
    people = frames[0].get("people", [])
    if not people:
        return {}
    return people[0].get("emotions", {})


def run_kairos(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    app_id = os.getenv("KAIROS_APP_ID")
    app_key = os.getenv("KAIROS_APP_KEY")
    if not app_id or not app_key:
        raise ValueError("KAIROS_APP_ID and KAIROS_APP_KEY are required.")

    timeout_seconds = int(os.getenv("KAIROS_TIMEOUT_SECONDS", "30"))
    upload_url = os.getenv("KAIROS_MEDIA_URL", "https://api.kairos.com/v2/media")

    headers = {
        "app_id": app_id,
        "app_key": app_key,
    }

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
        print(f"Kairos: {image_file}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = perf_time.perf_counter()
            response = requests.post(
                upload_url,
                headers=headers,
                files={"source": image_bytes},
                data={"timeout": str(timeout_seconds)},
                timeout=60,
            )
            latency_ms = (perf_time.perf_counter() - start_time) * 1000.0
            response.raise_for_status()
            payload = response.json()

            if not payload.get("frames") and payload.get("id"):
                payload = _poll_media(payload["id"], headers, timeout_seconds)

            raw_scores = _extract_emotions(payload)
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            top_emotion = pick_top_emotion(normalized)
            output = format_service_output(payload, normalized)
        except Exception as exc:  # noqa: BLE001
            latency_ms = None
            normalized = {}
            top_emotion = None
            output = format_service_output({"error": str(exc)}, normalized)

        data["id"].append(f"kairos_{sample_id:04d}")
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
    return run_kairos(vea_data)
