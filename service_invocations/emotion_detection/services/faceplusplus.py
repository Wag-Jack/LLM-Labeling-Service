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
RESULTS_FILE = "faceplusplus.csv"

_EMOTION_MAPPING = {
    "anger": "anger",
    "contempt": "contempt",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


def _get_api_url() -> str:
    return (
        os.getenv("FACEPP_DETECT_URL")
        or os.getenv("FACEPP_API_URL")
        or os.getenv("FACEPP_API_BASE")
        or "https://api-us.faceplusplus.com/facepp/v3/detect"
    )


def run_faceplusplus(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("FACEPP_API_KEY")
    api_secret = os.getenv("FACEPP_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("FACEPP_API_KEY and FACEPP_API_SECRET are required.")

    url = _get_api_url()

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
        print(f"Face++: {image_file}")

        latency_ms: float | None = None
        error: str | None = None
        normalized: dict[str, float | None] = {}

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = time.perf_counter()
            response = requests.post(
                url,
                data={
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "return_attributes": "emotion",
                },
                files={"image_file": image_bytes},
                timeout=30,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            response.raise_for_status()
            response_json = response.json()
            faces = response_json.get("faces", [])
            raw_scores = faces[0].get("attributes", {}).get("emotion", {}) if faces else {}
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        data["id"].append(f"faceplusplus_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(vea_data):
    return run_faceplusplus(vea_data)
