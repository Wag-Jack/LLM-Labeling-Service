import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.core.service_cost import record_service_call
from service_invocations.emotion_detection.services._shared import (
    build_service_output,
    label_to_name,
    normalize_emotions,
    request_with_retry,
)

load_dotenv()

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "microsoft_azure_face.csv"
_TASK_NAME = "emotion_detection"
_SERVICE_NAME = "microsoft_azure_face"

# Azure Face returns probability-like scores in [0, 1] that already sum
# to ~1 across the eight emotions, so values are directly comparable
# with other services on the same scale.
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


def _build_endpoint() -> str:
    endpoint = os.getenv("AZURE_FACE_ENDPOINT") or os.getenv(
        "AZURE_FACE_API_ENDPOINT"
    )
    if not endpoint:
        raise ValueError("AZURE_FACE_ENDPOINT is required.")
    endpoint = endpoint.rstrip("/")
    api_version = os.getenv("AZURE_FACE_API_VERSION", "v1.2")
    return f"{endpoint}/face/{api_version}/detect"


def run_microsoft_azure_face(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("AZURE_FACE_KEY") or os.getenv("AZURE_FACE_API_KEY")
    if not api_key:
        raise ValueError("AZURE_FACE_KEY is required.")

    endpoint = _build_endpoint()
    return_attrs = os.getenv("AZURE_FACE_RETURN_ATTRIBUTES", "emotion")
    detection_model = os.getenv("AZURE_FACE_DETECTION_MODEL")

    data = {
        "id": [],
        "label": [],
        "label_name": [],
        "latency_ms": [],
        "cost_usd": [],
        "service_output": [],
    }

    for _, row in vea_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"Microsoft Azure Face API: {image_file}")

        params = {"returnFaceAttributes": return_attrs}
        if detection_model:
            params["detectionModel"] = detection_model

        latency_ms: float | None = None
        error: str | None = None
        normalized: dict[str, float | None] = {}

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            start_time = time.perf_counter()
            response = request_with_retry(
                "POST",
                endpoint,
                params=params,
                headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "Content-Type": "application/octet-stream",
                },
                data=image_bytes,
                timeout=30,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            response.raise_for_status()
            response_json = response.json()
            face = response_json[0] if response_json else {}
            raw_scores = face.get("faceAttributes", {}).get("emotion", {})
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        cost = record_service_call(_TASK_NAME, _SERVICE_NAME, sample_id, count=1)
        data["id"].append(f"microsoft_azure_face_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(vea_data, results_path=None):
    return run_microsoft_azure_face(vea_data, results_path=results_path)
