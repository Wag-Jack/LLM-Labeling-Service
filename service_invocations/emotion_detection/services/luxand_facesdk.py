import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

from service_invocations.core.service_cost import record_service_call
from service_invocations.emotion_detection.services._shared import (
    build_service_output,
    call_until_emotion,
    label_to_name,
    normalize_emotions,
    pick_top_emotion,
    request_with_retry,
)

load_dotenv()

_TASK_NAME = "emotion_detection"
_SERVICE_NAME = "luxand_facesdk"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "luxand_facesdk.csv"

# Luxand can emit a "contempt" score as part of its emotion distribution.
# AffectNet-7 has no contempt class, so contempt is dropped on projection and
# the remaining seven scores are renormalized back to a proper distribution
# (renormalize=True below).
_RETURNS_CONTEMPT = True
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


def run_luxand_facesdk(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_token = os.getenv("LUXAND_API_TOKEN")
    if not api_token:
        raise ValueError("LUXAND_API_TOKEN is required.")

    url = os.getenv("LUXAND_EMOTION_URL", "https://api.luxand.cloud/photo/emotions")

    data = {
        "id": [],
        "label": [],
        "label_name": [],
        "latency_ms": [],
        "cost_usd": [],
        "service_output": [],
    }

    for _, row in affectnet_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"Luxand FaceSDK Cloud: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                with open(image_file, "rb") as f:
                    image_bytes = f.read()

                start_time = time.perf_counter()
                response = request_with_retry(
                    "POST",
                    url,
                    headers={"token": api_token},
                    files={"photo": image_bytes},
                    timeout=30,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response.raise_for_status()
                payload = response.json()
                raw_scores = _extract_emotions(payload)
                normalized = normalize_emotions(
                    raw_scores, mapping=_EMOTION_MAPPING, renormalize=True
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            return normalized, latency_ms, error

        # Re-attempt whenever no emotion comes back (error or no face detected).
        normalized, latency_ms, error, attempts = call_until_emotion(
            _attempt, label=f"{_SERVICE_NAME} {Path(image_file).name}"
        )

        top_name, top_score = pick_top_emotion(normalized)
        if error:
            print(f"  -> ERROR: {error}", flush=True)
        elif top_name is None:
            print(
                f"  -> (no face detected / no emotions returned after {attempts} attempts)",
                flush=True,
            )
        else:
            scores = ", ".join(
                f"{k}={v:.2f}" for k, v in normalized.items() if v is not None
            )
            print(f"  -> {top_name} ({top_score:.2f})  [{scores}]", flush=True)

        # One billed request per attempt (retries re-bill), face returned or not.
        cost = record_service_call(_TASK_NAME, _SERVICE_NAME, sample_id, count=attempts)
        data["id"].append(f"luxand_facesdk_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_luxand_facesdk(affectnet_data, results_path=results_path)
