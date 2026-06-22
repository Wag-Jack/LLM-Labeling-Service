import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
import pandas as pd

from service_invocations.core.service_cost import record_service_call
from service_invocations.emotion_detection.services._shared import (
    build_service_output,
    call_until_emotion,
    label_to_name,
    normalize_emotions,
    pick_top_emotion,
)

load_dotenv()

_TASK_NAME = "emotion_detection"
_SERVICE_NAME = "aws_rekognition"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "aws_rekognition.csv"

# Rekognition returns "CALM" and "CONFUSED" which have no direct AffectNet-7
# counterpart; both map to "neutral" for cross-service comparison. Rekognition
# has NO contempt class, so nothing is dropped / renormalized.
_RETURNS_CONTEMPT = False
_EMOTION_MAPPING = {
    "angry": "anger",
    "calm": "neutral",
    "confused": "neutral",
    "disgusted": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
}


def _extract_emotions(response: dict) -> dict[str, float | None]:
    faces = response.get("FaceDetails", [])
    if not faces:
        return {}
    scores: dict[str, float | None] = {}
    for entry in faces[0].get("Emotions", []):
        raw_type = str(entry.get("Type", "")).lower()
        confidence = entry.get("Confidence")
        # Rekognition reports confidence on a 0-100 scale; rescale to 0-1
        # so all FER providers share the same numeric range.
        scores[raw_type] = None if confidence is None else float(confidence) / 100.0
    return scores


def run_aws_rekognition(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    client = boto3.client(
        "rekognition",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

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
        print(f"AWS Rekognition: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                with open(image_file, "rb") as f:
                    image_bytes = f.read()

                start_time = time.perf_counter()
                response = client.detect_faces(
                    Image={"Bytes": image_bytes},
                    Attributes=["EMOTIONS"],
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw_scores = _extract_emotions(response)
                normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
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
        data["id"].append(f"aws_rekognition_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_aws_rekognition(affectnet_data, results_path=results_path)
