import base64
import os
import time as perf_time
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
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
RESULTS_FILE = "google_cloud_vision.csv"

_LIKELIHOOD_SCORES = {
    "UNKNOWN": 0.0,
    "VERY_UNLIKELY": 0.05,
    "UNLIKELY": 0.25,
    "POSSIBLE": 0.5,
    "LIKELY": 0.75,
    "VERY_LIKELY": 0.9,
}

_EMOTION_MAPPING = {
    "joy": "happy",
    "sorrow": "sad",
    "anger": "anger",
    "surprise": "surprise",
}


def _safe_json(response: requests.Response) -> dict | str:
    try:
        return response.json()
    except ValueError:
        return response.text


def _resolve_credentials_path() -> str | None:
    env_path = os.getenv("GOOGLE_VISION_CREDENTIALS") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if env_path:
        return env_path

    candidate = (
        Path.cwd() / "credentials" / "emotion_detection" / "llm-as-a-judge_gc.json"
    )
    if candidate.exists():
        return str(candidate)

    fallback = (
        Path.cwd() / "credentials" / "speech_recognition" / "llm-as-a-judge_gc.json"
    )
    if fallback.exists():
        return str(fallback)

    return None


def _likelihood_to_score(value: str | None) -> float | None:
    if not value:
        return None
    return _LIKELIHOOD_SCORES.get(value, None)


def _extract_emotions(face: dict) -> dict[str, float | None]:
    if not face:
        return {}
    scores = {
        "joy": _likelihood_to_score(face.get("joyLikelihood")),
        "sorrow": _likelihood_to_score(face.get("sorrowLikelihood")),
        "anger": _likelihood_to_score(face.get("angerLikelihood")),
        "surprise": _likelihood_to_score(face.get("surpriseLikelihood")),
    }
    return scores


def run_google_cloud_vision(vea_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    cred_path = _resolve_credentials_path()
    if not cred_path:
        raise FileNotFoundError(
            "Google Vision credentials not found. Set GOOGLE_VISION_CREDENTIALS or "
            "GOOGLE_APPLICATION_CREDENTIALS."
        )

    credentials = service_account.Credentials.from_service_account_file(
        cred_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = Request()
    quota_project = os.getenv("GOOGLE_VISION_QUOTA_PROJECT")

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
        print(f"Google Cloud Vision: {image_file}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            credentials.refresh(request)
            token = credentials.token
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }
            if quota_project:
                headers["x-goog-user-project"] = quota_project

            payload = {
                "requests": [
                    {
                        "image": {
                            "content": base64.b64encode(image_bytes).decode("utf-8")
                        },
                        "features": [{"type": "FACE_DETECTION", "maxResults": 1}],
                    }
                ]
            }

            start_time = perf_time.perf_counter()
            response = requests.post(
                "https://vision.googleapis.com/v1/images:annotate",
                headers=headers,
                json=payload,
                timeout=30,
            )
            latency_ms = (perf_time.perf_counter() - start_time) * 1000.0
            response_json = _safe_json(response)
            if not response.ok:
                normalized = {}
                top_emotion = None
                output = format_service_output(
                    {
                        "error": "google_cloud_vision_request_failed",
                        "status_code": response.status_code,
                        "reason": response.reason,
                        "url": response.url,
                        "credentials_project_id": credentials.project_id,
                        "quota_project": quota_project,
                        "response": response_json,
                    },
                    normalized,
                )
                data["id"].append(f"google_cloud_vision_{sample_id:04d}")
                data["image_file"].append(image_file)
                data["label"].append(label)
                data["label_name"].append(label_to_name(label))
                data["service_output"].append(output)
                data["top_emotion"].append(top_emotion)
                data["latency_ms"].append(round(latency_ms, 2))
                continue

            face_annotations = (
                response_json.get("responses", [{}])[0].get("faceAnnotations", [])
            )
            face = face_annotations[0] if face_annotations else {}
            raw_scores = _extract_emotions(face)
            normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            top_emotion = pick_top_emotion(normalized)
            output = format_service_output(response_json, normalized)
        except Exception as exc:  # noqa: BLE001
            latency_ms = None
            normalized = {}
            top_emotion = None
            output = format_service_output({"error": str(exc)}, normalized)

        data["id"].append(f"google_cloud_vision_{sample_id:04d}")
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
    return run_google_cloud_vision(vea_data)
