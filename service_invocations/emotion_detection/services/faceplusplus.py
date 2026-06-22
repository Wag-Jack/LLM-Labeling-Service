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

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "faceplusplus.csv"
_TASK_NAME = "emotion_detection"
_SERVICE_NAME = "faceplusplus"

# Face++ Detect (v3) with return_attributes=emotion reports seven emotions
# whose values sum to ~100. There is NO contempt class, so nothing is dropped
# and no renormalization is needed.
_RETURNS_CONTEMPT = False
_EMOTION_MAPPING = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


# Face++'s free tier rejects overlapping/too-frequent requests with
# HTTP 403 {"error_message": "CONCURRENCY_LIMIT_EXCEEDED"}. Treat that as a
# transient, retryable condition (backoff) rather than a permanent error, and
# space requests out slightly to reduce how often it trips.
_REQUEST_DELAY = float(os.getenv("FACEPP_REQUEST_DELAY", "0.6"))


def _is_concurrency_limit(response) -> bool:
    return response.status_code == 403 and "CONCURRENCY_LIMIT_EXCEEDED" in response.text


def _get_api_url() -> str:
    return (
        os.getenv("FACEPP_DETECT_URL")
        or os.getenv("FACEPP_API_URL")
        or os.getenv("FACEPP_API_BASE")
        or "https://api-us.faceplusplus.com/facepp/v3/detect"
    )


def run_faceplusplus(affectnet_data, results_path: Path | None = None):
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
        "cost_usd": [],
        "service_output": [],
    }

    for _, row in affectnet_data.iterrows():
        image_file = row["image"]
        sample_id = int(row["id"])
        label = row.get("label")
        print(f"Face++: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                with open(image_file, "rb") as f:
                    image_bytes = f.read()

                if _REQUEST_DELAY > 0:
                    time.sleep(_REQUEST_DELAY)
                start_time = time.perf_counter()
                response = request_with_retry(
                    "POST",
                    url,
                    data={
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "return_attributes": "emotion",
                    },
                    files={"image_file": image_bytes},
                    timeout=30,
                    extra_retryable=_is_concurrency_limit,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response.raise_for_status()
                response_json = response.json()
                faces = response_json.get("faces", [])
                raw_scores = faces[0].get("attributes", {}).get("emotion", {}) if faces else {}
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
        data["id"].append(f"faceplusplus_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_faceplusplus(affectnet_data, results_path=results_path)
