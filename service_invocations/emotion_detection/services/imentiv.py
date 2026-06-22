import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

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
_SERVICE_NAME = "imentiv"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "imentiv.csv"

# Imentiv's image emotion API is ASYNCHRONOUS: POST /v2/images queues the image
# (requires a Referer header + a "title" field) and returns an id; the per-face
# emotion scores are fetched by polling GET /v1/images/{id}, where they live in
# record["faces"][i]["emotions"]. Verified against the live API: the response is
# an EIGHT-emotion distribution (angry/disgust/fear/happy/sad/surprise/neutral
# + contempt) summing to ~1.0. AffectNet-7 has no contempt class, so contempt is
# dropped on projection and the remaining seven are renormalized (renormalize=True).
_RETURNS_CONTEMPT = True
_EMOTION_MAPPING = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}

# Emotion-label tokens used to recognize the per-face score dict inside the
# (loosely documented) Imentiv response without hard-coding its nesting.
_EMOTION_TOKENS = set(_EMOTION_MAPPING) | {"contempt"}

# Status strings that mean the asynchronous analysis is finished.
_DONE_STATUSES = {"completed", "complete", "success", "succeeded", "done", "processed", "finished"}
_FAILED_STATUSES = {"failed", "error", "rejected"}


def _api_base() -> str:
    return (os.getenv("IMENTIV_API_BASE") or "https://api.imentiv.ai").rstrip("/")


# NOTE: Imentiv's detector needs a margin around tightly-cropped faces (it finds
# 0 faces in raw 96x96 AffectNet crops). That padding is applied centrally at
# load time (data_management/affectnet.py -> preprocess_face_image) so EVERY
# service and LLM gets the same input -- this service sends the image as-is.


def _find_emotion_scores(obj) -> dict | None:
    """Recursively locate the per-face emotion score mapping in a response.

    Imentiv's exact JSON nesting is not fully documented, so rather than depend
    on a fixed path we search for the first dict whose keys overlap the known
    emotion tokens with numeric values (>=3 matches to avoid false hits).
    """
    if isinstance(obj, dict):
        lowered = {str(k).strip().lower(): v for k, v in obj.items()}
        matched = {
            k: v
            for k, v in lowered.items()
            if k in _EMOTION_TOKENS and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if len(matched) >= 3:
            return matched
        for value in obj.values():
            found = _find_emotion_scores(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_emotion_scores(item)
            if found:
                return found
    return None


def _extract_status(payload) -> str:
    if isinstance(payload, dict):
        for key in ("status", "state", "processing_status"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip().lower()
    return ""


def _extract_image_id(payload):
    if isinstance(payload, dict):
        for key in ("id", "image_id", "imageId", "_id", "uuid"):
            value = payload.get(key)
            if value:
                return value
        # Sometimes wrapped, e.g. {"data": {"id": ...}}.
        for nested_key in ("data", "image", "result"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                found = _extract_image_id(nested)
                if found:
                    return found
    return None


def _poll_for_scores(session_get, image_id, headers, interval, timeout):
    """Poll GET /v1/images/{id} until scores appear, fail, or time out."""
    deadline = time.monotonic() + timeout
    last_status = ""
    while True:
        resp = session_get(image_id, headers)
        resp.raise_for_status()
        payload = resp.json()
        scores = _find_emotion_scores(payload)
        if scores:
            return scores, None
        status = _extract_status(payload)
        last_status = status or last_status
        if status in _FAILED_STATUSES:
            return None, f"Imentiv reported status '{status}' for image {image_id}."
        # Analysis finished but no emotion scores -> no face was detected.
        # Stop polling instead of waiting out the full timeout.
        if status in _DONE_STATUSES:
            return None, (
                f"Imentiv completed image {image_id} with no emotion scores "
                "(no face detected)."
            )
        if time.monotonic() >= deadline:
            return None, (
                f"Imentiv analysis for image {image_id} did not return emotion "
                f"scores within {timeout:.0f}s (last status: '{last_status or 'unknown'}')."
            )
        time.sleep(interval)


def run_imentiv(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("IMENTIV_API_KEY")
    if not api_key:
        raise ValueError("IMENTIV_API_KEY is required.")

    base = _api_base()
    upload_url = f"{base}/v2/images"
    poll_interval = float(os.getenv("IMENTIV_POLL_INTERVAL", "2.0"))
    poll_timeout = float(os.getenv("IMENTIV_POLL_TIMEOUT", "90.0"))
    # The /v2/images endpoint rejects API-key auth without a Referer header
    # ("Missing Referer header or is_internal header"); the value is not
    # validated against a specific origin, only required to be present.
    referer = os.getenv("IMENTIV_REFERER", "https://imentiv.ai")
    headers = {"X-API-Key": api_key, "Referer": referer}

    def _get_image(image_id, hdrs):
        return request_with_retry(
            "GET", f"{base}/v1/images/{image_id}", headers=hdrs, timeout=30
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
        print(f"Imentiv: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                with open(image_file, "rb") as f:
                    image_bytes = f.read()

                start_time = time.perf_counter()
                upload = request_with_retry(
                    "POST",
                    upload_url,
                    headers=headers,
                    data={"title": f"affectnet_{sample_id:04d}"},  # required by /v2/images
                    files={"image_file": (Path(image_file).name, image_bytes)},
                    timeout=30,
                )
                upload.raise_for_status()
                upload_payload = upload.json()

                # Some responses may carry scores directly; otherwise poll by id.
                raw_scores = _find_emotion_scores(upload_payload)
                if not raw_scores:
                    image_id = _extract_image_id(upload_payload)
                    if image_id is None:
                        raise ValueError(
                            "Imentiv upload response contained no image id to poll: "
                            f"{upload_payload!r}"
                        )
                    raw_scores, poll_error = _poll_for_scores(
                        _get_image, image_id, headers, poll_interval, poll_timeout
                    )
                    if poll_error:
                        raise RuntimeError(poll_error)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                normalized = normalize_emotions(
                    raw_scores, mapping=_EMOTION_MAPPING, renormalize=True
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            return normalized, latency_ms, error

        # Re-attempt whenever no emotion comes back (error, no face, or a
        # completed analysis with no scores).
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

        # One billed upload per attempt (retries re-bill), face returned or not.
        cost = record_service_call(_TASK_NAME, _SERVICE_NAME, sample_id, count=attempts)
        data["id"].append(f"imentiv_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_imentiv(affectnet_data, results_path=results_path)
