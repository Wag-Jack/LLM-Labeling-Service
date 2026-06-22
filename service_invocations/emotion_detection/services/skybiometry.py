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
_SERVICE_NAME = "skybiometry"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "skybiometry.csv"

# STATUS: disabled in config/services.yaml. This code is correct per the live
# docs, but the current API key does NOT return mood -- live-verified 2026-06-17:
# faces/detect registers the face (1 tag) yet omits mood + ethnicity from the
# attribute set even with attributes=all, and attributes=mood returns only
# `face`. Mood is gated at the account/plan level on SkyBiometry's side; once it
# is provisioned this parser will work unchanged. See services/README.md §4.
#
# SkyBiometry's faces/detect (per the live docs) returns emotion in TWO forms
# under each tag's `attributes`:
#   1. mood -- the single dominant label: {"value": <happy|sad|angry|surprised|
#      disgusted|scared|neutral>, "confidence": 0-100}.
#   2. seven sibling per-emotion attributes (NOT nested under mood) -- anger,
#      disgust, fear, happiness, sadness, surprise, neutral_mood -- each a
#      boolean-style {"value": "true"/"false", "confidence": 0-100} pair, giving
#      a confidence for every basic emotion separately.
# The previous parser assumed mood was a nested dict of per-emotion {confidence}
# entries and iterated mood.items(); against the real {"value", "confidence"}
# shape that matched nothing and silently dropped EVERY result -- which is why
# SkyBiometry looked like it "returned no mood". We now read the seven companion
# attributes into a full 7-class distribution (so SkyBiometry is multi-emotion
# once mood is provisioned), and fall back to the single `mood` label if only
# that is present. These are
# independent per-emotion confidences (like Rekognition), so renormalize stays
# False. There is NO contempt class.
_RETURNS_CONTEMPT = False
_EMOTION_MAPPING = {
    # mood.value labels (single-label fallback)
    "happy": "happy",
    "sad": "sad",
    "angry": "anger",
    "surprised": "surprise",
    "disgusted": "disgust",
    "scared": "fear",
    "neutral": "neutral",
    # per-emotion companion attribute names (full distribution)
    "happiness": "happy",
    "sadness": "sad",
    "anger": "anger",
    "surprise": "surprise",
    "disgust": "disgust",
    "fear": "fear",
    "neutral_mood": "neutral",
}

# The seven sibling per-emotion attributes returned alongside `mood`.
_EMOTION_ATTRIBUTES = (
    "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral_mood",
)


def _present_probability(entry: dict) -> float | None:
    """0-1 "this emotion is present" score from a {value, confidence} attribute.

    SkyBiometry's per-emotion attributes are boolean-style: value is
    "true"/"false" and confidence is how sure it is of that value. A confident
    "false" therefore means the emotion is almost certainly absent, so we invert
    it (100 - confidence) to get a present-probability. A non-boolean/absent
    value falls back to the raw confidence.
    """
    confidence = entry.get("confidence")
    if confidence is None:
        return None
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if str(entry.get("value", "")).strip().lower() == "false":
        conf = 100.0 - conf
    return conf / 100.0


def _extract_emotions(payload: dict) -> dict[str, float | None]:
    photos = payload.get("photos") or []
    if not photos:
        return {}
    tags = photos[0].get("tags") or []
    if not tags:
        # No tag => SkyBiometry's detector did not register a face in the image.
        return {}
    attrs = tags[0].get("attributes") or {}

    # Preferred: the seven per-emotion companion confidences -> full distribution.
    scores: dict[str, float | None] = {}
    for name in _EMOTION_ATTRIBUTES:
        entry = attrs.get(name)
        if isinstance(entry, dict):
            score = _present_probability(entry)
            if score is not None:
                scores[name] = score
    if scores:
        return scores

    # Fallback: only the single dominant `mood` label is available.
    mood = attrs.get("mood") or {}
    value = mood.get("value")
    confidence = mood.get("confidence")
    if not value or confidence is None:
        return {}
    try:
        return {str(value).strip().lower(): float(confidence) / 100.0}
    except (TypeError, ValueError):
        return {}


def run_skybiometry(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("SKYBIOMETRY_API_KEY")
    api_secret = os.getenv("SKYBIOMETRY_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("SKYBIOMETRY_API_KEY and SKYBIOMETRY_API_SECRET are required.")

    url = os.getenv(
        "SKYBIOMETRY_DETECT_URL",
        "https://api.skybiometry.com/fc/faces/detect.json",
    )
    # "all" requests the full attribute set; per the docs that includes mood,
    # though the current key omits it (see the STATUS note above).
    attributes = os.getenv("SKYBIOMETRY_ATTRIBUTES", "all")
    # AffectNet ships tight, sometimes off-frontal crops; the default "Normal"
    # detector frequently fails to register a face on them (no tag -> no mood).
    # "aggressive" (lowercase, per the docs) finds faces at more angles/scales
    # (slower) and is what makes SkyBiometry actually register these faces.
    # Override via env if needed.
    detector = os.getenv("SKYBIOMETRY_DETECTOR", "aggressive")
    # Multipart field name for the uploaded image. SkyBiometry's detect accepts
    # a POSTed image; kept configurable so it can be adjusted without a code
    # change if the account/endpoint expects a different field.
    image_field = os.getenv("SKYBIOMETRY_IMAGE_FIELD", "image")

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
        print(f"SkyBiometry: {image_file}")

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
                    data={
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "attributes": attributes,
                        "detector": detector,
                    },
                    files={image_field: (Path(image_file).name, image_bytes)},
                    timeout=30,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response.raise_for_status()
                payload = response.json()
                raw_scores = _extract_emotions(payload)
                normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            return normalized, latency_ms, error

        # Re-attempt whenever no emotion comes back (error, no face registered,
        # or a face with no mood).
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
        data["id"].append(f"skybiometry_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_skybiometry(affectnet_data, results_path=results_path)
