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
)

load_dotenv()

_TASK_NAME = "emotion_detection"
_SERVICE_NAME = "deepface"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "deepface.csv"

# DeepFace is the local, open-source `deepface` library. It runs entirely
# on-device -- no API key, no network call -- so there is no per-image cloud cost
# (priced at $0 in services.yaml). DeepFace.analyze(actions=["emotion"]) returns
# a per-face `emotion` dict over the SEVEN labels below on a 0-100 scale (summing
# to ~100); we rescale to 0-1 to match the other FER services' convention. There
# is NO contempt class, so nothing is dropped and no renormalization is needed.
_RETURNS_CONTEMPT = False
_EMOTION_MAPPING = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

# Face-detector backend DeepFace uses to localize the face before classifying
# its emotion. "opencv" (default) is fast and dependency-light; "retinaface" /
# "mtcnn" are more accurate but heavier. Override with DEEPFACE_BACKEND.
_DETECTOR_BACKEND = os.getenv("DEEPFACE_BACKEND", "opencv")
# With enforce_detection=False, DeepFace still returns an emotion distribution
# when its detector is not confident (analyzing the whole frame) instead of
# raising -- so every preprocessed AffectNet crop yields a result. Set
# DEEPFACE_ENFORCE_DETECTION=1 to require a confident face instead (faceless
# crops then surface as no-emotion and hit the retry loop).
_ENFORCE_DETECTION = os.getenv("DEEPFACE_ENFORCE_DETECTION", "0").strip().lower() in (
    "1", "true", "yes",
)


def _load_bgr_image(image_file: str):
    """Read an image file as the BGR ndarray DeepFace.analyze expects."""
    import cv2

    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(f"Could not read image: {image_file}")
    return img


def _extract_emotions(analysis) -> dict[str, float | None]:
    """Pull the per-emotion scores for the most confident analyzed face.

    DeepFace.analyze returns a list of result dicts (one per face) in recent
    versions, or a single dict in older ones; each carries an `emotion` mapping
    on a 0-100 scale. AffectNet images are single faces; if several come back we
    keep the one whose top emotion is most confident. Scores are rescaled to 0-1
    to match the other FER services.
    """
    if isinstance(analysis, dict):
        analysis = [analysis]
    if not analysis:
        return {}

    def _peak(entry):
        scores = (entry or {}).get("emotion") or {}
        return max((v for v in scores.values() if v is not None), default=-1.0)

    best = max(analysis, key=_peak)
    emotions = (best or {}).get("emotion") or {}
    scores: dict[str, float | None] = {}
    for key, value in emotions.items():
        try:
            scores[str(key).strip().lower()] = float(value) / 100.0
        except (TypeError, ValueError):
            scores[str(key).strip().lower()] = None
    return scores


def run_deepface(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    from deepface import DeepFace

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
        print(f"DeepFace: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                img = _load_bgr_image(image_file)
                start_time = time.perf_counter()
                analysis = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    detector_backend=_DETECTOR_BACKEND,
                    enforce_detection=_ENFORCE_DETECTION,
                    silent=True,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw_scores = _extract_emotions(analysis)
                normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            return normalized, latency_ms, error

        # Re-attempt whenever no emotion comes back (load error or no face found
        # when enforce_detection is on).
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

        # Local inference is free; record_service_call prices it from services.yaml
        # ($0/image) so it round-trips with cost_usd=0 and feeds the grand total.
        cost = record_service_call(_TASK_NAME, _SERVICE_NAME, sample_id, count=attempts)
        data["id"].append(f"deepface_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_deepface(affectnet_data, results_path=results_path)
