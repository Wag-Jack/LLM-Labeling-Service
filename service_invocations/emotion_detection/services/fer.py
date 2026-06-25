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
_SERVICE_NAME = "fer"

_RESULTS_DIR = (
    Path.cwd()
    / "service_invocations"
    / "results"
    / "emotion_detection"
    / "services"
)
RESULTS_FILE = "fer.csv"

# FER is the local, open-source `fer` library (Justin Shenk's package, built on a
# Keras mini-Xception model). It runs entirely on-device -- no API key, no
# network call -- so there is no per-image cloud cost (priced at $0 in
# services.yaml). detect_emotions() returns a list with one entry per detected
# face; each carries an `emotions` dict over the SEVEN labels below, already on a
# 0-1 scale summing to ~1.0. There is NO contempt class, so nothing is dropped
# and no renormalization is needed (renormalize=False).
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

# Cache the (expensive to build) detector across all images in a run.
_DETECTOR = None


def _get_detector():
    """Build the FER detector once and reuse it for every image.

    The default OpenCV Haar-cascade face detector is fast and dependency-light;
    set FER_MTCNN=1 to use the (slower, more accurate) MTCNN detector instead,
    which can register faces the cascade misses on harder crops.
    """
    global _DETECTOR
    if _DETECTOR is None:
        # The class lived at `fer.FER` in older releases and moved to
        # `fer.fer.FER` in the 25.x rewrite; import from whichever is present.
        try:
            from fer import FER
        except ImportError:
            from fer.fer import FER

        use_mtcnn = os.getenv("FER_MTCNN", "0").strip().lower() in ("1", "true", "yes")
        _DETECTOR = FER(mtcnn=use_mtcnn)
    return _DETECTOR


def _load_bgr_image(image_file: str):
    """Read an image file as the BGR ndarray the FER detector expects."""
    import cv2

    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(f"Could not read image: {image_file}")
    return img


def _extract_emotions(detections) -> dict[str, float | None]:
    """Pull the per-emotion scores for the most confident detected face.

    detect_emotions() returns a list of {"box", "emotions"} entries (one per
    face) or an empty list when no face is found. AffectNet images are single
    faces; if several are detected we keep the one whose top emotion is most
    confident. An empty list -> no face -> empty dict (triggers the retry loop).
    """
    if not detections:
        return {}

    def _peak(entry):
        scores = entry.get("emotions") or {}
        return max((v for v in scores.values() if v is not None), default=-1.0)

    best = max(detections, key=_peak)
    emotions = best.get("emotions") or {}
    return {str(k).strip().lower(): v for k, v in emotions.items()}


def run_fer(affectnet_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    detector = _get_detector()

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
        print(f"FER: {image_file}")

        def _attempt():
            latency_ms: float | None = None
            error: str | None = None
            normalized: dict[str, float | None] = {}
            try:
                img = _load_bgr_image(image_file)
                start_time = time.perf_counter()
                detections = detector.detect_emotions(img)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw_scores = _extract_emotions(detections)
                normalized = normalize_emotions(raw_scores, mapping=_EMOTION_MAPPING)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            return normalized, latency_ms, error

        # Re-attempt whenever no emotion comes back (load error or no face found).
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
        data["id"].append(f"fer_{sample_id:04d}")
        data["label"].append(label)
        data["label_name"].append(label_to_name(label))
        data["latency_ms"].append(None if latency_ms is None else round(latency_ms, 2))
        data["cost_usd"].append(cost)
        data["service_output"].append(build_service_output(normalized, error=error))

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(affectnet_data, results_path=None):
    return run_fer(affectnet_data, results_path=results_path)
