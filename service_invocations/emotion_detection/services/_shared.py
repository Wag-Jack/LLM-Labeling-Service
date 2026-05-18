from __future__ import annotations

import json
from typing import Any, Mapping

# Canonical emotion set used across all FER services for cross-service
# comparison. Each service's _EMOTION_MAPPING projects provider-specific
# labels onto these canonical names. Services that do not natively report
# every canonical emotion will have None for the missing entries.
CANONICAL_EMOTIONS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# VEA dataset ground-truth label mapping.
LABEL_MAP = {
    0: "anger",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise",
}


def label_to_name(label: Any) -> str | None:
    if label is None:
        return None
    try:
        return LABEL_MAP.get(int(label))
    except (TypeError, ValueError):
        return None


def normalize_emotions(
    scores: Mapping[str, float | int | None] | None,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, float | None]:
    """
    Project provider-specific emotion scores onto CANONICAL_EMOTIONS.

    Scores are kept on their native scale (most providers report 0-1 or
    0-100); downstream comparison treats them as relative within a service.
    Use pick_top_emotion to retrieve the dominant emotion regardless of scale.
    """
    normalized: dict[str, float | None] = {k: None for k in CANONICAL_EMOTIONS}
    if not scores:
        return normalized

    for key, value in scores.items():
        canonical = mapping[key] if mapping and key in mapping else key
        if canonical not in normalized:
            continue
        if value is None:
            normalized[canonical] = None
            continue
        try:
            normalized[canonical] = float(value)
        except (TypeError, ValueError):
            normalized[canonical] = None
    return normalized


def pick_top_emotion(
    scores: Mapping[str, float | None],
) -> tuple[str | None, float | None]:
    """Return the (name, score) of the highest-scoring canonical emotion."""
    best_key: str | None = None
    best_value: float | None = None
    for key, value in scores.items():
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_key = key
            best_value = value
    return best_key, best_value


def build_service_output(
    emotions: Mapping[str, float | None],
    error: str | None = None,
) -> str:
    """
    Serialize the canonical FER service output payload.

    The payload contains the per-emotion scores plus a top_emotion summary
    so downstream comparators (LLM judge, SDS, majority voting) can read a
    single column without re-deriving the dominant emotion.
    """
    top_name, top_score = pick_top_emotion(emotions)
    payload: dict[str, Any] = {
        "emotions": dict(emotions),
        "top_emotion": {"name": top_name, "score": top_score},
    }
    if error is not None:
        payload["error"] = error
    return json.dumps(payload, ensure_ascii=True)
