from __future__ import annotations

import json
from typing import Any, Mapping

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
    normalized: dict[str, float | None] = {k: None for k in CANONICAL_EMOTIONS}
    if not scores:
        return normalized

    for key, value in scores.items():
        if mapping and key in mapping:
            canonical = mapping[key]
        else:
            canonical = key
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


def pick_top_emotion(scores: Mapping[str, float | None]) -> str | None:
    best_key = None
    best_value = None
    for key, value in scores.items():
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_key = key
            best_value = value
    return best_key


def format_service_output(raw: Any, emotions: Mapping[str, float | None]) -> str:
    payload = {
        "raw": raw,
        "emotions": emotions,
    }
    return json.dumps(payload, ensure_ascii=True)
