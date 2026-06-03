from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Mapping

import requests

# Transient HTTP failures worth retrying. 503 (Service Unavailable), 429
# (rate limited) and the 5xx/timeout family are routinely emitted by the
# hosted FER providers (Face++, Luxand, Azure) under load; a single attempt
# turns those into permanent recorded errors, so we back off and retry.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_RETRIES = int(os.getenv("FER_MAX_RETRIES", "5"))
_BASE_DELAY = float(os.getenv("FER_RETRY_BASE_DELAY", "1.5"))
_MAX_DELAY = float(os.getenv("FER_RETRY_MAX_DELAY", "30.0"))


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _backoff_delay(attempt: int) -> float:
    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    return delay + random.uniform(0, max(0.25, delay * 0.25))


def request_with_retry(method: str, url: str, *, max_retries: int = _MAX_RETRIES, **kwargs):
    """``requests.request`` with exponential backoff on transient failures.

    Retries on connection errors, timeouts, and retryable HTTP status codes
    (503/502/504/429/500/408/425). Non-retryable responses (e.g. 401/403) and
    the final retryable response are returned unchanged so the caller's
    ``raise_for_status()`` still records a permanent error. ``Retry-After`` is
    honored when present.
    """
    attempt = 0
    while True:
        try:
            response = requests.request(method, url, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            delay = _backoff_delay(attempt)
            print(
                f"[fer-retry] {method} {url} failed "
                f"({type(exc).__name__}); retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries + 1}).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
            continue

        if response.status_code in _RETRYABLE_STATUS and attempt < max_retries:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            delay = retry_after if retry_after is not None else _backoff_delay(attempt)
            delay = min(delay, _MAX_DELAY)
            print(
                f"[fer-retry] {method} {url} -> HTTP {response.status_code}; "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries + 1}).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            attempt += 1
            continue

        return response

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
