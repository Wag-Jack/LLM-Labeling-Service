from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
from typing import Any, Callable, Mapping

import requests

# Transient HTTP failures worth retrying. 503 (Service Unavailable), 429
# (rate limited) and the 5xx/timeout family are routinely emitted by the
# hosted FER providers (Face++, Luxand, Azure) under load; a single attempt
# turns those into permanent recorded errors, so we back off and retry.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_RETRIES = int(os.getenv("FER_MAX_RETRIES", "5"))
_BASE_DELAY = float(os.getenv("FER_RETRY_BASE_DELAY", "1.5"))
_MAX_DELAY = float(os.getenv("FER_RETRY_MAX_DELAY", "30.0"))

# request_with_retry only retries the HTTP layer (transient status codes /
# connection errors). A provider can still answer 200 OK with no face / no
# emotion, or fail with a non-retryable error -- in both cases the image ends
# up with no usable result. We re-issue the WHOLE attempt in those cases too,
# up to this many total attempts, to "try to get another result" (the empty
# response is sometimes a transient detector miss that clears on a fresh call).
_EMPTY_RESULT_MAX_ATTEMPTS = int(os.getenv("FER_EMPTY_MAX_ATTEMPTS", "3"))

# Transient 4xx codes that ARE worth retrying (already handled at the HTTP layer
# by request_with_retry); every other 4xx is permanent for the empty-result loop.
_TRANSIENT_4XX = {408, 425, 429}


def _is_permanent_error(error: str | None) -> bool:
    """True if an attempt's error is a permanent 4xx client error.

    Errors captured from requests' ``raise_for_status`` read like
    ``"402 Client Error: Payment Required for url: ..."`` -- a leading 3-digit
    status. A non-transient 4xx means the same request will keep failing no
    matter how many times it is re-issued (bad credentials = 401, out of credits
    = 402, forbidden = 403, not found = 404, malformed = 400), so the empty-
    result loop stops immediately instead of burning attempts + backoff on it.
    Errors with no leading HTTP status (timeouts, parse failures, "no face") are
    NOT permanent and stay retryable.
    """
    if not error:
        return False
    match = re.match(r"\s*(\d{3})\b", error)
    if not match:
        return False
    code = int(match.group(1))
    return 400 <= code < 500 and code not in _TRANSIENT_4XX


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


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    extra_retryable: Callable[[Any], bool] | None = None,
    **kwargs,
):
    """``requests.request`` with exponential backoff on transient failures.

    Retries on connection errors, timeouts, and retryable HTTP status codes
    (503/502/504/429/500/408/425). ``extra_retryable`` lets a caller treat an
    otherwise non-retryable response as transient -- e.g. Face++ returns
    ``403 CONCURRENCY_LIMIT_EXCEEDED`` for rate limiting, which is worth a
    backoff rather than a permanent error. Non-retryable responses and the final
    retryable response are returned unchanged so the caller's
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

        should_retry = response.status_code in _RETRYABLE_STATUS
        if not should_retry and extra_retryable is not None:
            try:
                should_retry = bool(extra_retryable(response))
            except Exception:  # noqa: BLE001 - a faulty predicate must not crash the call
                should_retry = False

        if should_retry and attempt < max_retries:
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


def call_until_emotion(
    attempt: Callable[[], tuple[dict[str, float | None], float | None, str | None]],
    *,
    label: str = "",
    max_attempts: int = _EMPTY_RESULT_MAX_ATTEMPTS,
) -> tuple[dict[str, float | None], float | None, str | None, int]:
    """Re-run a per-image FER attempt until it yields a usable top emotion.

    ``attempt`` performs one complete request for a single image and returns
    ``(normalized, latency_ms, error)``; it must capture provider failures into
    the ``error`` slot rather than raising. An attempt counts as a success when
    ``pick_top_emotion`` finds a non-null score. Otherwise the image came back
    with no emotion -- whether from a recorded error or a face-less / empty
    response -- and we retry with backoff, since both transient provider hiccups
    and flaky face detection sometimes clear on a fresh call.

    Permanent client errors (a non-transient 4xx such as 401/402/403/404) are
    the exception: retrying them never helps, so we stop on the first one rather
    than burn the remaining attempts + backoff (e.g. Imentiv's
    ``402 Insufficient credits`` would otherwise repeat for every image).

    The LAST attempt's result is returned regardless (so the sample is still
    recorded), along with the number of attempts actually made. Each attempt is
    a billed request, so callers should price the call as ``count=attempts``.
    """
    max_attempts = max(1, max_attempts)
    attempts = 0
    normalized: dict[str, float | None] = {}
    latency_ms: float | None = None
    error: str | None = None
    while attempts < max_attempts:
        normalized, latency_ms, error = attempt()
        attempts += 1
        if pick_top_emotion(normalized)[0] is not None:
            break
        if _is_permanent_error(error):
            print(
                f"[fer-retry] {label or 'FER'} -> {error}; not retrying "
                "(permanent error -- check the account/credentials/billing).",
                file=sys.stderr,
                flush=True,
            )
            break
        if attempts < max_attempts:
            delay = _backoff_delay(attempts - 1)
            reason = error or "no emotion returned"
            print(
                f"[fer-retry] {label or 'FER'} -> {reason}; retrying for another "
                f"result in {delay:.1f}s (attempt {attempts + 1}/{max_attempts}).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    return normalized, latency_ms, error, attempts


# Canonical emotion set used across all FER services for cross-service
# comparison. This is the AffectNet-7 class set: the seven discrete
# expression classes AffectNet-7 defines. AffectNet's eighth class,
# "contempt", is intentionally EXCLUDED because AffectNet-7 does not define
# it. Each service's _EMOTION_MAPPING projects provider-specific labels onto
# these canonical names; any provider-specific "contempt" score falls outside
# this set and is dropped during projection (see normalize_emotions). For
# services that emit contempt as part of a probability distribution, the
# remaining seven scores are renormalized back to sum to 1.0 (see
# renormalize_scores). Services that do not natively report every canonical
# emotion have None for the missing entries.
CANONICAL_EMOTIONS = (
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# Provider labels that denote "contempt". AffectNet-7 has no contempt class,
# so these never appear in CANONICAL_EMOTIONS and are dropped on projection.
CONTEMPT_LABELS = frozenset({"contempt"})

# AffectNet-7 ground-truth label mapping. AffectNet numbers its expression
# classes 0-7; AffectNet-7 uses classes 0-6 and omits class 7 ("contempt").
# This follows AffectNet's canonical numeric ordering so the numeric labels
# match the dataset's own convention.
LABEL_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
}

# Inverse of LABEL_MAP, for the dataset loader to turn a (canonicalized)
# class-folder name back into AffectNet's numeric label.
NAME_TO_LABEL = {name: idx for idx, name in LABEL_MAP.items()}

# Documented per-service emotion capabilities, surfaced in the results note so
# the user can see which services cannot report contempt at all (versus those
# whose contempt score is dropped + renormalized away because AffectNet-7 has
# no such class). "multi_emotion" records whether the service returns a full
# per-emotion distribution (required by this pipeline) rather than only a
# single dominant label.
# Reflects behavior verified against the live APIs (June 2026).
SERVICE_EMOTION_CAPABILITIES = {
    "aws_rekognition": {"returns_contempt": False, "multi_emotion": True},
    "faceplusplus": {"returns_contempt": False, "multi_emotion": True},
    "luxand_facesdk": {"returns_contempt": True, "multi_emotion": True},
    # Imentiv returns 8 emotions INCLUDING contempt (dropped + renormalized).
    "imentiv": {"returns_contempt": True, "multi_emotion": True},
    # SkyBiometry is DISABLED (see services.yaml): mood/emotion is not
    # provisioned on the current key -- live-verified that faces/detect returns
    # every attribute except mood + ethnicity. Per the docs it WOULD be
    # multi-emotion when mood is enabled (a confidence per basic emotion alongside
    # the dominant `mood`), and the parser now handles that shape; but the
    # live-verified behavior on this key is "no emotion data", so multi_emotion is
    # recorded False to match reality. No contempt class.
    "skybiometry": {"returns_contempt": False, "multi_emotion": False},
}


def label_to_name(label: Any) -> str | None:
    if label is None:
        return None
    try:
        return LABEL_MAP.get(int(label))
    except (TypeError, ValueError):
        return None


def renormalize_scores(
    scores: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Rescale the non-null canonical scores so they sum to 1.0.

    Used for services that emit a probability *distribution* which included a
    contempt mass: because AffectNet-7 has no contempt class, the contempt
    score is dropped during projection, leaving the remaining seven summing to
    <1. Renormalizing redistributes that dropped mass proportionally so the
    output is again a proper distribution over the AffectNet-7 classes.

    Only meaningful for services whose scores form a distribution. Services
    that report independent per-emotion confidences (e.g. Rekognition,
    SkyBiometry) should NOT renormalize, as rescaling would distort their
    native semantics. None entries are preserved as None.
    """
    present = {k: v for k, v in scores.items() if v is not None}
    total = sum(present.values())
    if total <= 0:
        return dict(scores)
    return {k: (v / total if v is not None else None) for k, v in scores.items()}


def contempt_was_reported(
    scores: Mapping[str, float | int | None] | None,
    mapping: Mapping[str, str] | None = None,
) -> bool:
    """True if the raw provider scores carried a (non-null) contempt value.

    Lets a service detect at runtime that the provider actually returned a
    contempt score (which is then dropped, since AffectNet-7 omits contempt),
    independent of the documented SERVICE_EMOTION_CAPABILITIES table.
    """
    if not scores:
        return False
    for key, value in scores.items():
        canonical = mapping[key] if mapping and key in mapping else key
        if str(canonical).strip().lower() in CONTEMPT_LABELS and value is not None:
            return True
    return False


def normalize_emotions(
    scores: Mapping[str, float | int | None] | None,
    mapping: Mapping[str, str] | None = None,
    renormalize: bool = False,
) -> dict[str, float | None]:
    """
    Project provider-specific emotion scores onto CANONICAL_EMOTIONS.

    Any provider score whose canonical name is not in CANONICAL_EMOTIONS (most
    notably "contempt", which AffectNet-7 does not define) is dropped here.

    Scores are otherwise kept on their native scale (most providers report 0-1
    or 0-100); downstream comparison treats them as relative within a service,
    and pick_top_emotion retrieves the dominant emotion regardless of scale.

    Set ``renormalize=True`` only for services that return a probability
    distribution INCLUDING contempt, so the remaining seven classes are
    rescaled back to sum to 1.0 after contempt is dropped (see
    renormalize_scores).
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

    if renormalize:
        normalized = renormalize_scores(normalized)
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


def preprocess_face_image(
    image_bytes: bytes,
    pad_ratio: float = 0.4,
    min_size: int = 0,
) -> bytes:
    """Standardize a face crop the SAME way for every FER service AND the LLM
    paradigms, so all systems in the study see identical input (fairness).

    Adds a uniform gray border (``pad_ratio`` of the larger dimension) so face
    detectors that need surrounding context can localize tightly-cropped faces
    (AffectNet ships 96x96 crops; Imentiv's detector finds 0 faces without a
    margin). Optionally upscales so the larger dimension is at least
    ``min_size`` (0 = no upscale; padding alone preserves the native face
    resolution and was verified sufficient for detection across all services).

    Returns PNG bytes; falls back to the original bytes if Pillow is missing or
    anything fails, so a preprocessing problem never drops a sample.
    """
    try:
        from PIL import Image, ImageOps

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if min_size and max(im.size) < min_size:
            scale = min_size / max(im.size)
            im = im.resize(
                (round(im.width * scale), round(im.height * scale)), Image.LANCZOS
            )
        pad = int(max(im.size) * pad_ratio)
        if pad > 0:
            im = ImageOps.expand(im, border=pad, fill=(128, 128, 128))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - preprocessing is best-effort
        return image_bytes


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
