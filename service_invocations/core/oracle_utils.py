import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

_ID_RE = re.compile(r"(\d+)$")
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

_NULLISH_STRINGS = {"", "n/a", "na", "null", "none", "nan"}

_DEFAULT_OUTPUT_RETRIES = int(os.getenv("LLM_OUTPUT_RETRIES", "3"))
_DEFAULT_OUTPUT_RETRY_DELAY = float(os.getenv("LLM_OUTPUT_RETRY_DELAY", "1.0"))

T = TypeVar("T")


def load_prompt(path: Path | str, **substitutions: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def resolve_prompt_path(prompts_root: Path, paradigm: str, prompt_name: str) -> Path:
    path = Path(prompts_root) / paradigm / f"{prompt_name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found for paradigm '{paradigm}' at {path}"
        )
    return path


def _strip_code_fence(content: str) -> str:
    match = _FENCE_RE.match(content)
    return match.group(1) if match else content


def parse_json_payload(content: str) -> dict:
    """Strip an optional ```json ... ``` fence and parse the response as a dict.

    Returns ``{}`` when the content is missing, malformed, or not a JSON object.
    Use this instead of bare ``json.loads(resp.content)`` so callers tolerate
    models that wrap their output in markdown fences.
    """
    if content is None:
        return {}
    try:
        payload = json.loads(_strip_code_fence(content).strip())
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_oracle(content: str, key: str = "llm_oracle") -> Any:
    """Extract the oracle payload from a model response.

    ``key`` is the JSON field the prompt instructs the model to return
    (e.g. ``"transcript"`` for ASR, ``"translation"`` for MT, ``"scores"``
    for emotion). If the response can't be parsed or the key is missing,
    returns ``"n/a"`` so ``is_nullish_output`` can route it through retry.
    """
    if content is None:
        return "n/a"
    candidate = _strip_code_fence(content).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return "n/a"
    if not isinstance(payload, dict):
        return "n/a"
    return payload.get(key, "n/a")


def normalize_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):04d}"
    value_str = str(value)
    match = _ID_RE.search(value_str)
    if match:
        digits = match.group(1)
        if len(digits) <= 4:
            return digits.zfill(4)
        return digits
    return value_str


_extract_oracle = extract_oracle
_normalize_id = normalize_id


_TRUTHY = {"1", "true", "yes", "on"}


def is_fresh_run_requested(explicit: bool = False) -> bool:
    """True when callers want resume disabled — either via explicit param or
    the ``LLM_FRESH_RUN`` env var. Centralized here so every runner agrees on
    the parsing rules."""
    if explicit:
        return True
    return os.getenv("LLM_FRESH_RUN", "").strip().lower() in _TRUTHY


def is_nullish_output(value: Any) -> bool:
    """True when an LLM-derived value is missing/null/empty.

    Handles the common payload shapes we extract from model JSON:
      * ``None`` and ``NaN`` floats → nullish
      * strings matching ``_NULLISH_STRINGS`` (case-insensitive) → nullish
      * empty dicts, or dicts whose every value is itself nullish → nullish
        (this catches the "model couldn't process input, returned all-null
        scores" case, which would otherwise pass validation)
      * empty lists, or lists whose every entry is nullish → nullish
    Anything else is treated as a real value.
    """
    if value is None:
        return True
    if isinstance(value, float):
        # NaN guards
        if value != value:  # noqa: PLR0124 (NaN check)
            return True
        return False
    if isinstance(value, str):
        return value.strip().lower() in _NULLISH_STRINGS
    if isinstance(value, dict):
        if not value:
            return True
        return all(is_nullish_output(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        if not value:
            return True
        return all(is_nullish_output(v) for v in value)
    return False


def judge_output_usable(parsed: Any, services: Any) -> bool:
    """Whether a judge / human-loop LLM output is a usable label.

    Usable means the model named a valid winner that is one of ``services``
    *and* assigned at least one real (non-negative) score. A billed response
    that lacks these (no winner, or all scores == -1) still counts toward
    total cost but is recorded as a "failed" call for the successful-cost split.
    """
    if not isinstance(parsed, dict):
        return False
    winner = parsed.get("winner")
    winner_ok = winner is not None and winner in services
    scores = parsed.get("scores")
    scores_ok = isinstance(scores, dict) and any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
        for v in scores.values()
    )
    return bool(winner_ok and scores_ok)


def retry_until_valid(
    call: Callable[[], T],
    validate: Callable[[T], bool],
    *,
    description: str,
    max_attempts: int = _DEFAULT_OUTPUT_RETRIES,
    base_delay: float = _DEFAULT_OUTPUT_RETRY_DELAY,
    on_attempt: Callable[[T, bool], None] | None = None,
) -> T:
    """Call ``call()`` up to ``max_attempts`` times until ``validate(result)``.

    If every attempt fails validation, returns the most recent result and logs
    a warning. The caller decides what to do with the (best-effort) value —
    typically: record it so downstream tabulation can still see *something*,
    rather than crashing the run.

    ``on_attempt(result, ok)``, if provided, is invoked after *every* attempt
    that returns a result (with ``ok`` = whether it passed ``validate``). This
    lets callers account for the cost of every billed call, including the
    invalid attempts that are otherwise discarded. It is not called when
    ``call()`` raises (e.g. a 503 that produced no response), so failed network
    calls are never counted.
    """
    last_result: Any = None
    for attempt in range(max_attempts):
        last_result = call()
        try:
            ok = validate(last_result)
        except Exception as exc:
            ok = False
            print(
                f"[output-retry] {description}: validator raised "
                f"{type(exc).__name__}: {exc}. Treating as invalid.",
                file=sys.stderr,
                flush=True,
            )
        if on_attempt is not None:
            try:
                on_attempt(last_result, ok)
            except Exception as exc:
                print(
                    f"[output-retry] {description}: on_attempt callback raised "
                    f"{type(exc).__name__}: {exc}. Continuing.",
                    file=sys.stderr,
                    flush=True,
                )
        if ok:
            return last_result
        if attempt < max_attempts - 1:
            delay = base_delay * (2 ** attempt)
            print(
                f"[output-retry] {description}: invalid/null output "
                f"(attempt {attempt + 1}/{max_attempts}). "
                f"Retrying in {delay:.1f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    print(
        f"[output-retry] {description}: gave up after {max_attempts} attempt(s); "
        f"using last (invalid) result so downstream tabulation can proceed.",
        file=sys.stderr,
        flush=True,
    )
    return last_result
