import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation" / "services"
RESULTS_FILE = "apptek_trans.csv"

# AppTek Machine Translation REST API
# Submit:  POST  https://api.apptek.com/api/v2/translate/{model}
#   Headers:  x-token (required), Content-Type: text/plain (required)
#   Body:     raw UTF-8 text
#   Response: 200 — the job is ACCEPTED asynchronously. The body is a JSON
#             envelope {"request_id": "...", "success": true, "error": null};
#             the translation is NOT in this response. Use request_id to poll.
#             400 model unavailable, 415 unsupported content type, 500 server error
# Poll:    GET   https://api.apptek.com/api/v2/translate/{request_id}
#   Headers:  x-token (required)
#   Response: 200 done — body contains the translated text
#             204 still processing — retry
_BASE_URL = "https://api.apptek.com/api/v2/translate"
_POLL_INTERVAL_S = 1.0
_POLL_TIMEOUT_S = 120

# Keys AppTek may wrap a finished translation under, if the poll endpoint
# returns JSON rather than raw text. Checked before treating a body as plain
# text so we never mistake an envelope for a translation.
_TRANSLATION_KEYS = ("translation", "translated_text", "text", "result", "output", "target")
# Keys that identify an async-acceptance envelope (no translation yet).
_ENVELOPE_KEYS = ("request_id", "requestId", "success")


def _extract_translation(body_text: str) -> str | None:
    """Return the translation from a response body, or None if it's an async envelope.

    AppTek's submit (and sometimes poll-while-processing) responses are JSON
    envelopes carrying a request_id rather than a translation. A finished
    translation is returned either as raw text or under one of a few known
    JSON keys. We return None when the body is empty or is an envelope so the
    caller knows to keep polling.
    """
    text = (body_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Not JSON -> a real translation came back as raw text.
        return text
    if isinstance(payload, dict):
        for key in _TRANSLATION_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # JSON object without a translation field -> async envelope, not done.
        return None
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return None


def _poll(request_id: str, api_key: str) -> str:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = requests.get(
            f"{_BASE_URL}/{request_id}",
            headers={"x-token": api_key},
            timeout=30,
        )
        if response.status_code == 200:
            translation = _extract_translation(response.text)
            if translation is not None:
                return translation
            # 200 but still an envelope without a translation: keep waiting.
            time.sleep(_POLL_INTERVAL_S)
            continue
        if response.status_code == 204:
            time.sleep(_POLL_INTERVAL_S)
            continue
        response.raise_for_status()
    raise TimeoutError(
        f"AppTek translation timed out after {_POLL_TIMEOUT_S}s (request_id={request_id})."
    )


def _translate(text: str, model: str, api_key: str) -> str:
    response = requests.post(
        f"{_BASE_URL}/{model}",
        headers={
            "x-token": api_key,
            "Content-Type": "text/plain",
        },
        data=text.encode("utf-8"),
        timeout=30,
    )
    response.raise_for_status()

    # The submit response is an async-acceptance envelope; the translation has
    # to be polled for using the returned request_id. Only treat the body as a
    # finished translation if it clearly isn't an envelope.
    translation = _extract_translation(response.text)
    if translation is not None:
        return translation

    request_id: str | None = None
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        if payload.get("success") is False or payload.get("error"):
            raise RuntimeError(
                f"AppTek submit returned an error: {payload.get('error')!r} "
                f"(success={payload.get('success')!r})."
            )
        request_id = payload.get("request_id") or payload.get("requestId")
    if not request_id:
        request_id = response.headers.get("x-requestid")
    if not request_id:
        raise RuntimeError(
            "AppTek returned no translation and no request_id to poll. "
            "Check that the model name is correct and the account has access."
        )
    return _poll(request_id, api_key)


def run_apptek_translation(europarl_data, results_path: Path | None = None):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = _RESULTS_DIR / RESULTS_FILE

    api_key = os.getenv("APPTEK_API_KEY")
    if not api_key:
        raise ValueError("APPTEK_API_KEY must be set in environment.")

    # Model name encodes the language pair. List available models via:
    # GET https://api.apptek.com/api/v2/translate/models (x-token header required)
    model = os.getenv("APPTEK_MODEL", "en-fr")

    data = {
        "id": [],
        "english_input": [],
        "service_output": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"AppTek Translate: ({sample_id:04d}) {english}")

        french = _translate(english, model, api_key)
        print(french)

        data["id"].append(f"apptek_trans_{sample_id:04d}")
        data["english_input"].append(english)
        data["service_output"].append(french)

    df = pd.DataFrame(data)
    df.to_csv(results_path, index=False)
    return df


def run(europarl_data, results_path=None):
    return run_apptek_translation(europarl_data, results_path=results_path)