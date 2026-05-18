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
#   Response: 200 — translated text is in the response body (synchronous for
#             plain-text inputs). x-requestid is returned as a tracking header
#             but is NOT guaranteed; for large/async jobs the body may be empty
#             and x-requestid can be used to poll.
#             400 model unavailable, 415 unsupported content type, 500 server error
# Poll:    GET   https://api.apptek.com/api/v2/translate/{request_id}
#   Headers:  x-token (required)
#   Response: 200 done — body contains translated text
#             204 still processing — retry
_BASE_URL = "https://api.apptek.com/api/v2/translate"
_POLL_INTERVAL_S = 1.0
_POLL_TIMEOUT_S = 120


def _poll(request_id: str, api_key: str) -> str:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = requests.get(
            f"{_BASE_URL}/{request_id}",
            headers={"x-token": api_key},
            timeout=30,
        )
        if response.status_code == 200:
            return response.text.strip()
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

    # For plain-text inputs the translation is returned synchronously in the
    # body. Fall back to polling only if the body is empty (async/large job).
    translation = response.text.strip()
    if translation:
        return translation

    request_id = response.headers.get("x-requestid")
    if not request_id:
        raise RuntimeError(
            "AppTek returned an empty body with no x-requestid header. "
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


def run(europarl_data):
    return run_apptek_translation(europarl_data)