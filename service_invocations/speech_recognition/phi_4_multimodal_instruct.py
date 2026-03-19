import base64
import json
import os
from pathlib import Path
import re
import time
from urllib import error as url_error
from urllib import request as url_request

import pandas as pd

_MODEL_ID = "phi-4-multimodal-instruct"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
_ID_RE = re.compile(r"(\d+)$")


def _normalize_id(value) -> str:
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


def _read_audio_bytes(path_value) -> bytes:
    if isinstance(path_value, (str, Path)):
        return Path(path_value).read_bytes()
    raise TypeError("audio input must be a filesystem path")


def _infer_audio_format(path_value, default: str = "wav") -> str:
    if isinstance(path_value, (str, Path)):
        suffix = Path(path_value).suffix.lower().lstrip(".")
        if suffix:
            return suffix
    return default


def _build_messages(prompt: str, audio_path) -> list[dict]:
    audio_bytes = _read_audio_bytes(audio_path)
    audio_format = _infer_audio_format(audio_path, "wav")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        "format": audio_format,
                    },
                },
            ],
        }
    ]


def _extract_oracle(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "n/a"
    if not isinstance(payload, dict):
        return "n/a"
    return payload.get("llm_oracle", "n/a")


def _post_json(target_uri: str, api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    req = url_request.Request(
        target_uri,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with url_request.urlopen(req) as resp:
            body = resp.read()
    except url_error.HTTPError as exc:
        body = exc.read()
        detail = body.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Microsoft Phi request failed: HTTP {exc.code} {exc.reason} - {detail}"
        ) from exc
    return json.loads(body.decode("utf-8"))


def _extract_tokens(response: dict) -> tuple[int | None, int | None]:
    usage = response.get("usage") if isinstance(response, dict) else None
    input_tokens = None
    output_tokens = None
    prompt_tokens = None
    completion_tokens = None
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens
    return input_tokens, output_tokens


def run(edacc_data, prompt: str, results_path: Path | None = None):
    if results_path is None:
        results_path = _RESULTS_DIR / "speech_oracle__phi_4_multimodal_instruct.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("MICROSOFT_PHI_KEY")
    target_uri = os.getenv("PHI_TARGET_URI")
    if not api_key or not target_uri:
        raise ValueError("MICROSOFT_PHI_KEY and PHI_TARGET_URI must be set in environment.")

    data = {
        "id": [],
        "llm_oracle": [],
        "latency_ms": [],
        "input_tokens": [],
        "output_tokens": [],
    }

    for _, row in edacc_data.iterrows():
        sample_id = row["id"]
        audio_file = row["audio"]
        print(f"LLM Oracle Transcript (phi_4_multimodal_instruct): {audio_file}")

        messages = _build_messages(prompt, audio_file)
        payload = {
            "model": _MODEL_ID,
            "messages": messages,
        }

        start_time = time.perf_counter()
        response = _post_json(target_uri, api_key, payload)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")

        print(content)
        llm_oracle = _extract_oracle(content)
        input_tokens, output_tokens = _extract_tokens(response)

        data["id"].append(_normalize_id(sample_id))
        data["llm_oracle"].append(llm_oracle)
        data["latency_ms"].append(round(latency_ms, 2))
        data["input_tokens"].append(input_tokens)
        data["output_tokens"].append(output_tokens)

    results_df = pd.DataFrame(data)
    results_df.to_csv(results_path, index=False)
    return results_df


def judge(audio_path, prompt: str) -> str:
    api_key = os.getenv("MICROSOFT_PHI_KEY")
    target_uri = os.getenv("PHI_TARGET_URI")
    if not api_key or not target_uri:
        raise ValueError("MICROSOFT_PHI_KEY and PHI_TARGET_URI must be set in environment.")

    messages = _build_messages(prompt, audio_path)
    payload = {
        "model": _MODEL_ID,
        "messages": messages,
    }
    response = _post_json(target_uri, api_key, payload)

    content = ""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
    return content
