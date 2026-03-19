import json
import os
from pathlib import Path
import re
import time

import pandas as pd

_MODEL_ID = "gemini-2.5-flash-lite"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation"
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


def _extract_oracle(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "n/a"
    if not isinstance(payload, dict):
        return "n/a"
    return payload.get("llm_oracle", "n/a")


def _load_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment.")

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception:
        genai = None
        genai_types = None

    if genai is not None and genai_types is not None:
        client = genai.Client(api_key=api_key)
        return "genai", client, genai_types

    try:
        import google.generativeai as genai_legacy  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Gemini client requires `google-genai` or `google-generativeai`."
        ) from exc

    genai_legacy.configure(api_key=api_key)
    return "generativeai", genai_legacy, None


def run(europarl_data, prompt: str, results_path: Path | None = None):
    if results_path is None:
        results_path = _RESULTS_DIR / "language_oracle__gemini_2_5_flash_lite.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    mode, client, types = _load_client()

    data = {
        "id": [],
        "llm_oracle": [],
        "latency_ms": [],
        "input_tokens": [],
        "output_tokens": [],
    }

    for _, row in europarl_data.iterrows():
        sample_id = row["id"]
        english = row["english"]
        print(f"LLM Oracle Translation (gemini_2_5_flash_lite): {english}")

        start_time = time.perf_counter()
        if mode == "genai":
            parts = [
                types.Part.from_text(prompt),
                types.Part.from_text(english),
            ]
            contents = [types.Content(role="user", parts=parts)]
            response = client.models.generate_content(
                model=_MODEL_ID,
                contents=contents,
            )
            content = getattr(response, "text", None)
            if content is None and getattr(response, "candidates", None):
                candidate = response.candidates[0]
                content_parts = getattr(candidate, "content", None)
                if content_parts and getattr(content_parts, "parts", None):
                    content = getattr(content_parts.parts[0], "text", None)
            if content is None:
                content = str(response)
        else:
            model_client = client.GenerativeModel(_MODEL_ID)
            parts = [prompt, english]
            response = model_client.generate_content(parts)
            content = getattr(response, "text", None) or str(response)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        print(content)
        llm_oracle = _extract_oracle(content)

        data["id"].append(_normalize_id(sample_id))
        data["llm_oracle"].append(llm_oracle)
        data["latency_ms"].append(round(latency_ms, 2))
        data["input_tokens"].append(None)
        data["output_tokens"].append(None)

    results_df = pd.DataFrame(data)
    results_df.to_csv(results_path, index=False)
    return results_df
