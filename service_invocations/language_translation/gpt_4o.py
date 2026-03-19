import json
import os
from pathlib import Path
import re
import time

import pandas as pd
from openai import OpenAI

_MODEL_ID = "gpt-4o"
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


def _build_messages(prompt: str, text_input: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": text_input},
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


def _extract_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens
    return input_tokens, output_tokens


def run(europarl_data, prompt: str, results_path: Path | None = None):
    if results_path is None:
        results_path = _RESULTS_DIR / "language_oracle__gpt_4o.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        print(f"LLM Oracle Translation (gpt_4o): {english}")

        messages = _build_messages(prompt, english)
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=_MODEL_ID,
            modalities=["text"],
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = response.choices[0].message.content
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
