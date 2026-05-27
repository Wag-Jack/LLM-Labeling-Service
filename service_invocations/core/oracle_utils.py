import json
import re
from pathlib import Path

_ID_RE = re.compile(r"(\d+)$")
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


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


def extract_oracle(content: str) -> str:
    if content is None:
        return "n/a"
    candidate = _strip_code_fence(content).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return "n/a"
    if not isinstance(payload, dict):
        return "n/a"
    return payload.get("llm_oracle", "n/a")


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
