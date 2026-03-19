from importlib import import_module
from pathlib import Path
import re

import pandas as pd
import yaml

_PROMPT = """
Please give me a transcript for the following audio file.
You MUST return ONLY valid JSON. Do not include markdown, code fences, or explanations.
JSON schema:
{
  "llm_oracle": string|null
}
If you do not receive the WAV file, enter llm_oracle as 'n/a'.
Do NOT mention that you need the WAV file, only give the JSON schema output.
If you violate this, the output will be discarded.
"""

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_model(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "model"


def _load_enabled_models(models_path: Path, task_name: str) -> list[str]:
    if not models_path.exists():
        raise FileNotFoundError(f"Models config not found: {models_path}")
    with models_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("models.yaml root must be a mapping.")

    task_cfg = config.get(task_name, {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"models.yaml '{task_name}' must be a mapping.")

    enabled = []
    for name, entry in task_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def generate_oracle_transcripts(
    edacc_data,
    use_existing: bool = False,
    results_dir: Path | None = None,
    models_path: Path | None = None,
    task_name: str = "speech_recognition",
):
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    results_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = _load_enabled_models(models_path, task_name)
    if not enabled_models:
        print("Skipping LLM Oracle Transcript (no enabled models).")
        return {}

    results_by_model: dict[str, pd.DataFrame] = {}
    multiple_models = len(enabled_models) > 1

    for model_name in enabled_models:
        model_slug = _slugify_model(model_name)
        results_path = results_dir / (
            f"speech_oracle__{model_slug}.csv" if multiple_models else "speech_oracle.csv"
        )

        if use_existing and results_path.exists():
            results_by_model[model_name] = pd.read_csv(results_path)
            continue

        module = import_module(f"service_invocations.speech_recognition.{model_name}")
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Model script '{model_name}' must define a run(...) function."
            )
        results_by_model[model_name] = runner(
            edacc_data,
            prompt=_PROMPT,
            results_path=results_path,
        )

    if multiple_models:
        return results_by_model
    return next(iter(results_by_model.values()))
