from __future__ import annotations

from importlib import import_module
from pathlib import Path
import json
import re

import pandas as pd
import yaml

from service_invocations.core.oracle_utils import normalize_id as _normalize_id

_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
_TASK_NAME = "speech_recognition"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_model(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "model"


def _load_enabled_entries(config_path: Path, task_name: str) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")

    task_cfg = config.get(task_name, {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"Config '{task_name}' must be a mapping.")

    enabled = []
    for name, entry in task_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def judge_transcripts(
    results_by_service: dict[str, pd.DataFrame],
    edacc_data: pd.DataFrame,
    results_dir: Path | None = None,
    services_path: Path | None = None,
    models_path: Path | None = None,
    task_name: str = _TASK_NAME,
):
    if results_dir is None:
        results_dir = _RESULTS_DIR
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    results_dir.mkdir(parents=True, exist_ok=True)

    enabled_services = _load_enabled_entries(services_path, task_name)
    enabled_models = _load_enabled_entries(models_path, task_name)

    if not enabled_models:
        print("--- Skipping LLM Judging (no enabled models) ---")
        return None

    services = {
        name: df for name, df in results_by_service.items() if name in enabled_services
    }
    if not services:
        print("--- Skipping LLM Judging (no enabled service results) ---")
        return None

    transcripts_by_service = {}
    for name, df in services.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        transcripts_by_service[name] = dict(
            zip(df["id"].map(_normalize_id), df["service_output"])
        )

    wav_files = edacc_data["audio"].tolist()
    ids = edacc_data["id"].tolist()

    def build_prompt(service_blocks: str) -> str:
        return f"""
You are evaluating multiple speech-to-text services.
Tasks:
1. Listen to the audio.
2. Provide your own transcript (llm_transcript).
3. Score each service transcript from 1.0 to 10.0 (step size 0.1).
Return ONLY valid JSON with this schema:
{{
  \"llm_transcript\": string|null,
  \"scores\": {{ \"<service_name>\": number }}
}}
If the audio is missing, set llm_transcript to 'n/a' and all scores to -1.
Service transcripts:
{service_blocks}
"""

    for model_name in enabled_models:
        module = import_module(f"service_invocations.models.{model_name}")
        generator = getattr(module, "generate", None)
        if generator is None or not callable(generator):
            raise AttributeError(
                f"Model script '{model_name}' must define a generate(...) function."
            )

        model_slug = _slugify_model(model_name)
        score_column = "llm_judge_score" if len(enabled_models) == 1 else f"llm_judge_score__{model_slug}"
        results_file_name = "speech_results.csv" if len(enabled_models) == 1 else f"speech_results__{model_slug}.csv"

        data = {
            "id": [],
            "llm_transcript": [],
            "scores": [],
        }

        for sample_id, wav in zip(ids, wav_files):
            id_key = _normalize_id(sample_id)
            print(f"LLM Judging ({model_name}): {wav}")

            service_blocks = "\n".join(
                f"{name}: {transcripts_by_service[name].get(id_key, '')}"
                for name in services.keys()
            )
            prompt = build_prompt(service_blocks)

            response = generator(prompt, inputs={"audio": wav})
            content = response.content
            print(content)

            default_scores = {name: -1 for name in services.keys()}
            try:
                llm_output = json.loads(content)
                if isinstance(llm_output, dict):
                    scores = llm_output.get("scores", {})
                else:
                    scores = {}
            except json.JSONDecodeError:
                llm_output = {"llm_transcript": "n/a"}
                scores = {}

            if not isinstance(scores, dict):
                scores = {}

            merged_scores = {**default_scores, **scores}
            data["id"].append(id_key)
            data["llm_transcript"].append(llm_output.get("llm_transcript", "n/a"))
            data["scores"].append(merged_scores)

        scores_by_service: dict[str, dict[str, float]] = {name: {} for name in services.keys()}
        for id_key, score_map in zip(data["id"], data["scores"]):
            for name, score in score_map.items():
                if name in scores_by_service:
                    scores_by_service[name][id_key] = score

        for name, df in services.items():
            score_map = scores_by_service.get(name, {})
            df = df.drop(columns=[score_column], errors="ignore")
            df[score_column] = df["id"].map(_normalize_id).map(score_map).fillna(-1)

            service_module = import_module(
                f"service_invocations.speech_recognition.services.{name}"
            )
            results_file = getattr(service_module, "RESULTS_FILE", f"{name}.csv")
            df.to_csv(results_dir / results_file, index=False)

        judge_results = pd.DataFrame({
            "id": data["id"],
            "llm_transcript": data["llm_transcript"],
            "scores": data["scores"],
        })
        judge_results.to_csv(results_dir / results_file_name, index=False)

    return True
