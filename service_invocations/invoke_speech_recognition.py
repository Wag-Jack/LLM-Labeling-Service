from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re

import pandas as pd
import yaml

from service_invocations.speech_recognition.speech_oracle import generate_oracle_transcripts
from service_invocations.speech_recognition.speech_judge import judge_transcripts
from service_invocations.speech_recognition.wer import compute_wer_counts, compute_wer_summary


_TASK_NAME = "speech_recognition"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
_SERVICES_DIR = _RESULTS_DIR / "services"
_ORACLE_DIR = _RESULTS_DIR / "oracle"
_WER_DIR = _RESULTS_DIR / "wer"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_model(name: str) -> str:
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    return slug or "model"


def _has_oracle_results(oracle_results) -> bool:
    if oracle_results is None:
        return False
    if isinstance(oracle_results, dict):
        return bool(oracle_results)
    empty_attr = getattr(oracle_results, "empty", None)
    if empty_attr is None:
        return True
    return not empty_attr


def _load_enabled_entries(config_path: Path, task_name: str) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("services.yaml root must be a mapping.")

    task_cfg = config.get(task_name, {})
    if not isinstance(task_cfg, dict):
        raise ValueError(f"services.yaml '{task_name}' must be a mapping.")

    enabled = []
    for name, entry in task_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def run_speech_recognition(
    edacc_df: pd.DataFrame,
    services_path: Path | None = None,
    models_path: Path | None = None,
):
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    if edacc_df is None:
        raise ValueError("edacc_df is required. Load data in main before invoking.")

    enabled_services = _load_enabled_entries(services_path, _TASK_NAME)
    if not enabled_services:
        print("--- Skipping speech services (none enabled) ---")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _SERVICES_DIR.mkdir(parents=True, exist_ok=True)
    _ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    _WER_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for service_name in enabled_services:
        print(f"--- {service_name} ---")
        module = import_module(
            f"service_invocations.speech_recognition.services.{service_name}"
        )
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Service script '{service_name}' must define a run(...) function."
            )
        results[service_name] = runner(edacc_df)

    print("--- LLMaaS ---")
    oracle_results = generate_oracle_transcripts(
        edacc_df,
        results_dir=_ORACLE_DIR,
        models_path=models_path,
    )

    if results and _has_oracle_results(oracle_results):
        print("--- WER ---")
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                model_slug = _slugify_model(model_name)
                wer_counts = compute_wer_counts(results, model_oracle, edacc_df)
                wer_counts.to_csv(_WER_DIR / f"wer_counts__{model_slug}.csv", index=False)
                wer_summary = compute_wer_summary(wer_counts, list(results.keys()))
                wer_summary.to_csv(_WER_DIR / f"wer_summary__{model_slug}.csv", index=False)
        else:
            wer_counts = compute_wer_counts(results, oracle_results, edacc_df)
            wer_counts.to_csv(_WER_DIR / "wer_counts.csv", index=False)
            wer_summary = compute_wer_summary(wer_counts, list(results.keys()))
            wer_summary.to_csv(_WER_DIR / "wer_summary.csv", index=False)
    elif results:
        print("--- Skipping WER (no LLM oracle results) ---")
    else:
        print("--- Skipping WER (no speech service results) ---")

    """
    if results:
        print("--- LLM Judging ---")
        judge_transcripts(
            results,
            edacc_df,
            results_dir=_SERVICES_DIR,
            services_path=services_path,
            models_path=models_path,
        )
    """
        
    return results, oracle_results
