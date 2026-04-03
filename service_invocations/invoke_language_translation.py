from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re

import pandas as pd
import yaml

from service_invocations.language_translation.comet import compute_comet_scores, compute_comet_summary
from service_invocations.language_translation.language_oracle import generate_oracle_translations


_TASK_NAME = "language_translation"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "language_translation"
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


def run_language_translation(
    europarl_df: pd.DataFrame,
    services_path: Path | None = None,
    models_path: Path | None = None,
):
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"

    if europarl_df is None:
        raise ValueError("europarl_df is required. Load data in main before invoking.")

    enabled_services = _load_enabled_entries(services_path, _TASK_NAME)
    if not enabled_services:
        print("--- Skipping translation services (none enabled) ---")
        return {}, None

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for service_name in enabled_services:
        print(f"--- {service_name} ---")
        module = import_module(
            f"service_invocations.language_translation.services.{service_name}"
        )
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Service script '{service_name}' must define a run(...) function."
            )
        results[service_name] = runner(europarl_df)

    print("--- LLM Oracle Translation ---")
    oracle_results = generate_oracle_translations(
        europarl_df,
        results_dir=_RESULTS_DIR,
        models_path=models_path,
    )

    if results and _has_oracle_results(oracle_results):
        print("--- COMET ---")
        if isinstance(oracle_results, dict):
            for model_name, model_oracle in oracle_results.items():
                model_slug = _slugify_model(model_name)
                comet_scores = compute_comet_scores(results, model_oracle, europarl_df)
                comet_scores.to_csv(_RESULTS_DIR / f"comet_scores__{model_slug}.csv", index=False)
                comet_summary = compute_comet_summary(comet_scores, list(results.keys()))
                comet_summary.to_csv(_RESULTS_DIR / f"comet_summary__{model_slug}.csv", index=False)
        else:
            comet_scores = compute_comet_scores(results, oracle_results, europarl_df)
            comet_scores.to_csv(_RESULTS_DIR / "comet_scores.csv", index=False)
            comet_summary = compute_comet_summary(comet_scores, list(results.keys()))
            comet_summary.to_csv(_RESULTS_DIR / "comet_summary.csv", index=False)
    elif results:
        print("--- Skipping COMET (no LLM oracle results) ---")
    else:
        print("--- Skipping COMET (no translation service results) ---")

    return results, oracle_results
