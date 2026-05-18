from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd
import yaml


_TASK_NAME = "emotion_detection"
_RESULTS_DIR = Path.cwd() / "service_invocations" / "results" / "emotion_detection"
_SERVICES_DIR = _RESULTS_DIR / "services"


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


def run_emotion_detection(
    vea_df: pd.DataFrame,
    services_path: Path | None = None,
):
    if services_path is None:
        services_path = Path.cwd() / "config" / "services.yaml"

    if vea_df is None:
        raise ValueError("vea_df is required. Load data in main before invoking.")

    enabled_services = _load_enabled_entries(services_path, _TASK_NAME)
    if not enabled_services:
        print("--- Skipping emotion services (none enabled) ---")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _SERVICES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for service_name in enabled_services:
        print(f"--- {service_name} ---")
        module = import_module(
            f"service_invocations.emotion_detection.services.{service_name}"
        )
        runner = getattr(module, "run", None)
        if runner is None or not callable(runner):
            raise AttributeError(
                f"Service script '{service_name}' must define a run(...) function."
            )
        results[service_name] = runner(vea_df)

    return results
