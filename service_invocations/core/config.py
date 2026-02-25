from __future__ import annotations  # Use modern type hints without forward refs.

# Centralized YAML config loader for the project.
# This module is intentionally small so other code can depend on a single,
# well-defined place to read/validate config values.
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load the YAML config file and return it as a dict.
    Raises helpful errors if the file is missing or malformed.
    """
    config_path = Path(path)  # Normalize to Path for consistent file handling.
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:  # UTF-8 for YAML portability.
        data = yaml.safe_load(f) or {}

    # Expect a YAML mapping at the root (e.g. service_sets/runtime/metrics).
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    return data  # Dict shape validated above.


def get_service_set(config: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    """
    Return a named service set from the config.
    Example: get_service_set(config, "speech_stt_v1")
    """
    service_sets = config.get("service_sets", {})  # Optional; defaults to empty.
    if not isinstance(service_sets, dict):
        raise ValueError("service_sets must be a mapping.")

    service_set = service_sets.get(name)  # List of service entries for this set.
    # Fail fast if the set doesn't exist; helps catch typos early.
    if service_set is None:
        raise KeyError(f"Unknown service set: {name}")
    if not isinstance(service_set, list):
        raise ValueError("service_sets entries must be lists.")

    return service_set  # Caller can filter enabled services.


def get_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return runtime settings (timeouts, retries, concurrency).
    """
    runtime = config.get("runtime", {})  # Optional runtime parameters.
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping.")
    return runtime  # No defaults imposed here.


def get_metrics_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return metrics settings (output paths, cost tables, etc.).
    """
    metrics = config.get("metrics", {})  # Optional metrics config.
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a mapping.")
    return metrics  # Caller decides how to use metrics fields.


def get_models_config(config: Dict[str, Any]) -> Dict[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping.")
    return models


def get_model_set(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    models = get_models_config(config)
    model_set = models.get(name)
    if model_set is None:
        raise KeyError(f"Unknown model set: {name}")
    if not isinstance(model_set, dict):
        raise ValueError("model set entries must be mappings.")
    return model_set
