from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from service_invocations.core.config import load_config


def _resolve_runner(runner_path: str):
    if ":" not in runner_path:
        raise ValueError("runner must be in 'module:function' format.")
    module_path, function_name = runner_path.split(":", 1)
    module = import_module(module_path)
    runner = getattr(module, function_name, None)
    if runner is None or not callable(runner):
        raise TypeError(f"runner '{runner_path}' is not callable.")
    return runner


def _load_service_registry(config_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    if config_path is None:
        config_path = Path.cwd() / "config" / "services.yaml"
    config = load_config(config_path)
    registry = config.get("service_registry", {})
    if not isinstance(registry, dict):
        raise ValueError("service_registry must be a mapping.")

    resolved: Dict[str, Dict[str, Any]] = {}
    for name, entry in registry.items():
        if not isinstance(entry, dict):
            raise ValueError("service_registry entries must be mappings.")
        runner_path = entry.get("runner")
        results_file = entry.get("results_file")
        if not runner_path:
            raise ValueError(f"service_registry entry '{name}' missing runner.")
        if not results_file:
            raise ValueError(f"service_registry entry '{name}' missing results_file.")
        resolved[name] = {
            **entry,
            "runner": _resolve_runner(runner_path),
        }
    return resolved


def load_service_registry(config_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    return _load_service_registry(config_path)


def _normalize_service_set(service_set: Iterable[dict],
                           service_registry: Dict[str, Dict[str, Any]]) -> List[dict]:
    normalized = []
    for entry in service_set:
        if not isinstance(entry, dict):
            raise ValueError("Each service entry must be a mapping.")
        if "name" not in entry:
            raise ValueError("Each service entry must include a name.")
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        if name not in service_registry:
            raise KeyError(f"Unknown speech service: {name}")
        merged = {**service_registry[name], **entry}
        if merged.get("task") != "stt":
            continue
        normalized.append(merged)
    return normalized


def run_speech_services(edacc_df, service_set: Iterable[dict], use_existing: bool = False,
                        results_dir: Path | None = None,
                        config_path: Path | None = None,
                        service_registry: Dict[str, Dict[str, Any]] | None = None,
                        ) -> Dict[str, pd.DataFrame]:
    if results_dir is None:
        results_dir = Path.cwd() / "service_invocations" / "results" / "speech_recognition"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Service results are stored under a task-scoped folder.
    if service_registry is None:
        service_registry = _load_service_registry(config_path)

    services = _normalize_service_set(service_set, service_registry)

    results: Dict[str, pd.DataFrame] = {}
    for entry in services:
        name = entry["name"]
        runner = entry["runner"]
        results_file = results_dir / entry["results_file"]
        print(f'--- {name} ---')

        if use_existing and results_file.exists():
            results[name] = pd.read_csv(results_file)
            continue
        results[name] = runner(edacc_df)

    return results
