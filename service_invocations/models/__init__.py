from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Callable
import os
import re

import yaml

from service_invocations.core.llm_adapters import LLMResponse, get_llm_adapter


_ENV_VAR_RE = re.compile(r"[^A-Za-z0-9]+")


def infer_modalities(inputs: Dict[str, Any] | None) -> List[str]:
    """Infer modalities from inputs; always include text for the prompt."""
    modalities = ["text"]
    if not inputs:
        return modalities
    if inputs.get("audio") is not None:
        modalities.append("audio")
    if inputs.get("image") is not None:
        modalities.append("image")
    return modalities


def _load_models_config(models_path: Path) -> Dict[str, Any]:
    if not models_path.exists():
        raise FileNotFoundError(f"Models config not found: {models_path}")
    with models_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("models.yaml root must be a mapping.")
    return config


def _get_models_section(config: Dict[str, Any]) -> Dict[str, Any]:
    models_cfg = config.get("models")
    if models_cfg is not None:
        if not isinstance(models_cfg, dict):
            raise ValueError("models.yaml 'models' must be a mapping.")
        return models_cfg

    if any(
        isinstance(value, dict) and ("enabled" in value or "provider" in value)
        for value in config.values()
    ):
        return config

    raise ValueError(
        "models.yaml must contain a top-level 'models' mapping "
        "of model names to configuration entries."
    )


def _get_model_entry(models_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    entry = models_cfg.get(model_name)
    if not isinstance(entry, dict):
        raise ValueError(f"models.yaml missing entry for model '{model_name}'.")
    return entry


def _default_env_key(model_name: str) -> str:
    normalized = _ENV_VAR_RE.sub("_", model_name).upper().strip("_")
    return f"{normalized}_MODEL_ID"


def _resolve_model_id(model_name: str, entry: Dict[str, Any]) -> str:
    env_key = entry.get("model_id_env") or _default_env_key(model_name)
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    model_id = entry.get("model_id")
    if not model_id:
        raise ValueError(
            f"models.yaml entry for '{model_name}' is missing model_id."
        )
    return model_id


def get_enabled_models(models_path: Path | None = None) -> List[str]:
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    config = _load_models_config(models_path)
    models_cfg = _get_models_section(config)
    enabled = []
    for name, entry in models_cfg.items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            enabled.append(name)
    return enabled


def get_model_generator(
    model_name: str,
    models_path: Path | None = None,
) -> Callable[[str, Dict[str, Any] | None, List[str] | None], LLMResponse]:
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    config = _load_models_config(models_path)
    models_cfg = _get_models_section(config)
    entry = _get_model_entry(models_cfg, model_name)
    provider = entry.get("provider")
    if not provider:
        raise ValueError(
            f"models.yaml entry for '{model_name}' is missing provider."
        )
    model_id = _resolve_model_id(model_name, entry)

    def generate(
        prompt: str,
        inputs: Dict[str, Any] | None = None,
        modalities: List[str] | None = None,
    ) -> LLMResponse:
        payload_inputs = inputs or {}
        requested_modalities = modalities or infer_modalities(payload_inputs)
        adapter = get_llm_adapter(provider)
        return adapter.generate(model_id, prompt, payload_inputs, requested_modalities)

    return generate


__all__ = ["get_enabled_models", "get_model_generator", "infer_modalities"]
