"""Snapshot & restore the in-code run settings (module-level tunables).

A *run* is pinned to the settings it began with. The file-based settings
(``config/services.yaml`` / ``models.yaml`` / ``prompts.yaml``) are snapshotted
into the run folder by :mod:`run_context`; this module covers the other half —
the tunable constants the three ``invoke_*`` modules expose (prompt choice for
single-task runs, the SDS top-k cutoff, the majority-voting toggle, ...).

On a NEW run :func:`collect_settings` captures the current values; on a RESUMED
run :func:`apply_settings` writes the snapshotted values back onto the modules so
the run continues exactly as it started — even if the constants were edited in
source in the meantime. A new run started after such an edit picks up the edit,
so different runs can carry different settings concurrently.
"""
from __future__ import annotations

import importlib
from typing import Any

SETTINGS_FILE = "run_settings.json"

# The tunable surface is identical across the three task entry points.
_INVOKE_SETTINGS = (
    "ORACLE_PROMPT",
    "JUDGE_PROMPT",
    "HUMAN_LOOP_PROMPT",
    "HUMAN_LOOP_NO_THRESHOLD_PROMPT",
    "QUIET_SKIP_PROMPTS",
    "SDS_TOP_K",
    "RUN_MAJORITY_VOTING",
)

# module dotted-path -> the attribute names that count as run settings.
_SETTING_MODULES: dict[str, tuple[str, ...]] = {
    "service_invocations.invoke_speech_recognition": _INVOKE_SETTINGS,
    "service_invocations.invoke_language_translation": _INVOKE_SETTINGS,
    "service_invocations.invoke_emotion_detection": _INVOKE_SETTINGS,
}


def collect_settings() -> dict[str, dict[str, Any]]:
    """Read the current value of every registered setting, grouped by module.

    Modules that fail to import (or attributes that are absent) are skipped, so
    capturing settings can never break a run.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for module_name, attrs in _SETTING_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - a missing module just isn't captured
            continue
        snapshot[module_name] = {a: getattr(module, a) for a in attrs if hasattr(module, a)}
    return snapshot


def apply_settings(snapshot: dict[str, dict[str, Any]] | None) -> list[str]:
    """Write a previously-collected snapshot back onto the modules.

    Only attributes that are part of the known setting surface are applied (so a
    stale or hand-edited snapshot can't set arbitrary module globals). Returns
    the dotted ``module.attr`` names that were applied, for logging.
    """
    applied: list[str] = []
    for module_name, attrs in (snapshot or {}).items():
        allowed = _SETTING_MODULES.get(module_name)
        if not allowed or not isinstance(attrs, dict):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        for attr, value in attrs.items():
            if attr in allowed:
                setattr(module, attr, value)
                applied.append(f"{module_name.rsplit('.', 1)[-1]}.{attr}")
    return applied


__all__ = ["SETTINGS_FILE", "collect_settings", "apply_settings"]
