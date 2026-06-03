"""Per-service input-failure accounting.

Counts, for each service, how many of its inputs failed to yield a usable
label. A failure is any of: an empty output, a JSON parse error, an explicit
``error`` field, an error / async-acceptance envelope (e.g. AppTek's
``{"request_id": ...}``), or a missing top prediction (FER).

Writes a per-domain ``service_failures.csv`` and merges each domain's rows into
a combined cross-domain ``service_failures.csv`` so services can be compared on
reliability alongside accuracy and cost.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Keys that mark an async-acceptance / tracking envelope rather than a real
# text output (the translation/transcript itself never arrived in the row).
_ENVELOPE_KEYS = ("request_id", "requestId")


def _is_failure_emotion(output: Any) -> bool:
    """FER failure: no parseable JSON, an error field, or no top emotion."""
    if output is None or (isinstance(output, float) and pd.isna(output)):
        return True
    text = str(output).strip()
    if not text:
        return True
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(payload, dict):
        return True
    if payload.get("error"):
        return True
    top = payload.get("top_emotion")
    name = top.get("name") if isinstance(top, dict) else None
    return name is None or str(name).strip() == ""


def _is_failure_text(output: Any) -> bool:
    """ASR/MT failure: empty output or an error / acceptance envelope."""
    if output is None or (isinstance(output, float) and pd.isna(output)):
        return True
    text = str(output).strip()
    if not text:
        return True
    # A real transcript/translation is plain text and won't parse as JSON.
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(payload, dict):
        if payload.get("error"):
            return True
        if payload.get("success") is False:
            return True
        if any(key in payload for key in _ENVELOPE_KEYS):
            return True
    return False


_DETECTORS = {
    "emotion": _is_failure_emotion,
    "text": _is_failure_text,
}


def compute_failure_report(
    results_by_service: Dict[str, pd.DataFrame],
    task: str,
    output_kind: str,
    output_column: str = "service_output",
) -> pd.DataFrame:
    """One row per service: total_inputs, failed_inputs, failure_rate."""
    if output_kind not in _DETECTORS:
        raise ValueError(
            f"Unknown output_kind '{output_kind}'. Expected one of {list(_DETECTORS)}"
        )
    detector = _DETECTORS[output_kind]

    rows = []
    for service_name, df in results_by_service.items():
        columns = getattr(df, "columns", [])
        if df is None or output_column not in columns:
            total = 0 if df is None else len(df)
            failed = total
        else:
            outputs = df[output_column].tolist()
            total = len(outputs)
            failed = sum(1 for value in outputs if detector(value))
        rate = (failed / total) if total else 0.0
        rows.append({
            "task": task,
            "service": service_name,
            "total_inputs": total,
            "failed_inputs": failed,
            "failure_rate": round(rate, 4),
        })

    report = pd.DataFrame(
        rows, columns=["task", "service", "total_inputs", "failed_inputs", "failure_rate"]
    )
    if not report.empty:
        report = report.sort_values(
            ["failure_rate", "service"], ascending=[False, True]
        ).reset_index(drop=True)
    return report


def save_failure_report(
    report: pd.DataFrame, results_dir: Path, prefix: str = "service_failures"
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{prefix}.csv"
    report.to_csv(path, index=False)
    return path


def update_combined_failure_report(
    report: pd.DataFrame,
    task: str,
    combined_dir: Path,
    prefix: str = "service_failures",
) -> Path:
    """Merge this domain's rows into the cross-domain report (replacing its own)."""
    combined_dir.mkdir(parents=True, exist_ok=True)
    path = combined_dir / f"{prefix}.csv"

    existing = None
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = None

    if existing is not None and "task" in existing.columns:
        existing = existing[existing["task"] != task]
        combined = pd.concat([existing, report], ignore_index=True)
    else:
        combined = report.copy()

    if not combined.empty:
        combined = combined.sort_values(
            ["task", "failure_rate", "service"], ascending=[True, False, True]
        ).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return path


def print_failure_summary(report: pd.DataFrame, task: str) -> None:
    if report.empty:
        print(f"--- Service failure report ({task}): no service results ---")
        return
    print(f"--- Service failure report ({task}) ---")
    for _, r in report.iterrows():
        print(
            f"    {r['service']}: {int(r['failed_inputs'])}/{int(r['total_inputs'])} "
            f"inputs failed ({r['failure_rate']:.1%})"
        )


__all__ = [
    "compute_failure_report",
    "save_failure_report",
    "update_combined_failure_report",
    "print_failure_summary",
]
