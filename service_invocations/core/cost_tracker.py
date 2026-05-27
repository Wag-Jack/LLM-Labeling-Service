"""Cost tracking for LLM calls.

Pricing is read from config/models.yaml (input_per_million_usd /
output_per_million_usd). When token counts are missing (some providers do not
return them), the call contributes 0 to the running total and `cost_usd` is
recorded as NaN so the row can still be joined back to the labelling result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import pandas as pd
import yaml


_PRICING_CACHE: Dict[Path, Dict[str, Dict[str, float]]] = {}
_LOCK = Lock()


def _load_pricing(models_path: Path) -> Dict[str, Dict[str, float]]:
    resolved = models_path.resolve()
    cached = _PRICING_CACHE.get(resolved)
    if cached is not None:
        return cached
    if not resolved.exists():
        _PRICING_CACHE[resolved] = {}
        return {}
    with resolved.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    section = config.get("models") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        section = config if isinstance(config, dict) else {}
    pricing: Dict[str, Dict[str, float]] = {}
    for name, entry in section.items():
        if not isinstance(entry, dict):
            continue
        input_price = entry.get("input_per_million_usd")
        output_price = entry.get("output_per_million_usd")
        if input_price is None and output_price is None:
            continue
        pricing[name] = {
            "input_per_million_usd": float(input_price or 0.0),
            "output_per_million_usd": float(output_price or 0.0),
        }
    _PRICING_CACHE[resolved] = pricing
    return pricing


def compute_cost(
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    models_path: Path | None = None,
) -> float | None:
    """Return the USD cost for a single LLM call, or None if pricing/tokens missing."""
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    pricing = _load_pricing(models_path).get(model_name)
    if pricing is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    in_tokens = int(input_tokens or 0)
    out_tokens = int(output_tokens or 0)
    cost = (
        in_tokens * pricing["input_per_million_usd"] / 1_000_000.0
        + out_tokens * pricing["output_per_million_usd"] / 1_000_000.0
    )
    return round(cost, 6)


@dataclass
class CostEntry:
    timestamp: str
    task: str
    paradigm: str
    model: str
    sample_id: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


@dataclass
class CostTracker:
    entries: List[CostEntry] = field(default_factory=list)

    def record(
        self,
        task: str,
        paradigm: str,
        model: str,
        sample_id: Any,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        with _LOCK:
            self.entries.append(
                CostEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    task=task,
                    paradigm=paradigm,
                    model=model,
                    sample_id=str(sample_id),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(
                columns=[
                    "timestamp", "task", "paradigm", "model", "sample_id",
                    "input_tokens", "output_tokens", "cost_usd",
                ]
            )
        return pd.DataFrame([e.__dict__ for e in self.entries])

    def total_usd(self) -> float:
        return float(sum((e.cost_usd or 0.0) for e in self.entries))

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        grouped = (
            df.groupby(["task", "paradigm", "model"], dropna=False)
            .agg(
                calls=("cost_usd", "size"),
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                cost_usd=("cost_usd", "sum"),
            )
            .reset_index()
        )
        return grouped

    def write(self, results_root: Path | None = None, task_filter: str | None = None) -> Path | None:
        """Append entries to a single rolling cost.csv under ``results_root``.

        When ``task_filter`` is set, only entries matching that task name are
        written (so each per-task invoke writes its own slice). The summary
        is recomputed and overwritten on every call.
        """
        if not self.entries:
            return None
        if results_root is None:
            results_root = Path.cwd() / "service_invocations" / "results"
        results_root.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if task_filter is not None:
            df = df[df["task"] == task_filter]
            if df.empty:
                return None

        log_path = results_root / "cost.csv"
        if log_path.exists():
            existing = pd.read_csv(log_path)
            if task_filter is not None:
                existing = existing[existing["task"] != task_filter]
            else:
                existing = existing.iloc[0:0]
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(log_path, index=False)
        return log_path


_SESSION = CostTracker()


def session_tracker() -> CostTracker:
    return _SESSION


def reset_session() -> None:
    global _SESSION
    _SESSION = CostTracker()


__all__ = [
    "compute_cost",
    "CostTracker",
    "CostEntry",
    "session_tracker",
    "reset_session",
]
