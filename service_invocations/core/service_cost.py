"""Cost tracking for third-party (non-LLM) labelling services.

This is the service-side companion to ``cost_tracker.py`` (which prices LLM
calls by token count). Cloud services bill by a usage *quantity* that differs
per task family:

  * Speech-to-text  -> audio duration (priced per minute)
  * Translation     -> input characters (priced per million characters)
  * Emotion / FER   -> one request per image (priced per image)

Pricing lives in ``config/services.yaml`` under each service's ``pricing`` block
(unit + a rate key, e.g. ``usd_per_minute`` / ``usd_per_million_chars`` /
``usd_per_image``). When a service has no pricing, or the usage quantity is
unknown, the call contributes 0 to the running total and ``cost_usd`` is
recorded as NaN so the row still round-trips with the labelling result — the
same convention ``cost_tracker`` uses for unpriced LLM calls.

A process-lifetime session tracker (:func:`session_service_tracker`) accumulates
every priced service call so the run can report a grand total that combines
LLM spend and service spend (see :func:`format_cost_summary`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import pandas as pd
import yaml


_PRICING_CACHE: Dict[Path, Dict[str, Dict[str, Dict[str, Any]]]] = {}
_LOCK = Lock()


# Maps each supported rate key to (usage_kind, factor). The cost of a call is
#   usage * rate * factor
# where ``usage`` is the caller-supplied quantity in the usage_kind's base unit
# (minutes / characters / count). The factor reconciles the rate's quoted unit
# with that base unit, so the YAML can mirror however a provider's docs quote
# the price (per minute, per hour, per million chars, per 1000 images, ...).
_RATE_KEYS: Dict[str, tuple[str, float]] = {
    "usd_per_minute": ("minutes", 1.0),
    "usd_per_hour": ("minutes", 1.0 / 60.0),
    "usd_per_second": ("minutes", 60.0),
    "usd_per_million_chars": ("characters", 1.0 / 1_000_000.0),
    "usd_per_thousand_chars": ("characters", 1.0 / 1_000.0),
    "usd_per_character": ("characters", 1.0),
    "usd_per_image": ("count", 1.0),
    "usd_per_thousand_images": ("count", 1.0 / 1_000.0),
    "usd_per_request": ("count", 1.0),
}


def _load_pricing(services_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{task: {service: pricing_block}}`` from services.yaml (cached)."""
    resolved = services_path.resolve()
    cached = _PRICING_CACHE.get(resolved)
    if cached is not None:
        return cached
    if not resolved.exists():
        _PRICING_CACHE[resolved] = {}
        return {}
    with resolved.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    pricing: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if isinstance(config, dict):
        for task, services in config.items():
            if not isinstance(services, dict):
                continue
            for service, entry in services.items():
                if not isinstance(entry, dict):
                    continue
                block = entry.get("pricing")
                if isinstance(block, dict):
                    pricing.setdefault(task, {})[service] = block
    _PRICING_CACHE[resolved] = pricing
    return pricing


def _default_services_path() -> Path:
    return Path.cwd() / "config" / "services.yaml"


def audio_minutes(row: Any) -> float | None:
    """Billable audio minutes for an STT sample row, from its ``duration`` (s).

    Returns None when the duration is absent or non-numeric (e.g. NaN reloaded
    from a CSV) so the call prices to NaN rather than guessing a duration.
    """
    duration = None
    if hasattr(row, "get"):
        duration = row.get("duration")
    if duration is None or pd.isna(duration):
        return None
    try:
        return float(duration) / 60.0
    except (TypeError, ValueError):
        return None


def compute_service_cost(
    task: str,
    service: str,
    *,
    minutes: float | None = None,
    characters: int | None = None,
    count: int | None = None,
    services_path: Path | None = None,
) -> tuple[float | None, str | None, float | None]:
    """Price a single service call.

    Pass exactly one usage quantity — ``minutes`` (STT), ``characters`` (MT) or
    ``count`` (FER, one per image). Returns ``(cost_usd, unit, usage)`` where
    ``unit`` names the billed quantity and ``usage`` is the amount billed. Cost
    is ``None`` when the service is unpriced or the matching usage is missing.
    """
    if services_path is None:
        services_path = _default_services_path()
    block = _load_pricing(services_path).get(task, {}).get(service)
    if not block:
        return None, None, None

    usage_by_kind = {"minutes": minutes, "characters": characters, "count": count}
    for key, (usage_kind, factor) in _RATE_KEYS.items():
        rate = block.get(key)
        if rate is None:
            continue
        usage = usage_by_kind.get(usage_kind)
        if usage is None:
            return None, usage_kind, None
        try:
            cost = float(usage) * float(rate) * factor
        except (TypeError, ValueError):
            return None, usage_kind, None
        return round(cost, 6), usage_kind, float(usage)
    return None, None, None


@dataclass
class ServiceCostEntry:
    timestamp: str
    task: str
    service: str
    sample_id: str
    unit: str | None
    usage_units: float | None
    cost_usd: float | None


@dataclass
class ServiceCostTracker:
    entries: List[ServiceCostEntry] = field(default_factory=list)

    def record(
        self,
        task: str,
        service: str,
        sample_id: Any,
        unit: str | None,
        usage_units: float | None,
        cost_usd: float | None,
    ) -> None:
        with _LOCK:
            self.entries.append(
                ServiceCostEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    task=task,
                    service=service,
                    sample_id=str(sample_id),
                    unit=unit,
                    usage_units=usage_units,
                    cost_usd=cost_usd,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "timestamp", "task", "service", "sample_id",
            "unit", "usage_units", "cost_usd",
        ]
        if not self.entries:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([e.__dict__ for e in self.entries])[columns]

    def total_usd(self) -> float:
        return float(sum((e.cost_usd or 0.0) for e in self.entries))

    def total_by_task(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for e in self.entries:
            totals[e.task] = totals.get(e.task, 0.0) + (e.cost_usd or 0.0)
        return totals

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.groupby(["task", "service"], dropna=False)
            .agg(
                calls=("cost_usd", "size"),
                usage_units=("usage_units", "sum"),
                cost_usd=("cost_usd", "sum"),
            )
            .reset_index()
        )

    def write(self, results_root: Path | None = None, task_filter: str | None = None) -> Path | None:
        """Append entries to a rolling ``service_cost.csv`` under ``results_root``.

        Mirrors ``CostTracker.write``: when ``task_filter`` is set only that
        task's rows are (re)written, so each per-task invoke writes its own slice.
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

        log_path = results_root / "service_cost.csv"
        if log_path.exists():
            existing = pd.read_csv(log_path)
            if task_filter is not None:
                existing = existing[existing["task"] != task_filter]
            else:
                existing = existing.iloc[0:0]
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(log_path, index=False)
        return log_path


_SESSION_SERVICE = ServiceCostTracker()


def session_service_tracker() -> ServiceCostTracker:
    return _SESSION_SERVICE


def reset_session_service() -> None:
    global _SESSION_SERVICE
    _SESSION_SERVICE = ServiceCostTracker()


def record_service_call(
    task: str,
    service: str,
    sample_id: Any,
    *,
    minutes: float | None = None,
    characters: int | None = None,
    count: int | None = None,
    services_path: Path | None = None,
) -> float | None:
    """Price a service call and record it on the session tracker; return the cost.

    The returned value is appended to the service's ``cost_usd`` result column,
    while the recorded entry feeds the run-wide grand total. ``cost_usd`` is
    ``None`` (-> NaN in the CSV) for unpriced services or missing usage.
    """
    cost, unit, usage = compute_service_cost(
        task, service,
        minutes=minutes, characters=characters, count=count,
        services_path=services_path,
    )
    session_service_tracker().record(task, service, sample_id, unit, usage, cost)
    return cost


def format_cost_summary(scope: str = "this run") -> str:
    """A combined LLM + service grand-total block for the end of a run.

    Reads both process-lifetime session trackers, so it reflects everything
    invoked so far: every LLM paradigm/model call and every priced service call.
    """
    from service_invocations.core.cost_tracker import session_tracker

    llm_total = session_tracker().total_usd()
    svc_tracker = session_service_tracker()
    svc_total = svc_tracker.total_usd()
    grand = llm_total + svc_total

    lines = [
        "=" * 56,
        f" TOTAL INVOCATION COST ({scope})",
        "=" * 56,
        f"  LLM calls (all paradigms/models):   ${llm_total:>12.4f}",
        f"  Service calls (all providers):      ${svc_total:>12.4f}",
    ]
    for task, amount in sorted(svc_tracker.total_by_task().items()):
        lines.append(f"      {task:<28} ${amount:>12.4f}")
    lines.append("-" * 56)
    lines.append(f"  GRAND TOTAL:                        ${grand:>12.4f}")
    lines.append("=" * 56)
    return "\n".join(lines)


__all__ = [
    "compute_service_cost",
    "record_service_call",
    "audio_minutes",
    "ServiceCostTracker",
    "ServiceCostEntry",
    "session_service_tracker",
    "reset_session_service",
    "format_cost_summary",
]
