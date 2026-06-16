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
from typing import Any, Callable, Dict, List

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
        audio_price = entry.get("audio_input_per_million_usd")
        if input_price is None and output_price is None and audio_price is None:
            continue
        pricing[name] = {
            "input_per_million_usd": float(input_price or 0.0),
            "output_per_million_usd": float(output_price or 0.0),
            # Falls back to the text input rate when the model doesn't list a
            # separate audio rate, so old configs keep pricing audio tokens.
            "audio_input_per_million_usd": float(
                audio_price if audio_price is not None else (input_price or 0.0)
            ),
        }
    _PRICING_CACHE[resolved] = pricing
    return pricing


def compute_cost(
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    models_path: Path | None = None,
    audio_input_tokens: int | None = None,
) -> float | None:
    """Return the USD cost for a single LLM call, or None if pricing/tokens missing.

    ``input_tokens`` is the *total* prompt tokens as reported by the provider.
    ``audio_input_tokens``, when provided, is the audio subset of those prompt
    tokens; it is billed at ``audio_input_per_million_usd`` while the remaining
    (text) prompt tokens are billed at ``input_per_million_usd``.
    """
    if models_path is None:
        models_path = Path.cwd() / "config" / "models.yaml"
    pricing = _load_pricing(models_path).get(model_name)
    if pricing is None:
        return None
    if input_tokens is None and output_tokens is None and audio_input_tokens is None:
        return None
    in_tokens = int(input_tokens or 0)
    out_tokens = int(output_tokens or 0)
    audio_tokens = int(audio_input_tokens or 0)
    text_in_tokens = max(in_tokens - audio_tokens, 0)
    cost = (
        text_in_tokens * pricing["input_per_million_usd"] / 1_000_000.0
        + audio_tokens * pricing["audio_input_per_million_usd"] / 1_000_000.0
        + out_tokens * pricing["output_per_million_usd"] / 1_000_000.0
    )
    return round(cost, 6)


def make_attempt_recorder(
    tracker: "CostTracker",
    *,
    task: str,
    paradigm: str,
    model: str,
    sample_id: Any,
    usable: "Callable[[Any], bool]",
    models_path: Path | None = None,
):
    """Build an ``on_attempt`` callback for ``retry_until_valid`` that records
    the cost of *every* billed attempt and returns the per-sample total.

    ``usable(result)`` decides the recorded ``status`` ("success" if the output
    is a usable label for the paradigm, else "failed"). ``result`` is whatever
    the retried ``call()`` returns — these recorders return ``(response, parsed)``
    tuples, so the response (carrying token counts) is taken from ``result[0]``.

    Returns ``(on_attempt, total_cost)`` where ``total_cost()`` yields the summed
    USD across all attempts (or None if nothing was priced), suitable for the
    per-sample labelling row so it reconciles with cost.csv.
    """
    costs: List[float] = []

    def _response_of(result: Any) -> Any:
        return result[0] if isinstance(result, (tuple, list)) else result

    def on_attempt(result: Any, ok: bool) -> None:
        resp = _response_of(result)
        audio_in = getattr(resp, "audio_input_tokens", None)
        cost = compute_cost(
            model,
            getattr(resp, "input_tokens", None),
            getattr(resp, "output_tokens", None),
            models_path,
            audio_input_tokens=audio_in,
        )
        try:
            status = "success" if usable(result) else "failed"
        except Exception:
            status = "failed"
        tracker.record(
            task=task,
            paradigm=paradigm,
            model=model,
            sample_id=sample_id,
            input_tokens=getattr(resp, "input_tokens", None),
            output_tokens=getattr(resp, "output_tokens", None),
            cost_usd=cost,
            status=status,
            latency_ms=getattr(resp, "latency_ms", None),
            audio_input_tokens=audio_in,
        )
        if cost is not None:
            costs.append(cost)

    def total_cost() -> float | None:
        return round(sum(costs), 6) if costs else None

    return on_attempt, total_cost


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
    # Wall-clock latency of the billed LLM call (ms), taken from the response.
    # None when the adapter did not report it.
    latency_ms: float | None = None
    # "success"  -> the call produced a usable label for the paradigm
    # "failed"   -> the LLM responded (and was billed) but the output was
    #               unusable (null transcript / no winner / all scores -1).
    # Note: calls that never returned a response (e.g. Gemini 503 /
    # ModelUnavailableError) are not recorded at all, so they never appear here.
    status: str = "success"
    # Subset of ``input_tokens`` attributed to audio parts of the prompt, when
    # the provider reports it. None for providers that don't break this out.
    audio_input_tokens: int | None = None


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
        status: str = "success",
        latency_ms: float | None = None,
        audio_input_tokens: int | None = None,
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
                    latency_ms=latency_ms,
                    status=status,
                    audio_input_tokens=audio_input_tokens,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(
                columns=[
                    "timestamp", "task", "paradigm", "model", "sample_id",
                    "input_tokens", "output_tokens", "cost_usd", "latency_ms",
                    "status", "audio_input_tokens",
                ]
            )
        return pd.DataFrame([e.__dict__ for e in self.entries])

    def total_usd(self) -> float:
        """USD across every billed call (both successful and failed outputs)."""
        return float(sum((e.cost_usd or 0.0) for e in self.entries))

    def successful_usd(self) -> float:
        """USD spent only on calls that produced a usable label."""
        return float(
            sum((e.cost_usd or 0.0) for e in self.entries if e.status == "success")
        )

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        if "status" not in df.columns:
            df = df.assign(status="success")
        if "audio_input_tokens" not in df.columns:
            df = df.assign(audio_input_tokens=pd.NA)
        grouped = (
            df.groupby(["task", "paradigm", "model", "status"], dropna=False)
            .agg(
                calls=("cost_usd", "size"),
                input_tokens=("input_tokens", "sum"),
                audio_input_tokens=("audio_input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                cost_usd=("cost_usd", "sum"),
                avg_latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
        )
        if "avg_latency_ms" in grouped.columns:
            grouped["avg_latency_ms"] = grouped["avg_latency_ms"].round(2)
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
            # Older cost.csv files predate the status column; assume those rows
            # were the successful final attempts so totals still reconcile.
            if "status" not in existing.columns:
                existing["status"] = "success"
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
    "make_attempt_recorder",
    "CostTracker",
    "CostEntry",
    "session_tracker",
    "reset_session",
]
