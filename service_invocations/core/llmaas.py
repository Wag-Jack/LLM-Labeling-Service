"""Helpers for scoring the LLM's own oracle answer as a standalone service.

The "oracle" paradigm has each model answer the raw input with no service
outputs in context — i.e. the LLM acting as a service itself (LLMaaS). The
existing metric functions (``wer.py`` / ``comet.py`` / ``metrics.py``) already
score any entry of a ``results_by_service`` mapping against the human
reference, so we reuse them by feeding the oracle output as one extra
pseudo-service named :data:`LLMAAS_SERVICE`. Its ``human_*`` column is then the
model's true standalone accuracy.

Keeping the LLMaaS rows in dedicated files (rather than ``accuracy.csv``) is
deliberate: ``accuracy.csv`` drives best-service / winner-consistency logic that
must only ever see real services the judge/human-loop could pick.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

LLMAAS_SERVICE = "llmaas"


def oracle_as_service(
    oracle_results: pd.DataFrame,
    transform: Callable[[object], object] | None = None,
) -> pd.DataFrame:
    """Adapt an oracle results frame (``id``, ``llm_oracle``) into the
    ``id``/``service_output`` shape the metric functions expect.

    ``transform`` optionally maps each ``llm_oracle`` cell into the form the
    task's metric reads from a service output. ASR/MT use the raw oracle text
    as-is, but FER service outputs are parsed for a top-1 label, so its oracle
    score distribution must first be collapsed to that label.
    """
    out = oracle_results.rename(columns={"llm_oracle": "service_output"})[
        ["id", "service_output"]
    ].copy()
    if transform is not None:
        out["service_output"] = out["service_output"].map(transform)
    return out


def split_llmaas_rows(per_sample_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition long-format metric rows into (real-service rows, llmaas rows)."""
    service_rows = [r for r in per_sample_rows if r.get("service") != LLMAAS_SERVICE]
    llmaas_rows = [r for r in per_sample_rows if r.get("service") == LLMAAS_SERVICE]
    return service_rows, llmaas_rows


__all__ = ["LLMAAS_SERVICE", "oracle_as_service", "split_llmaas_rows"]
