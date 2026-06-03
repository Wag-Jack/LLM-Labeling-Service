"""Run samples across multiple LLMs with per-model failover and a deferred queue.

If a model raises ``ModelUnavailableError`` (the adapter has exhausted its
retries on transient errors), this runner stops processing that model's
remaining samples, queues them, and moves on to the next model. Once every
model has had a first pass, the runner drains the queue — sleeping for a
cooldown period between passes — and gives up after a configurable number
of attempts.

Callers supply:
- ``models``: the list of model names to run.
- ``samples``: the sequence of samples to evaluate (opaque to the runner).
- ``make_processor(model_name)``: builds the per-model setup (generator,
  prompt, etc.) and returns a ``process(sample) -> dict | None`` callable.
- ``on_progress(model, rows, is_final)``: optional callback fired each time
  a model's batch finishes a pass — either run-to-completion, deferred, or
  given up after the final drain pass. Use this to persist partial CSVs.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from service_invocations.core.llm_adapters import ModelUnavailableError

_DEFAULT_COOLDOWN = float(os.getenv("LLM_FAILOVER_COOLDOWN", "30.0"))
_DEFAULT_DRAIN_PASSES = int(os.getenv("LLM_FAILOVER_DRAIN_PASSES", "3"))


ProcessorFactory = Callable[[str], Callable[[Any], Optional[Dict[str, Any]]]]
ProgressCallback = Callable[[str, List[Dict[str, Any]], bool], None]


def run_with_failover(
    *,
    models: List[str],
    samples: List[Any],
    make_processor: ProcessorFactory,
    on_progress: Optional[ProgressCallback] = None,
    cooldown_seconds: float = _DEFAULT_COOLDOWN,
    max_drain_passes: int = _DEFAULT_DRAIN_PASSES,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    processors: Dict[str, Callable[[Any], Optional[Dict[str, Any]]]] = {}

    def _get_processor(model: str) -> Callable[[Any], Optional[Dict[str, Any]]]:
        proc = processors.get(model)
        if proc is None:
            proc = make_processor(model)
            processors[model] = proc
        return proc

    def _run_batch(model: str, batch: List[Any]) -> List[Any]:
        """Process ``batch`` for ``model``. Returns samples that were deferred.

        A *fatal* ``ModelUnavailableError`` (auth/permission) gives up on the
        model immediately — the remaining samples are dropped rather than
        deferred, so we don't burn drain-pass cooldowns retrying credentials
        that won't change mid-run.
        """
        processor = _get_processor(model)
        for idx, sample in enumerate(batch):
            try:
                row = processor(sample)
            except ModelUnavailableError as exc:
                remaining = batch[idx:]
                if getattr(exc, "fatal", False):
                    print(
                        f"[failover] Model '{model}' permanently unavailable: {exc}. "
                        f"Skipping {len(remaining)} remaining sample(s) "
                        f"(no retry).",
                        file=sys.stderr,
                        flush=True,
                    )
                    return []
                print(
                    f"[failover] Model '{model}' unavailable: {exc}. "
                    f"Deferring {len(remaining)} sample(s) to the queue.",
                    file=sys.stderr,
                    flush=True,
                )
                return remaining
            if row is not None:
                results[model].append(row)
        return []

    def _emit_progress(model: str, is_final: bool) -> None:
        if on_progress is None:
            return
        try:
            on_progress(model, list(results[model]), is_final)
        except Exception as exc:
            print(
                f"[failover] on_progress callback for '{model}' raised "
                f"{type(exc).__name__}: {exc}. Continuing.",
                file=sys.stderr,
                flush=True,
            )

    pending: Dict[str, List[Any]] = {}

    for model in models:
        remaining = _run_batch(model, list(samples))
        if remaining:
            pending[model] = remaining
            _emit_progress(model, is_final=False)
        else:
            _emit_progress(model, is_final=True)

    for pass_idx in range(max_drain_passes):
        if not pending:
            break
        is_last_pass = pass_idx == max_drain_passes - 1
        if cooldown_seconds > 0:
            print(
                f"[failover] Drain pass {pass_idx + 1}/{max_drain_passes}: "
                f"{len(pending)} model(s) pending "
                f"({', '.join(sorted(pending))}). "
                f"Cooling down {cooldown_seconds:.0f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(cooldown_seconds)
        next_pending: Dict[str, List[Any]] = {}
        for model, batch in pending.items():
            remaining = _run_batch(model, batch)
            if remaining and not is_last_pass:
                next_pending[model] = remaining
                _emit_progress(model, is_final=False)
            else:
                if remaining:
                    print(
                        f"[failover] Giving up on '{model}' after "
                        f"{max_drain_passes} drain pass(es); "
                        f"{len(remaining)} sample(s) unprocessed.",
                        file=sys.stderr,
                        flush=True,
                    )
                _emit_progress(model, is_final=True)
        pending = next_pending

    return results


__all__ = ["run_with_failover", "ModelUnavailableError"]
