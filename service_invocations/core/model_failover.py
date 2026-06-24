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

  It is also fired mid-batch every ``checkpoint_every`` newly processed
  samples, so a hard crash (OOM, SIGKILL, unexpected exception) loses at most
  that many samples' work instead of the entire in-progress model pass. The
  callback must therefore be idempotent — it always receives every row
  accumulated so far, and the runners persist via keyed upserts.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from service_invocations.core.llm_adapters import ModelUnavailableError

_DEFAULT_COOLDOWN = float(os.getenv("LLM_FAILOVER_COOLDOWN", "30.0"))
_DEFAULT_DRAIN_PASSES = int(os.getenv("LLM_FAILOVER_DRAIN_PASSES", "3"))

# TEMPORARY (remove when gemini_3_5_flash is replaced): gemini_3_5_flash has a
# much tighter rate-limit quota than the other models and routinely exhausts its
# retries mid-run, getting its remaining samples dropped. Give just that model a
# longer cooldown between drain passes so its quota window has time to recover;
# every other model keeps the default. The drain sleep is shared across all
# pending models in a pass, so a pass waits the longest cooldown among the models
# still pending — in practice gemini_3_5_flash is the lone late straggler, so it
# is the only one that triggers the longer wait. Override via env, e.g.
# LLM_FAILOVER_COOLDOWN_GEMINI_3_5_FLASH=240.
_MODEL_COOLDOWN_OVERRIDES: Dict[str, float] = {
    "gemini_3_5_flash": float(
        os.getenv("LLM_FAILOVER_COOLDOWN_GEMINI_3_5_FLASH", "180.0")
    ),
}
# How often to checkpoint partial progress within a model's pass. ``0`` disables
# mid-batch checkpoints (persist only at pass/defer boundaries, the old
# behavior); ``1`` flushes after every sample.
_DEFAULT_CHECKPOINT_EVERY = int(os.getenv("LLM_FAILOVER_CHECKPOINT_EVERY", "10"))


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
    checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    progress_task: Optional[str] = None,
    progress_paradigm: Optional[str] = None,
    progress_prompt: Optional[str] = None,
    progress_total: Optional[int] = None,
    progress_task_dir: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    processors: Dict[str, Callable[[Any], Optional[Dict[str, Any]]]] = {}

    def _get_processor(model: str) -> Callable[[Any], Optional[Dict[str, Any]]]:
        proc = processors.get(model)
        if proc is None:
            proc = make_processor(model)
            processors[model] = proc
        return proc

    def _emit_progress(model: str, is_final: bool) -> None:
        if on_progress is not None:
            try:
                on_progress(model, list(results[model]), is_final)
            except Exception as exc:
                print(
                    f"[failover] on_progress callback for '{model}' raised "
                    f"{type(exc).__name__}: {exc}. Continuing.",
                    file=sys.stderr,
                    flush=True,
                )
        _record_progress(model)

    def _record_progress(model: str) -> None:
        """Mirror this slice's progress into run_status.json (best-effort).

        Counts the unique sample ids already persisted to the consolidated CSV
        — the source of truth — so the tally is correct across resumes (it
        includes work done in earlier sessions, not just this process's rows).
        on_progress has already flushed the current rows, so the CSV is current.
        """
        if progress_task is None or progress_task_dir is None:
            return
        try:
            from service_invocations.core import run_context as _rc
            from service_invocations.core.results_io import load_completed_ids

            done = len(load_completed_ids(
                progress_task_dir, progress_paradigm, progress_prompt, model
            ))
            total = progress_total if progress_total is not None else done
            _rc.record_progress(
                progress_task, progress_paradigm, progress_prompt, model, done, total,
            )
        except Exception:
            pass

    def _run_batch(model: str, batch: List[Any]) -> List[Any]:
        """Process ``batch`` for ``model``. Returns samples that were deferred.

        A *fatal* ``ModelUnavailableError`` (auth/permission) gives up on the
        model immediately — the remaining samples are dropped rather than
        deferred, so we don't burn drain-pass cooldowns retrying credentials
        that won't change mid-run.

        Partial progress is checkpointed every ``checkpoint_every`` newly
        processed samples (when > 0) by firing ``on_progress`` mid-batch, so a
        hard crash loses at most that many samples' work rather than the whole
        pass. The boundary emits done by the caller still run; the extra writes
        are idempotent upserts, so they never duplicate rows.
        """
        processor = _get_processor(model)
        since_checkpoint = 0
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
                since_checkpoint += 1
                if checkpoint_every > 0 and since_checkpoint >= checkpoint_every:
                    _emit_progress(model, is_final=False)
                    since_checkpoint = 0
        return []

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
        # Per-model cooldown: a pass waits the longest cooldown among the models
        # still pending, so a tighter-quota model (see _MODEL_COOLDOWN_OVERRIDES)
        # gets its longer recovery window without penalizing passes it isn't in.
        effective_cooldown = max(
            _MODEL_COOLDOWN_OVERRIDES.get(m, cooldown_seconds) for m in pending
        )
        if effective_cooldown > 0:
            print(
                f"[failover] Drain pass {pass_idx + 1}/{max_drain_passes}: "
                f"{len(pending)} model(s) pending "
                f"({', '.join(sorted(pending))}). "
                f"Cooling down {effective_cooldown:.0f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(effective_cooldown)
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
