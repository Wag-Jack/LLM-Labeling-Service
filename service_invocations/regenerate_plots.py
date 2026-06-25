"""Regenerate plots for an existing results run — without re-running any LLM calls.

This is the offline companion to the invoke_* scripts. It re-reads the CSVs a
run already produced and re-renders every plot. It can also **backfill** the
LLM-as-a-service (LLMaaS) accuracy that older runs predate: the metric that
scores each model's own oracle transcript against the human reference, so the
LLM shows up as a peer "service" in the accuracy plots.

The backfill needs only two artifacts the run already stored:
  * ``samples_<task>.csv`` — the exact sampled dataset, incl. the human ``text``
    reference (written next to the task dirs by the run).
  * ``oracle.csv``         — each model's LLM transcript per (prompt, model, id).
No audio, no service outputs, and no network/dataset reload are required.

Usage:
    python -m service_invocations.regenerate_plots <run_or_task_dir> [options]

<run_or_task_dir> may be either a single task dir (…/speech_recognition) or a
benchmark run dir that contains task subdirs.

Options:
    --task TASK       Only process this task (speech_recognition |
                      language_translation | emotion_detection).
    --plots-only      Skip the LLMaaS backfill; just re-render plots.
    --no-plots        Only backfill; do not re-render plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from service_invocations.core.llmaas import (
    LLMAAS_SERVICE,
    oracle_as_service,
    split_llmaas_rows,
)
from service_invocations.core.plotting import plot_all_for_task
from service_invocations.core.results_io import (
    write_llmaas_accuracy,
    write_llmaas_summary,
)

_KNOWN_TASKS = (
    "speech_recognition",
    "language_translation",
    "emotion_detection",
)


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if not df.empty else None


def _find_samples_csv(task_dir: Path, task: str) -> Path | None:
    """Locate samples_<task>.csv near the task dir (it is written one level up)."""
    for candidate in (
        task_dir / f"samples_{task}.csv",
        task_dir.parent / f"samples_{task}.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _backfill_llmaas(
    task_dir: Path,
    samples_csv: Path,
    *,
    task: str,
    ref_cols: tuple[str, ...],
    compute_rows,
    compute_summary,
    transform=None,
) -> bool:
    """Backfill llmaas_accuracy.csv / llmaas_summary.csv for one task's run.

    Scores each model's stored oracle answer against the human reference using
    the same metric machinery the live pipeline uses (reused verbatim), so the
    numbers match a fresh run exactly. ``ref_cols`` are the human-reference
    columns the metric reads from ``samples_<task>.csv``; ``transform`` collapses
    an oracle cell into the form the metric reads from a service output (None for
    ASR/MT text; the top-1 label for FER). Returns True if anything was written.
    """
    oracle = _read_csv_or_none(task_dir / "oracle.csv")
    if oracle is None or "llm_oracle" not in oracle.columns:
        print(f"[regenerate] {task_dir.name}: no usable oracle.csv — skipping backfill.")
        return False
    label_df = _read_csv_or_none(samples_csv)
    needed = {"id", *ref_cols}
    if label_df is None or not needed.issubset(label_df.columns):
        print(f"[regenerate] {task_dir.name}: {samples_csv.name} lacks "
              f"{sorted(needed)} — skipping backfill.")
        return False

    wrote = False
    group_cols = [c for c in ("prompt", "model") if c in oracle.columns]
    for keys, grp in oracle.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        prompt = str(meta.get("prompt", "default"))
        model = str(meta.get("model", "default"))

        oracle_results = grp[["id", "llm_oracle"]].copy()
        # Feed the oracle answer as the only pseudo-service; compute_rows then
        # scores it against the human reference (human_*) and against itself
        # (oracle_* is the trivial self-match). split_llmaas_rows isolates them.
        augmented = {LLMAAS_SERVICE: oracle_as_service(oracle_results, transform=transform)}
        per_sample = compute_rows(augmented, oracle_results, label_df)
        _, llmaas_rows = split_llmaas_rows(per_sample)
        if not llmaas_rows:
            continue
        write_llmaas_accuracy(task_dir, task, prompt, model, llmaas_rows)
        summary = compute_summary(llmaas_rows, [LLMAAS_SERVICE])
        write_llmaas_summary(task_dir, task, prompt, model, summary)
        wrote = True

    if wrote:
        print(f"[regenerate] {task_dir.name}: wrote llmaas_summary.csv / llmaas_accuracy.csv.")
    return wrote


def _backfill_llmaas_speech(task_dir: Path, samples_csv: Path) -> bool:
    from service_invocations.speech_recognition.wer import (
        compute_wer_rows,
        compute_wer_summary_rows,
    )
    return _backfill_llmaas(
        task_dir, samples_csv, task="speech_recognition", ref_cols=("text",),
        compute_rows=compute_wer_rows, compute_summary=compute_wer_summary_rows,
    )


def _backfill_llmaas_language(task_dir: Path, samples_csv: Path) -> bool:
    from service_invocations.language_translation.comet import (
        compute_comet_rows,
        compute_comet_summary_rows,
    )
    return _backfill_llmaas(
        task_dir, samples_csv, task="language_translation", ref_cols=("english", "french"),
        compute_rows=compute_comet_rows, compute_summary=compute_comet_summary_rows,
    )


def _backfill_llmaas_emotion(task_dir: Path, samples_csv: Path) -> bool:
    from service_invocations.emotion_detection.metrics import (
        compute_emotion_rows,
        compute_emotion_summary_rows,
        oracle_top_emotion,
    )
    return _backfill_llmaas(
        task_dir, samples_csv, task="emotion_detection", ref_cols=("label",),
        compute_rows=compute_emotion_rows, compute_summary=compute_emotion_summary_rows,
        transform=oracle_top_emotion,
    )


# Per-task backfill dispatch. All three tasks reconstruct LLMaaS from oracle.csv
# + samples_<task>.csv using their own metric builders (WER / COMET / FER).
_BACKFILL = {
    "speech_recognition": _backfill_llmaas_speech,
    "language_translation": _backfill_llmaas_language,
    "emotion_detection": _backfill_llmaas_emotion,
}


def regenerate_task(task_dir: Path, task: str, *, backfill: bool, plots: bool) -> None:
    if backfill:
        fn = _BACKFILL.get(task)
        if fn is None:
            print(f"[regenerate] {task}: LLMaaS backfill not implemented — re-plotting only.")
        else:
            samples = _find_samples_csv(task_dir, task)
            if samples is None:
                print(f"[regenerate] {task}: samples_{task}.csv not found — cannot backfill.")
            else:
                fn(task_dir, samples)
    if plots:
        plot_all_for_task(task_dir, task)
        print(f"[regenerate] {task}: plots written to {task_dir / 'plots'}")


def _discover_task_dirs(root: Path, only: str | None) -> list[tuple[Path, str]]:
    """Return (task_dir, task) pairs for a task dir or a run dir of task subdirs."""
    if root.name in _KNOWN_TASKS:
        return [(root, root.name)]
    found: list[tuple[Path, str]] = []
    for task in _KNOWN_TASKS:
        if only and task != only:
            continue
        sub = root / task
        if sub.is_dir():
            found.append((sub, task))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="A task dir (…/speech_recognition) or a benchmark run dir.")
    parser.add_argument("--task", choices=_KNOWN_TASKS, default=None,
                        help="Restrict to one task (when given a run dir).")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip the LLMaaS backfill; only re-render plots.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only backfill the LLMaaS metric; do not re-render plots.")
    args = parser.parse_args()

    root = args.run_dir.resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    targets = _discover_task_dirs(root, args.task)
    if not targets:
        parser.error(f"No task results found under {root}")

    for task_dir, task in targets:
        print(f"--- Regenerating {task} ({task_dir}) ---")
        regenerate_task(
            task_dir, task,
            backfill=not args.plots_only,
            plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
