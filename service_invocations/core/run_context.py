"""Timestamped run directories for results, with resume support.

A *run* is one user-initiated labeling pass. Each run writes into its own
timestamped folder that mirrors the ``invocation_report`` layout::

    service_invocations/results/
      2026-06-02/
        18-42-10_emotion_detection/      # single-task run (flat)
          run_status.json  samples.csv
          services/...  oracle.csv  judge.csv ...
        19-05-33_benchmark/              # multi-task run (one subdir per task)
          run_status.json
          samples_speech_recognition.csv  samples_language_translation.csv ...
          speech_recognition/   language_translation/   emotion_detection/

All writers resolve their output location through :func:`task_results_dir`, so
the same code path writes either into the active timestamped run (when one is
started) or the legacy ``results/<task>/`` location (when none is active).

A single-task run is *flat*: ``task_results_dir`` returns the run folder
directly. A multi-task run (the prompt benchmark) sets ``subdir_by_task`` so
each task gets its own subfolder under the run.

Resume: a run whose ``run_status.json`` is not ``finished`` can be continued.
Continuing reuses the same folder (so the per-sample skip logic in the oracle /
judge / human-loop writers picks up where it left off) and the same
``invocation_report`` log file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_RESULTS_ROOT = Path.cwd() / "service_invocations" / "results"
_INVOCATION_ROOT = Path.cwd() / "invocation_report"

_STATUS_FILE = "run_status.json"
_SAMPLES_FILE = "samples.csv"

_active_run: "RunInfo | None" = None


@dataclass
class RunInfo:
    label: str          # directory/log suffix: a task name or "benchmark"
    dir: Path
    date: str
    time: str
    is_continue: bool
    subdir_by_task: bool = False

    @property
    def display(self) -> str:
        return f"{self.date}/{self.time}_{self.label}"

    @property
    def log_path(self) -> Path:
        """Matching invocation_report log path (shared timestamp with results)."""
        return _INVOCATION_ROOT / self.date / f"{self.time}_{self.label}.log"


def results_root() -> Path:
    return _DEFAULT_RESULTS_ROOT


def active_run() -> "RunInfo | None":
    return _active_run


def active_run_dir() -> Path | None:
    return _active_run.dir if _active_run is not None else None


def is_continue() -> bool:
    return _active_run is not None and _active_run.is_continue


def task_results_dir(task: str) -> Path:
    """Per-task results directory.

    With an active run this is the run folder (flat) or ``<run>/<task>`` for a
    multi-task run; with no active run it is the legacy ``results/<task>``.
    """
    if _active_run is None:
        return _DEFAULT_RESULTS_ROOT / task
    if _active_run.subdir_by_task:
        return _active_run.dir / task
    return _active_run.dir


def task_services_dir(task: str) -> Path:
    return task_results_dir(task) / "services"


# --------------------------------------------------------------------------
# Run lifecycle
# --------------------------------------------------------------------------


def _parse_run_dir(run_dir: Path) -> tuple[str, str]:
    """Recover (date, time) from a run directory named ``<date>/<time>_<label>``."""
    date = run_dir.parent.name
    time = run_dir.name.partition("_")[0]
    return date, time


def start_run(
    label: str,
    *,
    continue_dir: Path | None = None,
    subdir_by_task: bool = False,
) -> RunInfo:
    """Begin (or resume) a run and make it the active results target.

    ``label`` is the run's directory/log suffix — a task name for single-task
    runs, or ``"benchmark"``. ``subdir_by_task=True`` namespaces each task in
    its own subfolder (used by the multi-task benchmark).
    """
    global _active_run
    if continue_dir is not None:
        continue_dir = Path(continue_dir)
        date, time = _parse_run_dir(continue_dir)
        info = RunInfo(
            label=label, dir=continue_dir, date=date, time=time,
            is_continue=True, subdir_by_task=subdir_by_task,
        )
    else:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H-%M-%S")
        run_dir = _DEFAULT_RESULTS_ROOT / date / f"{time}_{label}"
        info = RunInfo(
            label=label, dir=run_dir, date=date, time=time,
            is_continue=False, subdir_by_task=subdir_by_task,
        )

    info.dir.mkdir(parents=True, exist_ok=True)
    _active_run = info
    _write_status(info, status="in_progress")
    return info


def mark_finished() -> None:
    if _active_run is not None:
        _write_status(_active_run, status="finished")


def end_run() -> None:
    global _active_run
    _active_run = None


def _write_status(info: RunInfo, status: str) -> None:
    path = info.dir / _STATUS_FILE
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload["task"] = info.label
    payload["date"] = info.date
    payload["time"] = info.time
    payload["subdir_by_task"] = info.subdir_by_task
    payload.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    payload["status"] = status
    if status == "finished":
        payload["finished"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Sample persistence (so a continued run replays the exact same inputs)
# --------------------------------------------------------------------------


def _samples_filename(name: str | None) -> str:
    return f"samples_{name}.csv" if name else _SAMPLES_FILE


def save_samples(df: pd.DataFrame, name: str | None = None, run_dir: Path | None = None) -> Path:
    target = (run_dir or active_run_dir())
    if target is None:
        raise RuntimeError("save_samples called with no active run.")
    target.mkdir(parents=True, exist_ok=True)
    path = target / _samples_filename(name)
    df.to_csv(path, index=False)
    return path


def load_samples(run_dir: Path, name: str | None = None) -> pd.DataFrame:
    path = Path(run_dir) / _samples_filename(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot continue run: no {path.name} found in {run_dir}."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Discovery of resumable runs
# --------------------------------------------------------------------------


@dataclass
class ResumableRun:
    info: RunInfo
    started: str
    progress: str


def _dir_progress(run_dir: Path) -> str:
    """How far a single (flat) results dir got, by artifact presence."""
    parts: list[str] = []
    services_dir = run_dir / "services"
    if services_dir.is_dir():
        n = len(list(services_dir.glob("*.csv")))
        if n:
            parts.append(f"services={n}")
    for name in ("oracle.csv", "judge.csv", "human_loop.csv"):
        path = run_dir / name
        if path.exists():
            try:
                rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            except OSError:
                rows = 0
            parts.append(f"{name.removesuffix('.csv')}={max(rows, 0)} rows")
    if (run_dir / "plots").is_dir():
        parts.append("plots")
    return ", ".join(parts) if parts else "no output yet"


def _run_progress(run_dir: Path, subdir_by_task: bool) -> str:
    if not subdir_by_task:
        return _dir_progress(run_dir)
    parts: list[str] = []
    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        sub_progress = _dir_progress(sub)
        if sub_progress != "no output yet":
            parts.append(f"{sub.name}: {sub_progress}")
    return "; ".join(parts) if parts else "no output yet"


def find_continuable_runs(label: str) -> list[ResumableRun]:
    """All not-yet-finished runs with this ``label`` (task name or "benchmark"), newest first."""
    runs: list[ResumableRun] = []
    if not _DEFAULT_RESULTS_ROOT.exists():
        return runs
    for status_path in _DEFAULT_RESULTS_ROOT.glob("*/*/" + _STATUS_FILE):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("task") != label:
            continue
        if payload.get("status") == "finished":
            continue
        run_dir = status_path.parent
        date, time = _parse_run_dir(run_dir)
        subdir_by_task = bool(payload.get("subdir_by_task", False))
        info = RunInfo(
            label=label, dir=run_dir, date=date, time=time,
            is_continue=True, subdir_by_task=subdir_by_task,
        )
        runs.append(
            ResumableRun(
                info=info,
                started=payload.get("started", "?"),
                progress=_run_progress(run_dir, subdir_by_task),
            )
        )
    runs.sort(key=lambda r: (r.info.date, r.info.time), reverse=True)
    return runs


__all__ = [
    "RunInfo",
    "ResumableRun",
    "results_root",
    "active_run",
    "active_run_dir",
    "is_continue",
    "task_results_dir",
    "task_services_dir",
    "start_run",
    "mark_finished",
    "end_run",
    "save_samples",
    "load_samples",
    "find_continuable_runs",
]
