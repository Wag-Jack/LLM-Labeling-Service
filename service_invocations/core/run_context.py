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

Settings are pinned per run. At creation a run snapshots its settings into its
own folder — the ``config/*.yaml`` files into ``<run>/config`` and the in-code
``invoke_*`` tunables into ``<run>/run_settings.json`` (see
:mod:`run_settings`). All config reads resolve through :func:`config_path`, so a
run always reads its own snapshot: resuming it reuses the exact settings it began
with even if the live ``config/`` (or the in-code defaults) changed in the
meantime, while a brand-new run snapshots the current settings. Two runs can
therefore carry different settings at the same time.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_RESULTS_ROOT = Path.cwd() / "service_invocations" / "results"
_INVOCATION_ROOT = Path.cwd() / "invocation_report"

_STATUS_FILE = "run_status.json"
_SAMPLES_FILE = "samples.csv"

# Each run is pinned to the settings it began with. The live config/ directory
# is snapshotted into the run folder at creation; all config reads resolve
# through config_path() so an active run reads its own snapshot (kept fixed
# across resumes) while everything else reads the live config.
_LIVE_CONFIG_DIR = Path.cwd() / "config"
_CONFIG_SNAPSHOT_DIRNAME = "config"
_CONFIG_FILES = ("services.yaml", "models.yaml", "prompts.yaml")

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


def config_dir() -> Path:
    """Directory the active run reads its settings (``*.yaml``) from.

    An active run reads its own per-run snapshot (``<run>/config``), so its
    settings stay fixed for the whole run — including every resume — regardless
    of later edits to the live ``config/``. With no active run (or a run with no
    snapshot, e.g. one created before snapshots existed) the live ``config/`` is
    used.
    """
    if _active_run is not None:
        snapshot = _active_run.dir / _CONFIG_SNAPSHOT_DIRNAME
        if snapshot.is_dir():
            return snapshot
    return _LIVE_CONFIG_DIR


def config_path(name: str) -> Path:
    """Resolve a config file (``services.yaml`` / ``models.yaml`` / ``prompts.yaml``)
    against the active run's snapshot, falling back to the live ``config/``."""
    return config_dir() / name


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
    _snapshot_config(info)
    _snapshot_or_restore_settings(info)
    return info


def _snapshot_config(info: RunInfo) -> None:
    """Pin the run's file-based settings: copy live ``config/*.yaml`` into the
    run folder, once.

    Files already present in the snapshot are left untouched, so resuming a run
    preserves the settings it began with even if the live config has since
    changed. (A run created before snapshots existed has none, so on its first
    resume the current live config is captured — the only settings available.)
    """
    dest = info.dir / _CONFIG_SNAPSHOT_DIRNAME
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for name in _CONFIG_FILES:
            target = dest / name
            if target.exists():
                continue
            src = _LIVE_CONFIG_DIR / name
            if src.exists():
                shutil.copy2(src, target)
    except OSError:
        # Snapshotting is best-effort: config_path() falls back to live config.
        pass


def _snapshot_or_restore_settings(info: RunInfo) -> None:
    """Pin/restore the in-code run settings (the ``invoke_*`` module tunables).

    New run -> capture the current values into ``run_settings.json``. Resumed
    run -> apply the snapshotted values back onto the modules so the run
    continues with the settings it started with. Best-effort; never raises into
    the caller.
    """
    from service_invocations.core import run_settings

    path = info.dir / run_settings.SETTINGS_FILE
    try:
        if path.exists():
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if info.is_continue:
                applied = run_settings.apply_settings(snapshot)
                if applied:
                    print(
                        f"[settings] restored {len(applied)} in-code setting(s) "
                        f"from this run's snapshot."
                    )
        else:
            path.write_text(
                json.dumps(run_settings.collect_settings(), indent=2),
                encoding="utf-8",
            )
    except Exception:  # noqa: BLE001 - settings bookkeeping must never crash a run
        pass


def attach_run(run_dir: Path) -> RunInfo:
    """Point the active run at ``run_dir`` without touching its status file.

    Use this for read-only operations like replotting — unlike ``start_run``
    with ``continue_dir``, this never marks the run ``in_progress``, so a
    finished run stays finished.
    """
    global _active_run
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    payload: dict[str, Any] = {}
    status_path = run_dir / _STATUS_FILE
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    date, time = _parse_run_dir(run_dir)
    label = payload.get("task") or run_dir.name.partition("_")[2] or "benchmark"
    info = RunInfo(
        label=label, dir=run_dir, date=date, time=time,
        is_continue=False, subdir_by_task=bool(payload.get("subdir_by_task", False)),
    )
    _active_run = info
    return info


def _progress_is_complete(payload: dict[str, Any]) -> bool:
    """Did the run actually process everything it planned?

    The failover runner *gives up* on samples it can't complete (e.g. a model
    rate-limited past its drain passes) rather than crashing, so the run loop
    returns normally even when slices are short. We decide completeness from the
    progress totals that ``record_progress`` wrote:

    * If a planned ``total_samples`` is known, the run is complete only when
      every planned sample was processed. This also catches slices that never
      started (a model that failed from its first sample), since those are
      missing from ``samples_done`` but still counted in the plan.
    * Otherwise fall back to the per-slice tally: complete iff at least one
      slice started and none were left short.

    With no progress recorded at all we can't prove incompleteness, so we treat
    the run as complete (preserving the prior unconditional behaviour).
    """
    totals = (payload.get("progress") or {}).get("totals") or {}
    planned = totals.get("samples_total_planned") or (payload.get("plan") or {}).get("total_samples")
    if planned:
        return int(totals.get("samples_done", 0)) >= int(planned)
    started = int(totals.get("slices_started", 0))
    if started:
        return int(totals.get("slices_complete", 0)) >= started
    return True


def mark_finished() -> None:
    """Seal the active run.

    Marks ``finished`` only when the run actually processed everything it
    planned; otherwise marks ``incomplete`` so the resume picker keeps offering
    it (``find_continuable_runs`` skips only ``finished`` runs). On resume the
    paradigm modules skip already-complete samples, so a finished-as-incomplete
    run just re-attempts the slices that were dropped.
    """
    if _active_run is None:
        return
    path = _active_run.dir / _STATUS_FILE
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    status = "finished" if _progress_is_complete(payload) else "incomplete"
    _write_status(_active_run, status=status)


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
# Live progress (continuously written into run_status.json, so a monitor or a
# post-crash inspection can see exactly how far the run got)
# --------------------------------------------------------------------------


def _update_status_payload(mutate) -> None:
    """Read run_status.json, apply ``mutate(payload)`` in place, and write back.

    Preserves whatever ``status`` the file already has (never flips a run to
    ``finished``), and is best-effort: a read/write hiccup is swallowed so
    progress bookkeeping can never crash the run it's reporting on.
    """
    if _active_run is None:
        return
    path = _active_run.dir / _STATUS_FILE
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    try:
        mutate(payload)
    except Exception:
        return
    payload.setdefault("task", _active_run.label)
    payload.setdefault("date", _active_run.date)
    payload.setdefault("time", _active_run.time)
    payload.setdefault("subdir_by_task", _active_run.subdir_by_task)
    payload.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("status", "in_progress")
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def set_plan(**fields: Any) -> None:
    """Record the overall planned scope of the run (e.g. ``total_samples``).

    Called once near the start of a benchmark so :func:`record_progress` can
    report a global percentage. Merged into any existing ``plan`` block.
    """
    if _active_run is None:
        return
    def _mutate(payload: dict[str, Any]) -> None:
        payload["plan"] = {**(payload.get("plan") or {}), **fields}
    _update_status_payload(_mutate)


def record_progress(
    task: str,
    paradigm: str,
    prompt: str,
    model: str,
    samples_done: int,
    samples_total: int,
) -> None:
    """Record progress for one (task, paradigm, prompt, model) slice.

    Updates ``run_status.json`` in place: the per-slice tally, a ``current``
    pointer to the slice last touched, and rolled-up ``totals`` (with a
    ``percent`` when a planned ``total_samples`` is known). Fired continuously
    (every failover checkpoint) so the file always reflects live progress and
    survives a crash. Best-effort — never raises into the caller.
    """
    if _active_run is None:
        return
    key = f"{task}/{paradigm}/{prompt}/{model}"
    now = datetime.now().isoformat(timespec="seconds")

    def _mutate(payload: dict[str, Any]) -> None:
        prog = payload.get("progress") or {}
        slices = prog.get("slices") or {}
        slices[key] = {"done": int(samples_done), "total": int(samples_total)}
        done_sum = sum(int(s.get("done", 0)) for s in slices.values())
        total_seen = sum(int(s.get("total", 0)) for s in slices.values())
        complete = sum(
            1 for s in slices.values()
            if s.get("total") and s.get("done", 0) >= s["total"]
        )
        totals: dict[str, Any] = {
            "samples_done": done_sum,
            "samples_total_seen": total_seen,
            "slices_started": len(slices),
            "slices_complete": complete,
        }
        plan_total = (payload.get("plan") or {}).get("total_samples")
        if plan_total:
            totals["samples_total_planned"] = plan_total
            totals["percent"] = round(100.0 * done_sum / plan_total, 1)
        prog["slices"] = slices
        prog["current"] = {
            "task": task, "paradigm": paradigm, "prompt": prompt, "model": model,
            "samples_done": int(samples_done), "samples_total": int(samples_total),
        }
        prog["totals"] = totals
        prog["updated"] = now
        payload["progress"] = prog

    _update_status_payload(_mutate)


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
    "config_dir",
    "config_path",
    "start_run",
    "attach_run",
    "mark_finished",
    "end_run",
    "set_plan",
    "record_progress",
    "save_samples",
    "load_samples",
    "find_continuable_runs",
]
