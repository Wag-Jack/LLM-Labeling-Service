"""Tee stdout/stderr and Python ``logging`` output to a per-trial log file
under ``invocation_report/``.

A *trial* is one user-initiated run — e.g., a single menu pick in
``main.py`` or one ``benchmark_prompts.run_all_prompts`` call. A trial can
internally invoke multiple tasks (speech / language / emotion) and many
LLM calls; every line that scrolls past in the terminal lands in the
same file, named by the trial's start time.

Layout::

    invocation_report/
      2026-06-01/
        20-00-00_benchmark.log    # one trial — covers all sub-tasks
        20-31-44_asr.log

Two scopes:

* :func:`trial_log` — explicit, called by entry points (``main.py``,
  ``benchmark_prompts``). Opens the file and stays open across nested
  task invocations.
* :func:`mirrored_run` / :func:`mirror_terminal` — used by task-level
  ``run_*`` functions. If a trial is already active they piggy-back on
  it (writing a sub-task banner into the same file); otherwise they
  start an ad-hoc trial for backward compatibility.

What gets captured:

* Any ``print``/``sys.stdout.write``/``sys.stderr.write`` after the
  trial opens (tee'd to the file).
* Python ``logging`` records routed through the root logger — this is
  how the Google GenAI / OpenAI / httpx SDKs emit "AFC is enabled" and
  "HTTP Request: POST ..." messages. We attach a ``StreamHandler``
  pointing at the trial file and ensure the root logger level is INFO
  or lower so those records propagate.

What is *not* captured: writes that bypass Python (C-level ``write(2)``
on fd 1/2). None of our SDKs do that today.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, TextIO, TypeVar


class _TeeStream:
    """Forward writes to both the original stream and a mirror file."""

    def __init__(self, original: TextIO, mirror: TextIO) -> None:
        self._original = original
        self._mirror = mirror

    def write(self, data):  # type: ignore[no-untyped-def]
        result = self._original.write(data)
        try:
            self._mirror.write(data)
        except Exception:
            pass
        return result

    def flush(self) -> None:
        try:
            self._original.flush()
        finally:
            try:
                self._mirror.flush()
            except Exception:
                pass

    def __getattr__(self, name):  # type: ignore[no-untyped-def]
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Trial state (module-level singleton). A trial is opened by trial_log and
# observed by mirror_terminal so nested calls share one file.
# ---------------------------------------------------------------------------

_active_trial: dict | None = None


def _resolve_root(root: Path | None) -> Path:
    return root if root is not None else (Path.cwd() / "invocation_report")


def _sanitize(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").strip("._") or "trial"


def _build_trial_path(label: str | None, root: Path | None, now: datetime) -> Path:
    base = _resolve_root(root)
    day_dir = base / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H-%M-%S")
    if label:
        return day_dir / f"{stamp}_{_sanitize(label)}.log"
    return day_dir / f"{stamp}.log"


@contextmanager
def trial_log(
    label: str | None = None,
    *,
    root: Path | None = None,
    path: Path | None = None,
    append: bool = False,
) -> Iterator[Path]:
    """Open a trial-level log capturing stdout, stderr, and ``logging``.

    Nested calls reuse the outer trial — only the outermost ``trial_log``
    actually opens a file. The filename uses the outer trial's start time
    so the whole run lives in a single, time-stamped report.

    ``path`` pins the log to an explicit file (used to keep the invocation
    report aligned with a timestamped results run). ``append=True`` reopens an
    existing log instead of truncating it — used when *continuing* a run so the
    resumed output is appended to the original report.
    """
    global _active_trial
    if _active_trial is not None:
        # Already inside a trial — caller is nested under benchmark or similar.
        yield _active_trial["path"]
        return

    now = datetime.now()
    if path is not None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = _build_trial_path(label, root, now)
    fh = open(log_path, "a" if append else "w", encoding="utf-8", buffering=1)
    if append:
        fh.write(f"\n# ---- resumed: {now.isoformat(timespec='seconds')} ----\n")

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, fh)
    sys.stderr = _TeeStream(orig_stderr, fh)

    # Route Python ``logging`` records (Google GenAI SDK, httpx, etc.) into
    # the same file. Writing through the file handle directly — not via the
    # tee — so SDK log lines reach the file even if their own handlers hold
    # a stale reference to the original stderr.
    log_handler = logging.StreamHandler(fh)
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    )
    log_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    prior_level = root_logger.level
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)

    _active_trial = {
        "path": log_path,
        "fh": fh,
        "started": now,
        "label": label,
        "log_handler": log_handler,
    }
    try:
        fh.write(
            "# invocation report\n"
            f"# trial:   {label or '(unlabeled)'}\n"
            f"# started: {now.isoformat(timespec='seconds')}\n"
            f"# log:     {log_path}\n"
            f"{'-' * 72}\n"
        )
        print(f"[mirror] Trial log: {log_path}")
        yield log_path
    finally:
        try:
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(prior_level)
            log_handler.close()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            try:
                ended = datetime.now().isoformat(timespec="seconds")
                fh.write(f"{'-' * 72}\n# finished: {ended}\n")
            finally:
                fh.close()
            _active_trial = None


@contextmanager
def mirror_terminal(
    task_name: str,
    *,
    root: Path | None = None,
) -> Iterator[Path]:
    """Task-level wrapper. If a trial is already active, write a sub-task
    banner into it; otherwise open an ad-hoc trial just for this call.
    """
    if _active_trial is not None:
        banner_ts = datetime.now().isoformat(timespec="seconds")
        print(f"\n=== {task_name} @ {banner_ts} ===")
        yield _active_trial["path"]
        return

    with trial_log(task_name, root=root) as path:
        yield path


_F = TypeVar("_F", bound=Callable[..., object])


def mirrored_run(task_name: str, *, root: Path | None = None) -> Callable[[_F], _F]:
    """Decorator: capture stdout/stderr/logging during the wrapped call.

    If a :func:`trial_log` is already active on the caller's stack, the
    output appends to that trial. Otherwise a single-task trial is opened
    for the duration of the call.
    """

    def decorator(fn: _F) -> _F:
        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            with mirror_terminal(task_name, root=root):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
