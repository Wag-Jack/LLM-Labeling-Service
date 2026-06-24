"""Consolidated long-format CSV writers for paradigm outputs.

Each writer is keyed on a fixed set of columns and is idempotent: re-running
the same (prompt, model[, service]) combination replaces just those rows in
the consolidated file rather than appending duplicates. This lets a single
file accumulate results across prompts and models without exploding the
number of files on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from service_invocations.core.oracle_utils import normalize_id as _normalize_id


SERVICES_KEY = ("service", "id")
ORACLE_KEY = ("prompt", "model", "id")
JUDGE_KEY = ("prompt", "model", "id", "service")
HUMAN_LOOP_KEY = JUDGE_KEY
ACCURACY_KEY = ("prompt", "model", "service", "id")
ACCURACY_SUMMARY_KEY = ("prompt", "model", "service")
# LLMaaS scores the LLM's own oracle answer as a standalone pseudo-service, so
# "service" is constant ("llmaas") and is dropped from the upsert keys.
LLMAAS_ACCURACY_KEY = ("prompt", "model", "id")
LLMAAS_SUMMARY_KEY = ("prompt", "model")


def _row_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """Build a per-row join key that is stable across the CSV round-trip.

    The ``id`` column is normalized with ``normalize_id`` rather than
    compared raw: a written zero-padded id like ``"0001"`` is reloaded by
    pandas as the int ``1``, so comparing the string forms (``"0001"`` from
    an in-memory row vs ``"1"`` from the reloaded CSV) would never match.
    That silently defeated the upsert, appending the whole cumulative batch
    as duplicate rows on every checkpoint/resume write. Columns are joined
    with a NUL separator so adjacent key parts cannot collide.
    """
    parts = {
        k: (df[k].map(_normalize_id) if k == "id" else df[k].astype(str))
        for k in keys
    }
    return pd.DataFrame(parts).agg("\x00".join, axis=1)


def _merge(existing: pd.DataFrame | None, new: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    if new.empty:
        return existing if existing is not None else new
    if existing is None or existing.empty:
        return new.reset_index(drop=True)

    new_cols = [c for c in new.columns if c not in existing.columns]
    if new_cols:
        for col in new_cols:
            existing[col] = pd.NA
    missing_in_new = [c for c in existing.columns if c not in new.columns]
    for col in missing_in_new:
        new[col] = pd.NA
    new = new[existing.columns]

    merge_keys = [k for k in key if k in existing.columns and k in new.columns]
    if not merge_keys:
        merged = pd.concat([existing, new], ignore_index=True)
        return merged.reset_index(drop=True)

    new_keys = _row_keys(new, merge_keys)
    existing_keys = _row_keys(existing, merge_keys)
    mask = existing_keys.isin(set(new_keys))
    kept = existing[~mask]
    merged = pd.concat([kept, new], ignore_index=True)
    return merged.reset_index(drop=True)


def _upsert(path: Path, new: pd.DataFrame, key: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else None
    merged = _merge(existing, new, key)
    merged.to_csv(path, index=False)


def write_services(task_dir: Path, task: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    if "task" not in df.columns:
        df.insert(0, "task", task)
    _upsert(task_dir / "services.csv", df, SERVICES_KEY)


def write_oracle(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "oracle.csv", df, ORACLE_KEY)


def write_judge(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "judge.csv", df, JUDGE_KEY)


def write_human_loop(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "human_loop.csv", df, HUMAN_LOOP_KEY)


def write_accuracy(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "accuracy.csv", df, ACCURACY_KEY)


def write_accuracy_summary(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "accuracy_summary.csv", df, ACCURACY_SUMMARY_KEY)


def write_llmaas_accuracy(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    """Per-sample metrics for the LLM's own oracle answer (LLMaaS), scored as a
    standalone pseudo-service against the human reference. Kept in a dedicated
    file so it never lands in accuracy.csv (which drives best-service / winner
    consistency logic that must only see real services)."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "llmaas_accuracy.csv", df, LLMAAS_ACCURACY_KEY)


def write_llmaas_summary(task_dir: Path, task: str, prompt: str, model: str, rows: Iterable[dict]) -> None:
    """Per-(prompt, model) summary of the LLMaaS standalone accuracy."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.insert(0, "task", task)
    df.insert(1, "prompt", prompt)
    df.insert(2, "model", model)
    _upsert(task_dir / "llmaas_summary.csv", df, LLMAAS_SUMMARY_KEY)


def accuracy_slice_complete(
    task_dir: Path,
    prompt: str,
    model: str,
    services: Iterable[str],
    sample_ids: Iterable,
) -> bool:
    """True if accuracy.csv already holds every (service, id) for (prompt, model).

    Lets a resumed run skip re-deriving metrics (COMET / WER / classification
    accuracy) for a model whose oracle was already fully labeled. Those scores
    are a pure function of the (stable) service outputs, references, and oracle
    labels, so a complete slice never needs recomputation — which avoids, in
    particular, reloading the COMET checkpoint and re-running inference.
    """
    path = task_dir / "accuracy.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    needed = {"prompt", "model", "service", "id"}
    if df.empty or not needed.issubset(df.columns):
        return False
    sl = df[(df["prompt"].astype(str) == str(prompt)) & (df["model"].astype(str) == str(model))]
    if sl.empty:
        return False
    have = {(str(s), _normalize_id(i)) for s, i in zip(sl["service"].tolist(), sl["id"].tolist())}
    expected = {(str(s), _normalize_id(i)) for s in services for i in sample_ids}
    return bool(expected) and expected.issubset(have)


_PARADIGM_FILES = {
    "oracle": "oracle.csv",
    "judge": "judge.csv",
    "human_loop": "human_loop.csv",
}


def load_completed_ids(
    task_dir: Path, paradigm: str, prompt: str, model: str
) -> set[str]:
    """Return the set of sample ids already present in the consolidated CSV
    for the given (paradigm, prompt, model). Used by runners to skip work
    that was already done — across runs, after crashes, or after the
    failover queue gives up on a model.

    ``paradigm`` must be one of ``"oracle"``, ``"judge"``, ``"human_loop"``.
    """
    filename = _PARADIGM_FILES.get(paradigm)
    if filename is None:
        raise ValueError(
            f"load_completed_ids: unknown paradigm '{paradigm}'. "
            f"Expected one of {sorted(_PARADIGM_FILES)}."
        )
    path = task_dir / filename
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty:
        return set()
    needed = {"prompt", "model", "id"}
    if not needed.issubset(df.columns):
        return set()
    matched = df[(df["prompt"].astype(str) == str(prompt)) & (df["model"].astype(str) == str(model))]
    if matched.empty:
        return set()
    # Normalize ids the same way the runners do before checking membership.
    # pandas re-reads a zero-padded id column (e.g. "0001") as the int 1, so a
    # raw str() would yield "1" and never match the runner's "0001" key — the
    # resume guard would then silently re-invoke every already-completed sample.
    return {_normalize_id(v) for v in matched["id"].tolist()}


def clear_completed_slice(
    task_dir: Path, paradigm: str, prompt: str, models: Iterable[str]
) -> int:
    """Drop rows matching ``prompt`` and ``model in models`` from the
    consolidated CSV for the given paradigm. Returns the number of rows
    removed. No-op if the file is missing or nothing matches.

    Used by the runners' ``fresh_run`` path to wipe prior results before a
    clean re-run, so neither the resume guard nor downstream tabulation
    picks up stale rows.
    """
    filename = _PARADIGM_FILES.get(paradigm)
    if filename is None:
        raise ValueError(
            f"clear_completed_slice: unknown paradigm '{paradigm}'. "
            f"Expected one of {sorted(_PARADIGM_FILES)}."
        )
    path = task_dir / filename
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if df.empty or "prompt" not in df.columns or "model" not in df.columns:
        return 0
    model_set = {str(m) for m in models}
    if not model_set:
        return 0
    mask = (df["prompt"].astype(str) == str(prompt)) & df["model"].astype(str).isin(model_set)
    removed = int(mask.sum())
    if removed > 0:
        df[~mask].reset_index(drop=True).to_csv(path, index=False)
    return removed


def load_completed_rows(
    task_dir: Path, paradigm: str, prompt: str, model: str
) -> pd.DataFrame:
    """Return the slice of the consolidated CSV for the given (paradigm,
    prompt, model). Returns an empty DataFrame if the file or slice is
    missing. Used by oracle runners to repopulate results_by_model after a
    resume."""
    filename = _PARADIGM_FILES.get(paradigm)
    if filename is None:
        raise ValueError(
            f"load_completed_rows: unknown paradigm '{paradigm}'. "
            f"Expected one of {sorted(_PARADIGM_FILES)}."
        )
    path = task_dir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or not {"prompt", "model"}.issubset(df.columns):
        return pd.DataFrame()
    matched = df[(df["prompt"].astype(str) == str(prompt)) & (df["model"].astype(str) == str(model))]
    return matched.reset_index(drop=True)


__all__ = [
    "write_services",
    "write_oracle",
    "write_judge",
    "write_human_loop",
    "write_accuracy",
    "write_accuracy_summary",
    "write_llmaas_accuracy",
    "write_llmaas_summary",
    "load_completed_ids",
    "load_completed_rows",
    "clear_completed_slice",
]
