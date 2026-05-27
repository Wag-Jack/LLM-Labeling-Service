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


SERVICES_KEY = ("service", "id")
ORACLE_KEY = ("prompt", "model", "id")
JUDGE_KEY = ("prompt", "model", "id", "service")
HUMAN_LOOP_KEY = JUDGE_KEY
ACCURACY_KEY = ("prompt", "model", "service", "id")
ACCURACY_SUMMARY_KEY = ("prompt", "model", "service")


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

    mask = pd.Series(False, index=existing.index)
    new_keys = new[merge_keys].astype(str).agg("".join, axis=1)
    existing_keys = existing[merge_keys].astype(str).agg("".join, axis=1)
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


__all__ = [
    "write_services",
    "write_oracle",
    "write_judge",
    "write_human_loop",
    "write_accuracy",
    "write_accuracy_summary",
]
