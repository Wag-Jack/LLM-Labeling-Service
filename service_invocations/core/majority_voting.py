"""Majority Voting baseline aggregation.

Follows the simple majority-voting label aggregation reviewed in Daniel et
al., "Crowdsourcing of Software Tasks" (BISE 2015): treat each labeler as an
independent voter and emit the most-popular label per sample. The result is
a pseudo-oracle that costs nothing extra to produce (the services have
already run) and serves as a non-LLM baseline against the LLM oracle.

For categorical labels (FER top emotion) the implementation is a pure mode.
For free-text outputs (ASR transcripts, MT translations) there is no exact
majority, so the medoid is returned instead: the candidate whose sum of
SequenceMatcher similarities to the other candidates is highest. This
generalizes mode to text and falls back to a pure mode when several
candidates are byte-identical.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from service_invocations.core.oracle_utils import normalize_id as _normalize_id


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    # None or a pandas NaN (empty CSV cell) is a missing ballot, not the
    # literal token "nan" — str(float("nan")) would otherwise leak "nan" past
    # the `if t` empties filter in _medoid/_mode and count as a real vote.
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return _WHITESPACE_RE.sub(" ", str(value).strip())


def _extract_top_emotion(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return value.strip().lower()
    else:
        payload = value
    if isinstance(payload, dict):
        top = payload.get("top_emotion")
        if isinstance(top, dict):
            name = top.get("name")
            if name:
                return str(name).strip().lower()
    return ""


def _mode(labels: Sequence[str]) -> Tuple[str, int, int]:
    cleaned = [l for l in labels if l]
    if not cleaned:
        return "", 0, 0
    counts = Counter(cleaned)
    label, votes = counts.most_common(1)[0]
    return label, votes, len(cleaned)


def _medoid(texts: Sequence[str]) -> Tuple[str, float, int]:
    cleaned = [t for t in texts if t]
    if not cleaned:
        return "", 0.0, 0
    if len(cleaned) == 1:
        return cleaned[0], 1.0, 1
    counts = Counter(cleaned)
    top_label, top_count = counts.most_common(1)[0]
    if top_count > 1:
        return top_label, top_count / len(cleaned), len(cleaned)
    best_text = cleaned[0]
    best_score = -1.0
    for i, candidate in enumerate(cleaned):
        sim_sum = 0.0
        for j, other in enumerate(cleaned):
            if i == j:
                continue
            sim_sum += SequenceMatcher(a=candidate, b=other).ratio()
        if sim_sum > best_score:
            best_score = sim_sum
            best_text = candidate
    avg_sim = best_score / (len(cleaned) - 1) if len(cleaned) > 1 else 1.0
    return best_text, avg_sim, len(cleaned)


def majority_vote(
    results_by_service: Dict[str, pd.DataFrame],
    sample_ids: Iterable[Any],
    output_kind: str,
    output_column: str = "service_output",
) -> pd.DataFrame:
    """Return a DataFrame with one row per sample_id: id, mv_label, agreement, voters.

    output_kind:
      - "emotion": exact-mode over top_emotion.name
      - "text":   medoid string over normalized service outputs
    """
    if output_kind not in ("emotion", "text"):
        raise ValueError(f"Unknown output_kind '{output_kind}'.")
    extractor = _extract_top_emotion if output_kind == "emotion" else _normalize_text

    lookups: Dict[str, Dict[str, str]] = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or output_column not in df.columns:
            continue
        lookups[name] = dict(
            zip(df["id"].map(_normalize_id), df[output_column].map(extractor))
        )

    rows: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        id_key = _normalize_id(sample_id)
        per_service = {name: lookups[name].get(id_key, "") for name in lookups}
        ballots = list(per_service.values())
        if output_kind == "emotion":
            label, votes, voters = _mode(ballots)
            agreement = (votes / voters) if voters else 0.0
        else:
            label, agreement, voters = _medoid(ballots)
            votes = sum(1 for b in ballots if b == label)
        row = {
            "id": id_key,
            "mv_label": label,
            "votes": votes,
            "voters": voters,
            "agreement": round(agreement, 4),
        }
        for name, value in per_service.items():
            row[f"ballot__{name}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def save_majority_voting(df: pd.DataFrame, results_dir: Path, prefix: str = "majority_voting") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{prefix}.csv"
    df.to_csv(path, index=False)
    return path


__all__ = ["majority_vote", "save_majority_voting"]
