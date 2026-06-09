"""Sample-Based Discriminatory Sampling (SDS).

Implements the sample-discrimination metric from Meng et al.,
"Measuring Discrimination to Boost Comparative Testing for Multiple Deep
Learning Models" (ICSE'21). For each sample, the discrimination score
quantifies how much the labelers disagree:

  - Categorical outputs: 1 - max_k p_k, where p_k is the fraction of
    labelers emitting label k. Equivalent to the Gini-style impurity used
    in the paper for K-way classification.
  - Text outputs (transcripts, translations): mean pairwise (1 -
    SequenceMatcher ratio) across the labelers' normalized outputs.

Samples with higher discrimination are the most informative to label,
so the same labeling budget yields a tighter ranking of the models.
"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence

import pandas as pd

from service_invocations.core.oracle_utils import normalize_id as _normalize_id


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(value: Any) -> str:
    # None or a pandas NaN (empty CSV cell) is a missing output, not the
    # literal token "nan" — str(float("nan")) would otherwise leak "nan" past
    # the `if t`/`if l` empties filter in the scorers below and skew scores.
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


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
        name = payload.get("label_name") or payload.get("label")
        if name:
            return str(name).strip().lower()
    return ""


def _categorical_discrimination(labels: Sequence[str]) -> float:
    cleaned = [l for l in labels if l]
    if len(cleaned) < 2:
        return 0.0
    counts = Counter(cleaned)
    top = counts.most_common(1)[0][1]
    return 1.0 - (top / len(cleaned))


def _text_discrimination(texts: Sequence[str]) -> float:
    cleaned = [t for t in texts if t]
    n = len(cleaned)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            ratio = SequenceMatcher(a=cleaned[i], b=cleaned[j]).ratio()
            total += 1.0 - ratio
            pairs += 1
    return total / pairs if pairs else 0.0


_EXTRACTORS: Dict[str, Callable[[Any], str]] = {
    "text": _normalize_text,
    "emotion": _extract_top_emotion,
}
_SCORERS: Dict[str, Callable[[Sequence[str]], float]] = {
    "text": _text_discrimination,
    "emotion": _categorical_discrimination,
}


def compute_discrimination(
    results_by_service: Dict[str, pd.DataFrame],
    sample_ids: Iterable[Any],
    output_kind: str,
    output_column: str = "service_output",
) -> pd.DataFrame:
    """Return a DataFrame of (id, discrimination, per-service outputs).

    output_kind is "text" for ASR/MT and "emotion" for FER.
    """
    if output_kind not in _EXTRACTORS:
        raise ValueError(
            f"Unknown output_kind '{output_kind}'. Expected one of {list(_EXTRACTORS)}"
        )
    extractor = _EXTRACTORS[output_kind]
    scorer = _SCORERS[output_kind]

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
        score = scorer(list(per_service.values()))
        row = {"id": id_key, "discrimination": round(score, 6)}
        for name, value in per_service.items():
            row[f"output__{name}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("discrimination", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
    return df


def select_top_k(discrimination_df: pd.DataFrame, k: int) -> List[str]:
    """Return the top-k sample IDs by discrimination score (normalized form)."""
    if discrimination_df.empty or k <= 0:
        return []
    return discrimination_df.head(k)["id"].tolist()


def filter_dataset(df: pd.DataFrame, id_column: str, keep_ids: Sequence[str]) -> pd.DataFrame:
    if not keep_ids:
        return df.iloc[0:0].copy()
    keep_set = set(keep_ids)
    mask = df[id_column].map(_normalize_id).isin(keep_set)
    return df[mask].copy().reset_index(drop=True)


def filter_service_results(
    results_by_service: Dict[str, pd.DataFrame],
    keep_ids: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    if not keep_ids:
        return {name: df.iloc[0:0].copy() for name, df in results_by_service.items()}
    keep_set = set(keep_ids)
    filtered: Dict[str, pd.DataFrame] = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns:
            filtered[name] = df.copy()
            continue
        mask = df["id"].map(_normalize_id).isin(keep_set)
        filtered[name] = df[mask].copy().reset_index(drop=True)
    return filtered


def save_discrimination(df: pd.DataFrame, results_dir: Path, prefix: str = "sds") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{prefix}_ranking.csv"
    df.to_csv(path, index=False)
    return path


__all__ = [
    "compute_discrimination",
    "select_top_k",
    "filter_dataset",
    "filter_service_results",
    "save_discrimination",
]
