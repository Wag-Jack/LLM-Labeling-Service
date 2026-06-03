"""Classification metrics for facial emotion recognition.

Mirrors the role that wer.py plays for ASR and comet.py plays for MT:
score each service's predictions against two references (the LLM oracle
and the human ground-truth label from VEA), producing per-sample counts
and a per-service summary.

Per-class precision/recall/F1 are computed for every canonical emotion;
the summary also reports accuracy along with macro, micro, and weighted
F1 (and the corresponding precision/recall averages) so services with
imbalanced confusion matrices are still comparable.
"""
from __future__ import annotations

import ast
import json
from typing import Dict, List, Sequence

import pandas as pd

from service_invocations.core.oracle_utils import normalize_id as _normalize_id
from service_invocations.emotion_detection.services._shared import (
    CANONICAL_EMOTIONS,
    label_to_name,
)


def _extract_top_emotion(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text.lower()
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


def _normalize_label(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text


# The FER oracle prompt reports scores under "happiness"/"sadness", while
# service predictions use the canonical "happy"/"sad". Align them so the
# oracle's argmax class matches the prediction vocabulary.
_ORACLE_LABEL_ALIASES = {
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "angry": "anger",
    "disgusted": "disgust",
}


def _coerce_scores(value):
    """Return a dict of emotion scores from a dict, JSON string, or Python repr."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _oracle_top_emotion(value) -> str:
    """Argmax canonical emotion from the oracle's score distribution.

    The FER oracle returns a probability distribution over emotion classes, so
    the label comparable to a service's top-1 prediction is the highest-scoring
    class. If the value is not a score mapping (e.g. a plain label), it is
    normalized as a label string instead.
    """
    scores = _coerce_scores(value)
    if scores is None:
        return _normalize_label(value)
    best_name = None
    best_score = None
    for name, score in scores.items():
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if best_score is None or score_f > best_score:
            best_score = score_f
            best_name = name
    if best_name is None:
        return ""
    key = str(best_name).strip().lower()
    return _ORACLE_LABEL_ALIASES.get(key, key)


def _build_predictions_by_service(results_by_service):
    predictions = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        ids = df["id"].map(_normalize_id)
        predictions[name] = dict(zip(ids, df["service_output"].map(_extract_top_emotion)))
    return predictions


def _safe_div(numer: float, denom: float):
    if denom <= 0:
        return None
    return numer / denom


def _per_class_counts(refs: Sequence[str], preds: Sequence[str], labels: Sequence[str]):
    """Return per-class {tp, fp, fn, support} using only rows with a usable reference."""
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}
    for ref, pred in zip(refs, preds):
        if not ref:
            continue
        if ref in counts:
            counts[ref]["support"] += 1
            if pred == ref:
                counts[ref]["tp"] += 1
            else:
                counts[ref]["fn"] += 1
        if pred and pred in counts and pred != ref:
            counts[pred]["fp"] += 1
    return counts


def _aggregate(counts: Dict[str, Dict[str, int]]):
    """Return (per_class_metrics, macro, micro, weighted, accuracy, n)."""
    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0
    macro_p = macro_r = macro_f = 0.0
    macro_n = 0
    weighted_p = weighted_r = weighted_f = 0.0

    for label, c in counts.items():
        tp, fp, fn, support = c["tp"], c["fp"], c["fn"], c["support"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_support += support
        if support > 0:
            macro_n += 1
            macro_p += precision or 0.0
            macro_r += recall or 0.0
            macro_f += f1 or 0.0
            weighted_p += (precision or 0.0) * support
            weighted_r += (recall or 0.0) * support
            weighted_f += (f1 or 0.0) * support

    micro_p = _safe_div(total_tp, total_tp + total_fp)
    micro_r = _safe_div(total_tp, total_tp + total_fn)
    if micro_p is None or micro_r is None or (micro_p + micro_r) == 0:
        micro_f = None
    else:
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)

    macro = {
        "precision": macro_p / macro_n if macro_n else None,
        "recall": macro_r / macro_n if macro_n else None,
        "f1": macro_f / macro_n if macro_n else None,
    }
    micro = {"precision": micro_p, "recall": micro_r, "f1": micro_f}
    weighted = {
        "precision": weighted_p / total_support if total_support else None,
        "recall": weighted_r / total_support if total_support else None,
        "f1": weighted_f / total_support if total_support else None,
    }
    accuracy = _safe_div(total_tp, total_support)
    return per_class, macro, micro, weighted, accuracy, total_support


def compute_emotion_rows(
    results_by_service,
    oracle_results,
    vea_data,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
):
    """Per-sample, per-service correctness rows (long format)."""
    if not results_by_service:
        raise ValueError("No emotion service results provided for metric calculation.")

    predictions_by_service = _build_predictions_by_service(results_by_service)
    oracle_by_id = dict(
        zip(
            oracle_results["id"].map(_normalize_id),
            oracle_results["llm_oracle"].map(_oracle_top_emotion),
        )
    )

    rows: List[Dict] = []
    for _, row in vea_data.iterrows():
        sample_id = row["id"]
        sample_id_key = _normalize_id(sample_id)
        human_label = _normalize_label(label_to_name(row.get("label")))
        oracle_label = oracle_by_id.get(sample_id_key, "")

        for name, predictions in predictions_by_service.items():
            pred = predictions.get(sample_id_key, "")
            rows.append({
                "id": sample_id_key,
                "service": name,
                "prediction": pred,
                "oracle_label": oracle_label,
                "human_label": human_label,
                "oracle_correct": 1 if (oracle_label and pred == oracle_label) else 0,
                "human_correct": 1 if (human_label and pred == human_label) else 0,
            })

    return rows


def compute_emotion_summary_rows(
    per_sample_rows: List[Dict],
    service_names: Sequence[str],
    labels: Sequence[str] = CANONICAL_EMOTIONS,
):
    """Per-service summary: per-class P/R/F1 + accuracy + macro/micro/weighted P/R/F1.

    Two reference frames are reported side by side:
      - oracle_*: service prediction vs LLM oracle label
      - human_*:  service prediction vs VEA ground-truth label
    """
    counts_df = pd.DataFrame(per_sample_rows)
    summary_rows: List[Dict] = []

    for name in service_names:
        slice_df = counts_df[counts_df["service"] == name]
        oracle_counts = _per_class_counts(
            slice_df["oracle_label"].tolist(),
            slice_df["prediction"].tolist(),
            labels,
        )
        human_counts = _per_class_counts(
            slice_df["human_label"].tolist(),
            slice_df["prediction"].tolist(),
            labels,
        )
        oracle_per, oracle_macro, oracle_micro, oracle_weighted, oracle_acc, oracle_n = _aggregate(oracle_counts)
        human_per, human_macro, human_micro, human_weighted, human_acc, human_n = _aggregate(human_counts)

        row: Dict = {
            "service": name,
            "n_samples": human_n or oracle_n,
            "oracle_accuracy": oracle_acc,
            "human_accuracy": human_acc,
            "oracle_macro_f1": oracle_macro["f1"],
            "human_macro_f1": human_macro["f1"],
            "oracle_micro_f1": oracle_micro["f1"],
            "human_micro_f1": human_micro["f1"],
            "oracle_weighted_f1": oracle_weighted["f1"],
            "human_weighted_f1": human_weighted["f1"],
        }
        summary_rows.append(row)

    return summary_rows


__all__ = ["compute_emotion_rows", "compute_emotion_summary_rows"]
