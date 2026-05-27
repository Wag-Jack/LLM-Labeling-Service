import re

import pandas as pd

from service_invocations.core.oracle_utils import normalize_id as _normalize_id


_NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
_WS_RE = re.compile(r"\s+")


def _normalize_text(text):
    if text is None:
        return []
    text = text.lower().strip()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return []
    return text.split(" ")


def word_error_counts(reference, hypothesis):
    ref_words = _normalize_text(reference)
    hyp_words = _normalize_text(hypothesis)

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0 and m == 0:
        return 0, 0
    if n == 0:
        return m, 0
    if m == 0:
        return n, n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return dp[n][m], n


def _build_transcripts_by_service(results_by_service):
    transcripts_by_service = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        ids = df["id"].map(_normalize_id)
        transcripts_by_service[name] = dict(zip(ids, df["service_output"]))
    return transcripts_by_service


def compute_wer_rows(results_by_service, oracle_results, edacc_data):
    """Long-format per-sample WER rows for write_accuracy(...)."""
    if not results_by_service:
        raise ValueError("No speech results provided for WER calculation.")

    transcripts_by_service = _build_transcripts_by_service(results_by_service)
    oracle_transcripts = dict(zip(
        oracle_results["id"].map(_normalize_id),
        oracle_results["llm_oracle"],
    ))

    rows = []
    for _, row in edacc_data.iterrows():
        sample_id_key = _normalize_id(row["id"])
        human_ref = row["text"]
        oracle_ref = oracle_transcripts.get(sample_id_key)

        for name, transcripts in transcripts_by_service.items():
            hyp = transcripts.get(sample_id_key, "")
            oracle_err, oracle_ref_words = word_error_counts(oracle_ref, hyp)
            human_err, human_ref_words = word_error_counts(human_ref, hyp)
            oracle_wer = (oracle_err / oracle_ref_words) if oracle_ref_words > 0 else None
            human_wer = (human_err / human_ref_words) if human_ref_words > 0 else None
            rows.append({
                "id": sample_id_key,
                "service": name,
                "oracle_errors": oracle_err,
                "oracle_ref_words": oracle_ref_words,
                "oracle_wer": oracle_wer,
                "human_errors": human_err,
                "human_ref_words": human_ref_words,
                "human_wer": human_wer,
            })
    return rows


def compute_wer_summary_rows(per_sample_rows, service_names):
    """Corpus-level WER per service (sum-errors / sum-ref-words)."""
    df = pd.DataFrame(per_sample_rows)
    out = []
    for name in service_names:
        slice_df = df[df["service"] == name]
        oracle_errors = slice_df["oracle_errors"].sum()
        oracle_ref_words = slice_df["oracle_ref_words"].sum()
        human_errors = slice_df["human_errors"].sum()
        human_ref_words = slice_df["human_ref_words"].sum()
        out.append({
            "service": name,
            "oracle_wer": (oracle_errors / oracle_ref_words) if oracle_ref_words > 0 else None,
            "human_wer": (human_errors / human_ref_words) if human_ref_words > 0 else None,
            "n_samples": int(len(slice_df)),
        })
    return out
