import re
import pandas as pd


_NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
_WS_RE = re.compile(r"\s+")
_ID_RE = re.compile(r"(\d+)$")


def _normalize_text(text):
    if text is None:
        return []
    text = text.lower().strip()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return []
    return text.split(" ")


def _normalize_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return f"{int(value):04d}"
    value_str = str(value)
    match = _ID_RE.search(value_str)
    if match:
        digits = match.group(1)
        if len(digits) <= 4:
            return digits.zfill(4)
        return digits
    return value_str


def word_error_counts(reference, hypothesis):
    """
    Return (error_count, reference_word_count) for WER calculation.
    """
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
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1]  # substitution
                )

    return dp[n][m], n


def _init_data(service_names):
    data = {"id": []}
    for name in service_names:
        data[f"{name}_oracle_errors"] = []
        data[f"{name}_oracle_ref_words"] = []
        data[f"{name}_oracle_wer"] = []
        data[f"{name}_human_errors"] = []
        data[f"{name}_human_ref_words"] = []
        data[f"{name}_human_wer"] = []
    return data


def _build_transcripts_by_service(results_by_service):
    transcripts_by_service = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        ids = df["id"].map(_normalize_id)
        transcripts_by_service[name] = dict(zip(ids, df["service_output"]))
    return transcripts_by_service


def compute_wer_counts(results_by_service, oracle_results, edacc_data):
    """
    Compute WER numerator/denominator counts for each service against:
    - LLM oracle transcripts
    - Human ground truth (edacc_data['text'])

    Returns a DataFrame with per-sample error counts and reference word counts.
    """
    if not results_by_service:
        raise ValueError("No speech results provided for WER calculation.")

    transcripts_by_service = _build_transcripts_by_service(results_by_service)
    oracle_transcripts = dict(zip(oracle_results['id'].map(_normalize_id),
                                  oracle_results['llm_oracle']))

    service_names = list(transcripts_by_service.keys())
    data = _init_data(service_names)

    for _, row in edacc_data.iterrows():
        sample_id = row["id"]
        sample_id_key = _normalize_id(sample_id)
        human_ref = row["text"]
        oracle_ref = oracle_transcripts.get(sample_id_key)

        data["id"].append(sample_id_key)
        for name, transcripts in transcripts_by_service.items():
            hyp = transcripts.get(sample_id_key, "")
            oracle_err, oracle_ref_words = word_error_counts(oracle_ref, hyp)
            human_err, human_ref_words = word_error_counts(human_ref, hyp)
            oracle_wer = (oracle_err / oracle_ref_words) if oracle_ref_words > 0 else None
            human_wer = (human_err / human_ref_words) if human_ref_words > 0 else None
            data[f"{name}_oracle_errors"].append(oracle_err)
            data[f"{name}_oracle_ref_words"].append(oracle_ref_words)
            data[f"{name}_oracle_wer"].append(oracle_wer)
            data[f"{name}_human_errors"].append(human_err)
            data[f"{name}_human_ref_words"].append(human_ref_words)
            data[f"{name}_human_wer"].append(human_wer)

    return pd.DataFrame(data)


def compute_wer_summary(wer_counts_df, service_names):
    """
    Compute corpus-level WERs (sum of errors / sum of reference words) per service.
    """
    summary = {"service": [], "oracle_wer": [], "human_wer": []}
    for name in service_names:
        oracle_errors = wer_counts_df[f"{name}_oracle_errors"].sum()
        oracle_ref_words = wer_counts_df[f"{name}_oracle_ref_words"].sum()
        human_errors = wer_counts_df[f"{name}_human_errors"].sum()
        human_ref_words = wer_counts_df[f"{name}_human_ref_words"].sum()

        summary["service"].append(name)
        summary["oracle_wer"].append(
            (oracle_errors / oracle_ref_words) if oracle_ref_words > 0 else None
        )
        summary["human_wer"].append(
            (human_errors / human_ref_words) if human_ref_words > 0 else None
        )

    return pd.DataFrame(summary)
