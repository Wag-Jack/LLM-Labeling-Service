import pandas as pd

from comet import download_model, load_from_checkpoint

from service_invocations.core.oracle_utils import normalize_id as _normalize_id


def _as_text(value) -> str:
    """Coerce a source/translation/reference cell to a plain string for COMET.

    ``None`` and pandas ``NaN`` (an empty CSV cell read back as a float) become
    "". Note that ``value or ""`` does NOT work here: ``bool(float("nan"))`` is
    True, so a NaN would slip through and be handed to the COMET model where a
    string is expected.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def _build_outputs_by_service(results_by_service):
    outputs_by_service = {}
    for name, df in results_by_service.items():
        if "id" not in df.columns or "service_output" not in df.columns:
            raise ValueError(f"Missing required columns for {name}.")
        ids = df["id"].map(_normalize_id)
        outputs_by_service[name] = dict(zip(ids, df["service_output"]))
    return outputs_by_service


def _score_pairs(model, sources, translations, references, batch_size=8):
    records = []
    for src, mt, ref in zip(sources, translations, references):
        records.append({
            "src": _as_text(src),
            "mt": _as_text(mt),
            "ref": _as_text(ref),
        })
    output = model.predict(records, batch_size=batch_size)
    return output.scores


def compute_comet_rows(results_by_service, oracle_results, europarl_data,
                       model_name: str = "Unbabel/wmt22-comet-da",
                       batch_size: int = 8):
    """Long-format per-sample COMET rows for write_accuracy(...)."""
    if not results_by_service:
        raise ValueError("No translation results provided for COMET scoring.")

    outputs_by_service = _build_outputs_by_service(results_by_service)
    oracle_by_id = dict(zip(oracle_results["id"].map(_normalize_id), oracle_results["llm_oracle"]))

    ids = europarl_data["id"].tolist()
    sources = europarl_data["english"].tolist()
    human_refs = europarl_data["french"].tolist()
    sample_keys = [_normalize_id(sample_id) for sample_id in ids]

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    rows = []
    for name, outputs in outputs_by_service.items():
        translations = [outputs.get(sample_key, "") for sample_key in sample_keys]
        oracle_refs = [oracle_by_id.get(sample_key, "") for sample_key in sample_keys]

        oracle_scores = _score_pairs(model, sources, translations, oracle_refs, batch_size=batch_size)
        human_scores = _score_pairs(model, sources, translations, human_refs, batch_size=batch_size)

        for sample_key, o_score, h_score in zip(sample_keys, oracle_scores, human_scores):
            rows.append({
                "id": sample_key,
                "service": name,
                "oracle_comet": float(o_score),
                "human_comet": float(h_score),
            })
    return rows


def compute_comet_summary_rows(per_sample_rows, service_names):
    df = pd.DataFrame(per_sample_rows)
    out = []
    for name in service_names:
        slice_df = df[df["service"] == name]
        out.append({
            "service": name,
            "oracle_comet": float(slice_df["oracle_comet"].mean()) if not slice_df.empty else None,
            "human_comet": float(slice_df["human_comet"].mean()) if not slice_df.empty else None,
            "n_samples": int(len(slice_df)),
        })
    return out
