import pandas as pd

from comet import download_model, load_from_checkpoint

from service_invocations.core.oracle_utils import normalize_id as _normalize_id

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
            "src": src or "",
            "mt": mt or "",
            "ref": ref or "",
        })
    output = model.predict(records, batch_size=batch_size, gpus=0)
    return output.scores


def compute_comet_scores(results_by_service, oracle_results, europarl_data,
                         model_name: str = "Unbabel/wmt22-comet-da",
                         batch_size: int = 8):
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

    service_names = list(outputs_by_service.keys())
    data = {"id": []}
    for name in service_names:
        data[f"{name}_oracle_comet"] = []
        data[f"{name}_human_comet"] = []

    data["id"] = sample_keys

    for name, outputs in outputs_by_service.items():
        translations = [outputs.get(sample_key, "") for sample_key in sample_keys]
        oracle_refs = [oracle_by_id.get(sample_key, "") for sample_key in sample_keys]

        oracle_scores = _score_pairs(
            model,
            sources,
            translations,
            oracle_refs,
            batch_size=batch_size,
        )
        human_scores = _score_pairs(
            model,
            sources,
            translations,
            human_refs,
            batch_size=batch_size,
        )

        data[f"{name}_oracle_comet"] = list(oracle_scores)
        data[f"{name}_human_comet"] = list(human_scores)

    return pd.DataFrame(data)


def compute_comet_summary(comet_scores_df, service_names):
    summary = {"service": [], "oracle_comet": [], "human_comet": []}
    for name in service_names:
        oracle_scores = comet_scores_df[f"{name}_oracle_comet"]
        human_scores = comet_scores_df[f"{name}_human_comet"]
        summary["service"].append(name)
        summary["oracle_comet"].append(oracle_scores.mean())
        summary["human_comet"].append(human_scores.mean())
    return pd.DataFrame(summary)
