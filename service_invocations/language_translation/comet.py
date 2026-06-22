import os

# COMET's predict() forces a "fork" DataLoader context whenever MPS is available
# (Apple Silicon), and a few XLM-R ops occasionally lack an MPS kernel. Enabling
# the CPU fallback keeps scoring correct when we run on the Metal GPU. Set before
# torch initialises its MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch

from comet import download_model, load_from_checkpoint

from service_invocations.core.oracle_utils import (
    normalize_id as _normalize_id,
    oracle_id_map as _oracle_id_map,
)


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


# Loaded COMET checkpoints are cached for the lifetime of the process, keyed by
# model name. The checkpoint is a ~2 GB XLM-R model and loading it takes several
# seconds; without this cache it would be reloaded once per (prompt x oracle
# model) slice, since compute_comet_rows is called once per slice.
_MODEL_CACHE = {}


def _get_model(model_name: str):
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        _MODEL_CACHE[model_name] = model
    return model


def _predict_kwargs() -> dict:
    """Pick the fastest available backend for ``model.predict``.

    On Apple Silicon this returns the MPS GPU path (``gpus=1``) rather than
    ``gpus=0``: COMET forces a "fork" DataLoader context whenever MPS is
    available, which crashes the ``gpus=0`` / ``num_workers=0`` path. On this
    hardware MPS measures ~1.7x faster than CPU and is numerically identical
    (max score delta ~2e-07). CUDA hosts use the GPU; everything else stays on
    CPU, where ``gpus=0`` is safe because MPS is absent.
    """
    if torch.cuda.is_available():
        return {"gpus": 1, "accelerator": "gpu"}
    if torch.backends.mps.is_available():
        return {"gpus": 1, "accelerator": "mps"}
    return {"gpus": 0}


# Human-reference COMET scores, memoized for the lifetime of the process and
# keyed by (model, src, mt, human_ref). human_comet = COMET(src, service_mt,
# human_ref) does NOT depend on the LLM oracle (prompt or model), yet
# compute_comet_rows is called once per (prompt x oracle model) slice — so
# without this cache the identical human-reference pass is re-scored on every
# slice. Keying on the actual mt text keeps the LLMAAS pseudo-service correct:
# its mt is the oracle output and changes per model, so it never collides with a
# real service's cached score.
_HUMAN_SCORE_CACHE: dict = {}


def compute_comet_rows(results_by_service, oracle_results, europarl_data,
                       model_name: str = "Unbabel/wmt22-comet-da",
                       batch_size: int = 16):
    """Long-format per-sample COMET rows for write_accuracy(...).

    Every (service, sample) pair is scored against both the LLM-oracle reference
    and the human reference. Rather than issuing a separate ``model.predict``
    call per service and per reference type (2 * n_services Lightning passes,
    each with its own trainer/dataloader spin-up), all records are concatenated
    into a single batch and scored in one pass. COMET scores each segment
    independently and ``predict`` restores input order, so the merged pass is
    numerically identical to the per-service calls while letting length-batching
    pack padding across the whole pool.
    """
    if not results_by_service:
        raise ValueError("No translation results provided for COMET scoring.")

    outputs_by_service = _build_outputs_by_service(results_by_service)
    # Tolerant lookup: a model that produced no oracle rows yields {} rather
    # than raising KeyError, so each sample falls back to an empty reference.
    oracle_by_id = _oracle_id_map(oracle_results)

    ids = europarl_data["id"].tolist()
    sources = europarl_data["english"].tolist()
    human_refs = europarl_data["french"].tolist()
    sample_keys = [_normalize_id(sample_id) for sample_id in ids]

    model = _get_model(model_name)

    # Build one flat batch covering every (service, sample) pair. oracle_comet
    # is always scored fresh (its reference is the oracle output, which is what
    # changes between slices). human_comet is looked up in the process cache and
    # only added to the batch on a miss. Each appended record carries a slot that
    # says where its score belongs, so the single predict pass can be split back
    # out in any order.
    pair_scores: dict = {}   # (name, idx) -> {"oracle": float, "human": float}
    records = []             # records actually sent to the model this call
    slots = []               # parallel to records: (name, idx, ref_type, human_key|None)

    for name, outputs in outputs_by_service.items():
        for idx, (sample_key, src, human_ref) in enumerate(zip(sample_keys, sources, human_refs)):
            src_text = _as_text(src)
            mt_text = _as_text(outputs.get(sample_key, ""))
            pair_scores[(name, idx)] = {}

            records.append({
                "src": src_text,
                "mt": mt_text,
                "ref": _as_text(oracle_by_id.get(sample_key, "")),
            })
            slots.append((name, idx, "oracle", None))

            human_ref_text = _as_text(human_ref)
            human_key = (model_name, src_text, mt_text, human_ref_text)
            cached = _HUMAN_SCORE_CACHE.get(human_key)
            if cached is not None:
                pair_scores[(name, idx)]["human"] = cached
            else:
                records.append({"src": src_text, "mt": mt_text, "ref": human_ref_text})
                slots.append((name, idx, "human", human_key))

    if not records:
        return []

    scores = model.predict(records, batch_size=batch_size, **_predict_kwargs()).scores
    for (name, idx, ref_type, human_key), score in zip(slots, scores):
        value = float(score)
        pair_scores[(name, idx)][ref_type] = value
        if human_key is not None:
            _HUMAN_SCORE_CACHE[human_key] = value

    rows = []
    for name in outputs_by_service:
        for idx, sample_key in enumerate(sample_keys):
            slot = pair_scores[(name, idx)]
            rows.append({
                "id": sample_key,
                "service": name,
                "oracle_comet": slot["oracle"],
                "human_comet": slot["human"],
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
