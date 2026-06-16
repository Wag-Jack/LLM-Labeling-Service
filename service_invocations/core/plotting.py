"""Plot generation for paradigm and cross-paradigm summaries.

Reads only the consolidated CSVs written by results_io.py and the cost
tracker. Each plot function is best-effort — it logs a warning and skips
if the required input is missing rather than failing the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Per-task metric configuration.
#   metric_col: the column in accuracy_summary.csv to plot
#   lower_is_better: True for WER, False for COMET / accuracy
#   ylabel: axis label for the metric plot
#   correctness: a (column, op, threshold) tuple telling us how to derive
#                "service was correct on this sample" from accuracy.csv for
#                judge/human-loop consistency calculations.
# metric_col / oracle_metric_col reference accuracy_summary.csv columns.
# per_sample_col references the per-sample accuracy.csv column used by the
# cross-paradigm Pareto helper (ED differs from ASR/LT because the per-sample
# correctness is 0/1 while the summary stores aggregated accuracy).
_TASK_CONFIG = {
    "speech_recognition": {
        "metric_col": "human_wer",
        "oracle_metric_col": "oracle_wer",
        "per_sample_col": "human_wer",
        "lower_is_better": True,
        "ylabel": "WER (lower = better)",
        "correctness": ("human_wer", "lte", 0.30),
    },
    "language_translation": {
        "metric_col": "human_comet",
        "oracle_metric_col": "oracle_comet",
        "per_sample_col": "human_comet",
        "lower_is_better": False,
        "ylabel": "COMET (higher = better)",
        "correctness": ("human_comet", "gte", 0.70),
    },
    "emotion_detection": {
        "metric_col": "human_accuracy",
        "oracle_metric_col": "oracle_accuracy",
        "per_sample_col": "human_correct",
        "lower_is_better": False,
        "ylabel": "Accuracy (higher = better)",
        "correctness": ("human_correct", "eq", 1),
    },
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if not df.empty else None


def _avg_suffix(df: pd.DataFrame | None) -> str:
    """Title suffix that names the silent averaging, but only for benchmarks.

    A single-prompt run has nothing to disclose, so we return "". When the data
    spans >1 prompt (a benchmark across prompts), the aggregate plots collapse
    the prompt dimension to a mean and the title should say so.
    """
    if df is None or df.empty or "prompt" not in df.columns:
        return ""
    n = df["prompt"].dropna().astype(str).nunique()
    return f" (mean over {n} prompts)" if n > 1 else ""


def _tight_ylim(ax, vmin, vmax, headroom: float = 0.22) -> None:
    """Zoom the y-axis to the data range (with headroom for rotated value labels).

    Auto-scaling to the data — rather than anchoring at 0 — makes small
    differences between bars legible. Every bar carries its value label, so the
    non-zero baseline never hides the true magnitude.
    """
    if vmin is None or vmax is None or pd.isna(vmin) or pd.isna(vmax):
        return
    span = float(vmax) - float(vmin)
    if span <= 0:
        span = abs(float(vmax)) or 1.0
    ax.set_ylim(float(vmin) - 0.05 * span, float(vmax) + headroom * span)


def _pivot_range(pivot) -> tuple[float, float]:
    """(min, max) of a pivot's values, ignoring NaN; (nan, nan) if all-NaN."""
    if pivot is None:
        return float("nan"), float("nan")
    vals = np.asarray(pivot.values, dtype=float).ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.min()), float(vals.max())


def _apply_autoscale(drawn: list, share_y: bool = True) -> None:
    """Tight-autoscale a set of (ax, pivot) panels.

    When ``share_y`` the same y-limits (spanning every panel's data) are applied
    to all axes so they can be compared directly; otherwise each axis is scaled
    to its own data.
    """
    ranges = [_pivot_range(p) for _, p in drawn if p is not None]
    ranges = [(lo, hi) for lo, hi in ranges if not (pd.isna(lo) or pd.isna(hi))]
    if not ranges:
        return
    if share_y:
        vmin = min(lo for lo, _ in ranges)
        vmax = max(hi for _, hi in ranges)
        for ax, p in drawn:
            if p is not None:
                _tight_ylim(ax, vmin, vmax)
    else:
        for ax, p in drawn:
            if p is not None:
                lo, hi = _pivot_range(p)
                _tight_ylim(ax, lo, hi)


def _flat_bar(ax, labels, values, ylabel: str, color: str = "#4C72B0") -> None:
    """Single-series bar chart with a 2-dp value label on every bar."""
    values = np.asarray(values, dtype=float)
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=color)
    ax.bar_label(
        bars, labels=["" if pd.isna(v) else f"{v:.2f}" for v in values],
        padding=2, fontsize=6, rotation=90,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)


def _grouped_bar(ax, df: pd.DataFrame, x: str, y: str, hue: str, ylabel: str):
    """Grouped bar chart into ``ax``; returns the pivot (for autoscaling) or None.

    Every bar is labelled with its value (2 dp). The caller sets y-limits (via
    ``_tight_ylim`` / ``_apply_autoscale``) so related charts can share a scale.
    """
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    if pivot.empty:
        return None
    x_labels = list(pivot.index)
    hue_labels = list(pivot.columns)
    bar_width = 0.8 / max(len(hue_labels), 1)
    x_pos = np.arange(len(x_labels))
    for i, hue_val in enumerate(hue_labels):
        values = pivot[hue_val].values
        bars = ax.bar(
            x_pos + i * bar_width - 0.4 + bar_width / 2,
            values,
            width=bar_width,
            label=str(hue_val),
        )
        value_labels = ["" if pd.isna(v) else f"{v:.2f}" for v in values]
        ax.bar_label(bars, labels=value_labels, padding=2, fontsize=6, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize="small")
    return pivot


def _grouped_bar_faceted(
    df: pd.DataFrame,
    facet_col: str,
    x: str,
    y: str,
    hue: str,
    ylabel: str,
    out_path: Path,
    suptitle: str,
    min_facets: int = 2,
    facet_label: str | None = None,
    share_y: bool = True,
) -> None:
    """One faceted PNG: a subplot per distinct ``facet_col`` value, each a grouped bar.

    Skipped when fewer than ``min_facets`` facet values are present. Subplots are
    tight-autoscaled, sharing one y-scale when ``share_y`` so they compare directly.
    """
    if facet_col not in df.columns or df.empty:
        return
    facets = sorted(df[facet_col].dropna().astype(str).unique())
    if len(facets) < min_facets:
        return
    facet_label = facet_label or facet_col
    ncols = 1 if len(facets) <= 2 else 2
    nrows = (len(facets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 4.5 * nrows), squeeze=False,
    )
    drawn: list = []
    for idx, fval in enumerate(facets):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df[facet_col].astype(str) == fval]
        pivot = _grouped_bar(ax, sub, x=x, y=y, hue=hue, ylabel=ylabel)
        ax.set_title(f"{facet_label}: {fval}")
        drawn.append((ax, pivot))
    _apply_autoscale(drawn, share_y=share_y)
    for idx in range(len(facets), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grouped_bar_per_prompt(
    df: pd.DataFrame, x: str, y: str, hue: str, ylabel: str,
    out_path: Path, suptitle: str,
) -> None:
    """A subplot per prompt (hue=model). Thin wrapper over ``_grouped_bar_faceted``."""
    _grouped_bar_faceted(
        df, facet_col="prompt", x=x, y=y, hue=hue, ylabel=ylabel,
        out_path=out_path, suptitle=suptitle,
        min_facets=2, facet_label="prompt",
    )


def _grouped_boxplot(ax, df: pd.DataFrame, x: str, hue: str, y: str,
                     ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    """Grouped box plot: one x-axis tick per ``x`` value, one colored box per ``hue``.

    Used to show e.g. the score distribution each model assigns to each service —
    so you can spot "model A is harsher than model B on this service" at a glance.
    """
    df = df.dropna(subset=[y]).copy()
    if df.empty:
        return
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[y])
    if df.empty:
        return
    x_labels = sorted(df[x].dropna().astype(str).unique())
    hue_labels = sorted(df[hue].dropna().astype(str).unique())
    if not x_labels or not hue_labels:
        return
    box_width = 0.8 / len(hue_labels)
    x_pos = np.arange(len(x_labels))
    cmap = plt.get_cmap("tab10")
    handles = []
    for j, hue_val in enumerate(hue_labels):
        positions: list[float] = []
        data: list[list[float]] = []
        for i, x_val in enumerate(x_labels):
            mask = (df[x].astype(str) == x_val) & (df[hue].astype(str) == hue_val)
            vals = df.loc[mask, y].astype(float).tolist()
            if vals:
                positions.append(x_pos[i] + j * box_width - 0.4 + box_width / 2)
                data.append(vals)
        if not data:
            continue
        color = cmap(j % 10)
        bp = ax.boxplot(
            data, positions=positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=True, manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
        for median in bp["medians"]:
            median.set_color("black")
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.65, label=str(hue_val)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if handles:
        ax.legend(handles=handles, loc="best", fontsize="small")


def _score_distribution_summary(
    df: pd.DataFrame, group_cols: list[str], value_col: str,
) -> pd.DataFrame:
    """Mean / std / quartile summary for ``value_col`` grouped by ``group_cols``."""
    rows: list[dict] = []
    if df.empty:
        return pd.DataFrame()
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        row = dict(zip(group_cols, keys))
        row["count"] = int(vals.size)
        row["mean"] = round(float(vals.mean()), 6)
        row["std"] = round(float(vals.std(ddof=1)), 6) if vals.size > 1 else 0.0
        row["min"] = round(float(vals.min()), 6)
        row["p25"] = round(float(vals.quantile(0.25)), 6)
        row["median"] = round(float(vals.median()), 6)
        row["p75"] = round(float(vals.quantile(0.75)), 6)
        row["max"] = round(float(vals.max()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_score_distributions(
    paradigm_df: pd.DataFrame, task: str, out_dir: Path, paradigm_label: str,
) -> None:
    """Emit the score-distribution box plots + summary CSV for one paradigm."""
    required = {"score", "service", "model"}
    if not required.issubset(paradigm_df.columns):
        return
    df = paradigm_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score", "service", "model"])
    # judge / human-loop emit score=-1 as a "model failed to rate this service"
    # sentinel. Drop those so they don't drag the mean/std/min toward -1.
    df = df[df["score"] >= 0]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    _grouped_boxplot(ax, df, x="service", hue="model", y="score",
                     ylabel="LLM-assigned score", ylim=(-0.05, 1.05))
    ax.set_title(
        f"{task} – {paradigm_label}: score distribution per service (by model)"
        f"{_avg_suffix(df)}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution_by_service.png", dpi=150)
    plt.close(fig)

    if "prompt" in df.columns:
        prompts = sorted(df["prompt"].dropna().astype(str).unique())
        if len(prompts) >= 2:
            ncols = 1 if len(prompts) <= 2 else 2
            nrows = (len(prompts) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(12 * ncols, 4.5 * nrows), squeeze=False,
            )
            for idx, prompt in enumerate(prompts):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                sub = df[df["prompt"].astype(str) == prompt]
                _grouped_boxplot(ax, sub, x="service", hue="model", y="score",
                                 ylabel="LLM-assigned score", ylim=(-0.05, 1.05))
                ax.set_title(f"prompt: {prompt}")
            for idx in range(len(prompts), nrows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].axis("off")
            fig.suptitle(
                f"{task} – {paradigm_label}: score distribution per service (by model, per prompt)"
            )
            fig.tight_layout()
            fig.savefig(out_dir / "score_distribution_by_service_by_prompt.png", dpi=150)
            plt.close(fig)

    summary_parts: list[pd.DataFrame] = []
    if "prompt" in df.columns:
        per_prompt = _score_distribution_summary(df, ["prompt", "model", "service"], "score")
        if not per_prompt.empty:
            per_prompt.insert(0, "scope", "per_prompt")
            summary_parts.append(per_prompt)
    aggregate = _score_distribution_summary(df, ["model", "service"], "score")
    if not aggregate.empty:
        aggregate.insert(0, "scope", "all_prompts")
        aggregate.insert(1, "prompt", "<all>")
        summary_parts.append(aggregate)
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True, sort=False)
        summary.to_csv(out_dir / "score_distribution_summary.csv", index=False)


def _best_services_per_sample(accuracy_df: pd.DataFrame, task: str) -> dict:
    """For each sample id, the set of services that achieve the best human metric.

    ED uses exact match against human_correct==1 (multiple services may tie).
    ASR picks min human_wer; LT picks max human_comet. Ties are kept so that
    a winner pick is counted correct if it matches any tied-best service.
    """
    cfg = _TASK_CONFIG[task]
    correctness_col, op, threshold = cfg["correctness"]
    best: dict[str, set] = {}

    if op == "eq" and correctness_col in accuracy_df.columns:
        wins = accuracy_df[accuracy_df[correctness_col] == threshold]
        for sample_id, grp in wins.groupby("id"):
            best[str(sample_id)] = set(grp["service"].astype(str))
        return best

    metric = cfg["metric_col"]
    if metric not in accuracy_df.columns:
        return {}
    df = accuracy_df[["id", "service", metric]].dropna(subset=[metric])
    pick = (lambda g: g.min()) if cfg["lower_is_better"] else (lambda g: g.max())
    for sample_id, grp in df.groupby("id"):
        target = pick(grp[metric])
        best[str(sample_id)] = set(grp[grp[metric] == target]["service"].astype(str))
    return best


def _winner_correctness(paradigm_df: pd.DataFrame, best_per_sample: dict,
                        exclude_fallback: bool = False) -> pd.DataFrame:
    """One row per (prompt, model, id): did the LLM winner match an actually-best service?

    Skips rows where winner is missing or the sample has no best-service info.
    When exclude_fallback=True and a 'fallback_used' column exists, fallback
    rows are dropped so we only measure genuine LLM judgement.
    """
    if "winner" not in paradigm_df.columns:
        return pd.DataFrame()
    df = paradigm_df.drop_duplicates(["prompt", "model", "id"])
    if exclude_fallback and "fallback_used" in df.columns:
        df = df[~df["fallback_used"].fillna(False).astype(bool)]
    df = df[df["winner"].notna() & (df["winner"] != "")]
    if df.empty:
        return df
    df = df.copy()
    df["correct"] = df.apply(
        lambda r: 1 if str(r["winner"]) in best_per_sample.get(str(r["id"]), set()) else 0,
        axis=1,
    )
    return df[["prompt", "model", "id", "winner", "correct"]]


# ---------------------------- per-paradigm plots ----------------------------


def _entity_accuracy_frame(
    summary: pd.DataFrame, llmaas: pd.DataFrame | None, metric: str,
    group_extra: list[str] | None = None,
) -> pd.DataFrame:
    """Combine real services and the LLMs into one 'entity' frame for ``metric``.

    Real services collapse to one row each (the human-ref metric is independent
    of the grading model/prompt). Each LLM becomes a row ``llm:<model>`` carrying
    its own standalone metric from ``llmaas_summary.csv``. ``group_extra`` (e.g.
    ["prompt"]) keeps that column so the caller can facet.
    """
    keep = (group_extra or [])
    frames: list[pd.DataFrame] = []
    if metric in summary.columns and "service" in summary.columns:
        svc = (
            summary.dropna(subset=[metric])
            .groupby(keep + ["service"])[metric].mean().reset_index()
        )
        svc = svc.rename(columns={"service": "entity"})
        svc["kind"] = "service"
        frames.append(svc)
    if (llmaas is not None and metric in llmaas.columns
            and "model" in llmaas.columns):
        llm = (
            llmaas.dropna(subset=[metric])
            .groupby(keep + ["model"])[metric].mean().reset_index()
        )
        llm["entity"] = "llm:" + llm["model"].astype(str)
        llm["kind"] = "llm"
        frames.append(llm[keep + ["entity", metric, "kind"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _flat_entity_bar(ax, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    """Flat (single-series) bar chart of ``metric`` per entity, colored by kind.

    Every bar is value-labelled (2 dp) and the y-axis is tight-autoscaled to the
    data so differences between entities are easy to read.
    """
    if frame.empty:
        return
    frame = frame.sort_values(["kind", "entity"]).reset_index(drop=True)
    colors = ["#4C72B0" if k == "service" else "#C44E52" for k in frame["kind"]]
    x_pos = np.arange(len(frame))
    bars = ax.bar(x_pos, frame[metric].values, color=colors)
    ax.bar_label(
        bars, labels=[f"{v:.2f}" if pd.notna(v) else "" for v in frame[metric]],
        padding=2, fontsize=6, rotation=90,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(frame["entity"], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    vals = pd.to_numeric(frame[metric], errors="coerce")
    _tight_ylim(ax, vals.min(), vals.max())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
        plt.Rectangle((0, 0), 1, 1, color="#C44E52"),
    ]
    ax.legend(handles, ["speech/STT service", "LLM (oracle) as service"],
              loc="best", fontsize="small")


def _plot_accuracy_all_entities(
    summary: pd.DataFrame, llmaas: pd.DataFrame | None, task: str, out_dir: Path,
) -> None:
    """Graph A: every transcriber as a service — services AND each LLM — scored
    by the task metric against the *human* reference, in one flat bar chart."""
    cfg = _TASK_CONFIG[task]
    metric = cfg["metric_col"]
    combined = _entity_accuracy_frame(summary, llmaas, metric)
    if combined.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(combined) + 4), 5))
    _flat_entity_bar(ax, combined, metric, cfg["ylabel"])
    ax.set_title(
        f"{task} – accuracy vs human reference: services and LLMs"
        f"{_avg_suffix(summary)}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_service.png", dpi=150)
    plt.close(fig)

    _plot_accuracy_services_vs_llms(summary, llmaas, task, out_dir)


def _plot_accuracy_services_vs_llms(
    summary: pd.DataFrame, llmaas: pd.DataFrame | None, task: str, out_dir: Path,
) -> None:
    """accuracy_by_service_by_prompt: two subplots side by side, shared y-scale.

      left  – the STT services on their own (one bar each; the human-ref metric
              does not depend on the prompt, so a single bar per service).
      right – the LLMs, each with one bar per prompt, so prompt sensitivity of
              the LLM-as-a-service is visible.
    """
    cfg = _TASK_CONFIG[task]
    metric = cfg["metric_col"]
    if metric not in summary.columns:
        return
    svc = (
        summary.dropna(subset=[metric])
        .groupby("service")[metric].mean().sort_index()
    )
    if llmaas is None or metric not in (llmaas.columns if llmaas is not None else []):
        return
    llm = llmaas.dropna(subset=[metric]).copy()
    if llm.empty or "prompt" not in llm.columns:
        return
    llm["entity"] = "llm:" + llm["model"].astype(str)

    fig, (ax_svc, ax_llm) = plt.subplots(1, 2, figsize=(16, 5))
    _flat_bar(ax_svc, list(svc.index), svc.values, cfg["ylabel"])
    ax_svc.set_title("STT services (human reference)")
    llm_pivot = _grouped_bar(ax_llm, llm, x="entity", y=metric, hue="prompt",
                             ylabel=cfg["ylabel"])
    ax_llm.set_title("LLMs as a service (per prompt)")

    # Shared, tight y-scale across both panels so services and LLMs compare directly.
    lo_svc = float(np.nanmin(svc.values)) if len(svc) else float("nan")
    hi_svc = float(np.nanmax(svc.values)) if len(svc) else float("nan")
    lo_llm, hi_llm = _pivot_range(llm_pivot)
    los = [v for v in (lo_svc, lo_llm) if not pd.isna(v)]
    his = [v for v in (hi_svc, hi_llm) if not pd.isna(v)]
    if los and his:
        _tight_ylim(ax_svc, min(los), max(his))
        _tight_ylim(ax_llm, min(los), max(his))
    fig.suptitle(
        f"{task} – accuracy vs human reference: services vs LLMs (by prompt)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_service_by_prompt.png", dpi=150)
    plt.close(fig)


def _plot_consistency_ref_split(
    summary: pd.DataFrame, task: str, hue: str, out_path: Path,
    suptitle: str, min_hue: int = 2,
) -> None:
    """Two subplots — human reference | oracle reference — sharing one y-scale.

    Within each subplot x=service and one bar per ``hue`` value (model or prompt),
    so e.g. the by-model view shows each service with one bar per grading model.
    Skipped when there are fewer than ``min_hue`` distinct hue values.
    """
    cfg = _TASK_CONFIG[task]
    needed = {hue, "service", cfg["metric_col"], cfg["oracle_metric_col"]}
    if not needed.issubset(summary.columns):
        return
    if summary[hue].dropna().astype(str).nunique() < min_hue:
        return
    fig, (ax_h, ax_o) = plt.subplots(1, 2, figsize=(20, 5.5))
    # Human reference: the service WER is independent of the grading model/prompt,
    # so show a single bar per service rather than redundant per-hue copies.
    svc = (
        summary.dropna(subset=[cfg["metric_col"]])
        .groupby("service")[cfg["metric_col"]].mean().sort_index()
    )
    _flat_bar(ax_h, list(svc.index), svc.values, cfg["ylabel"])
    ax_h.set_title("human reference")
    # Oracle reference depends on the model/prompt, so keep one bar per hue value.
    p_o = _grouped_bar(ax_o, summary, x="service", y=cfg["oracle_metric_col"], hue=hue,
                       ylabel=cfg["ylabel"])
    ax_o.set_title("oracle reference")
    # Same scale on both panels so the human vs oracle magnitudes compare directly.
    lo_o, hi_o = _pivot_range(p_o)
    lo_h = float(np.nanmin(svc.values)) if len(svc) else float("nan")
    hi_h = float(np.nanmax(svc.values)) if len(svc) else float("nan")
    los = [v for v in (lo_h, lo_o) if not pd.isna(v)]
    his = [v for v in (hi_h, hi_o) if not pd.isna(v)]
    if los and his:
        _tight_ylim(ax_h, min(los), max(his))
        _tight_ylim(ax_o, min(los), max(his))
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_oracle_bundle(task_dir: Path, task: str) -> None:
    summary = _read_csv_or_none(task_dir / "accuracy_summary.csv")
    if summary is None:
        print(f"[plot] {task}/oracle: no accuracy_summary.csv, skipping")
        return
    cfg = _TASK_CONFIG[task]
    out_dir = _ensure_dir(task_dir / "plots" / "oracle")
    llmaas = _read_csv_or_none(task_dir / "llmaas_summary.csv")

    # Graph A: every transcriber (services + each LLM) vs the HUMAN reference.
    _plot_accuracy_all_entities(summary, llmaas, task, out_dir)

    # Graph B: services scored against each LLM's transcript (oracle reference),
    # one subplot per model.
    _grouped_bar_faceted(
        summary, facet_col="model", x="service", y=cfg["oracle_metric_col"],
        hue="prompt", ylabel=cfg["ylabel"],
        out_path=out_dir / "accuracy_vs_llm_reference.png",
        suptitle=f"{task} – services scored against each model's LLM transcript (oracle reference)",
        min_facets=1, facet_label="model",
    )

    # Oracle-ref vs human-ref consistency. Aggregate (x=service, hue=reference),
    # plus two faceted views that split human vs oracle reference into subplots
    # with one bar per model / per prompt.
    long = summary.melt(
        id_vars=["prompt", "model", "service"],
        value_vars=[cfg["oracle_metric_col"], cfg["metric_col"]],
        var_name="reference",
        value_name="metric",
    )
    long["reference"] = long["reference"].replace({
        cfg["oracle_metric_col"]: "oracle ref",
        cfg["metric_col"]: "human ref",
    })
    n_models = summary["model"].dropna().astype(str).nunique() if "model" in summary.columns else 1
    model_note = f" (mean over {n_models} models)" if n_models > 1 else ""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = _grouped_bar(ax, long, x="service", y="metric", hue="reference", ylabel=cfg["ylabel"])
    _tight_ylim(ax, *_pivot_range(pivot))
    ax.set_title(f"{task} – oracle-ref vs human-ref consistency{model_note}{_avg_suffix(summary)}")
    fig.tight_layout()
    fig.savefig(out_dir / "consistency_oracle_vs_human.png", dpi=150)
    plt.close(fig)

    # human | oracle subplots, one bar per model per service.
    _plot_consistency_ref_split(
        summary, task, hue="model",
        out_path=out_dir / "consistency_oracle_vs_human_by_model.png",
        suptitle=f"{task} – oracle-ref vs human-ref consistency (one bar per model)",
    )
    # human | oracle subplots, one bar per prompt per service.
    _plot_consistency_ref_split(
        summary, task, hue="prompt",
        out_path=out_dir / "consistency_oracle_vs_human_by_prompt.png",
        suptitle=f"{task} – oracle-ref vs human-ref consistency (one bar per prompt)",
    )

    _plot_cost_breakdown(task_dir, task, paradigm="oracle", out_dir=out_dir)


def plot_llmaas_bundle(task_dir: Path, task: str) -> None:
    """Dedicated view of each model answering alone (LLMaaS), scored against the
    human reference — i.e. the LLM's quality *without* any service outcomes."""
    summary = _read_csv_or_none(task_dir / "llmaas_summary.csv")
    if summary is None:
        print(f"[plot] {task}/llmaas: no llmaas_summary.csv, skipping")
        return
    cfg = _TASK_CONFIG[task]
    out_dir = _ensure_dir(task_dir / "plots" / "llmaas")

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = _grouped_bar(ax, summary, x="model", y=cfg["metric_col"], hue="prompt", ylabel=cfg["ylabel"])
    _tight_ylim(ax, *_pivot_range(pivot))
    ax.set_title(f"{task} – LLMaaS standalone accuracy vs human reference (no services)")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_model.png", dpi=150)
    plt.close(fig)

    # LLMaaS reuses the oracle calls, so its spend is the oracle paradigm's cost.
    _plot_cost_breakdown(task_dir, task, paradigm="oracle", out_dir=out_dir)


def plot_judge_bundle(task_dir: Path, task: str) -> None:
    judge = _read_csv_or_none(task_dir / "judge.csv")
    if judge is None:
        print(f"[plot] {task}/judge: no judge.csv, skipping")
        return
    out_dir = _ensure_dir(task_dir / "plots" / "judge")
    _plot_winner_and_consistency(judge, task_dir, task, out_dir,
                                 paradigm_label="judge", exclude_fallback=False)
    _plot_score_distributions(judge, task, out_dir, paradigm_label="judge")
    _plot_cost_breakdown(task_dir, task, paradigm="judge", out_dir=out_dir)


def plot_human_loop_bundle(task_dir: Path, task: str) -> None:
    hl = _read_csv_or_none(task_dir / "human_loop.csv")
    if hl is None:
        print(f"[plot] {task}/human_loop: no human_loop.csv, skipping")
        return
    out_dir = _ensure_dir(task_dir / "plots" / "human_loop")
    _plot_winner_and_consistency(hl, task_dir, task, out_dir,
                                 paradigm_label="human-loop", exclude_fallback=True)
    _plot_score_distributions(hl, task, out_dir, paradigm_label="human-loop")
    _plot_confidence_histogram(hl, task, out_dir)

    if "fallback_used" in hl.columns:
        # fallback is per (sample, model, prompt) but stored per-service-row;
        # collapse so each sample/model counts once.
        fallback = (
            hl.drop_duplicates(["prompt", "model", "id"])
            .groupby(["prompt", "model"])["fallback_used"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot = _grouped_bar(ax, fallback, x="model", y="fallback_used", hue="prompt",
                             ylabel="Fraction of samples that fell back to human")
        _tight_ylim(ax, *_pivot_range(pivot))
        ax.set_title(f"{task} – human-loop fallback rate")
        fig.tight_layout()
        fig.savefig(out_dir / "human_fallback_rate.png", dpi=150)
        plt.close(fig)

    _plot_cost_breakdown(task_dir, task, paradigm="human_loop", out_dir=out_dir)


_CONFIDENCE_BINS = np.linspace(0.0, 1.0, 21)
_CONFIDENCE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def _draw_confidence_hist(ax, df: pd.DataFrame, models: list[str],
                          model_styles: dict[str, tuple]) -> int:
    """Overlay one stepped histogram per model onto ``ax``; returns the max count.

    ``model_styles`` is a shared {model: (color, marker)} map so the same model
    keeps the same look across every subplot when faceting.
    """
    centers = 0.5 * (_CONFIDENCE_BINS[:-1] + _CONFIDENCE_BINS[1:])
    width = _CONFIDENCE_BINS[1] - _CONFIDENCE_BINS[0]
    max_count = 0
    for model in models:
        vals = df.loc[df["model"].astype(str) == model, "confidence"].astype(float).values
        if vals.size == 0:
            continue
        counts, _ = np.histogram(vals, bins=_CONFIDENCE_BINS)
        if counts.max() > max_count:
            max_count = int(counts.max())
        color, marker = model_styles[model]
        # Filled step under the curve gives a rough density read at a glance;
        # the stepped outline + markers keep individual models readable when
        # the fills overlap.
        ax.fill_between(_CONFIDENCE_BINS[:-1], counts, step="post",
                        color=color, alpha=0.12, linewidth=0)
        ax.step(np.append(_CONFIDENCE_BINS[:-1], _CONFIDENCE_BINS[-1]),
                np.append(counts, counts[-1]),
                where="post", color=color, linewidth=2.0, alpha=0.95,
                label=f"{model} (n={vals.size})")
        ax.plot(centers, counts, linestyle="none", marker=marker,
                color=color, markersize=9, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.0 + 1e-9, 0.1))
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Sample count")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.margins(x=0.02)
    # Hint at the bin width for the reader.
    ax.text(0.99, 0.97, f"bin width = {width:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#555555")
    return max_count


def _plot_confidence_histogram(hl: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Overlaid confidence histograms — aggregate, plus per-prompt subplots.

    Confidence is recorded per (prompt, model, sample) but duplicated across
    every service row, so dedupe first. Each model gets a stepped histogram
    and a marker symbol at the bin centers so they remain distinguishable
    when the curves overlap. Per-model color/marker stays consistent across
    every subplot.
    """
    if "confidence" not in hl.columns or "model" not in hl.columns:
        return
    dedup_keys = [c for c in ("prompt", "model", "id") if c in hl.columns]
    df = hl.drop_duplicates(dedup_keys).copy() if dedup_keys else hl.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence", "model"])
    if df.empty:
        return

    models = sorted(df["model"].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    model_styles = {
        m: (cmap(i % 10), _CONFIDENCE_MARKERS[i % len(_CONFIDENCE_MARKERS)])
        for i, m in enumerate(models)
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _draw_confidence_hist(ax, df, models, model_styles)
    ax.set_title(
        f"{task} – human-loop: confidence distribution by model{_avg_suffix(df)}"
    )
    ax.legend(loc="best", fontsize="small", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)

    if "prompt" not in df.columns:
        return
    prompts = sorted(df["prompt"].dropna().astype(str).unique())
    if len(prompts) < 2:
        return
    ncols = 1 if len(prompts) <= 2 else 2
    nrows = (len(prompts) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11 * ncols, 5.0 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )
    panel_maxes: list[int] = []
    for idx, prompt in enumerate(prompts):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["prompt"].astype(str) == prompt]
        panel_max = _draw_confidence_hist(ax, sub, models, model_styles)
        panel_maxes.append(panel_max)
        ax.set_title(f"prompt: {prompt}")
        ax.legend(loc="best", fontsize="small", framealpha=0.9)
    for idx in range(len(prompts), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if panel_maxes:
        top = max(panel_maxes)
        if top > 0:
            for r in range(nrows):
                for c in range(ncols):
                    axes[r][c].set_ylim(0, top * 1.12)
    fig.suptitle(
        f"{task} – human-loop: confidence distribution by model (per prompt)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histogram_by_prompt.png", dpi=150)
    plt.close(fig)


def _plot_winner_and_consistency(paradigm_df: pd.DataFrame, task_dir: Path, task: str,
                                  out_dir: Path, paradigm_label: str,
                                  exclude_fallback: bool) -> None:
    """Shared judge/human-loop plots driven by the `winner` column."""
    if "winner" not in paradigm_df.columns:
        return

    # Per-sample winner picks (one row per prompt/model/sample).
    picks = paradigm_df.drop_duplicates(["prompt", "model", "id"]).copy()
    if exclude_fallback and "fallback_used" in picks.columns:
        picks = picks[~picks["fallback_used"].fillna(False).astype(bool)]
    has_winner = picks[picks["winner"].notna() & (picks["winner"] != "")]

    if not has_winner.empty:
        # Winner rate: fraction of samples each service was picked as winner,
        # broken down by LLM model. Counts samples per (prompt, model) as the
        # denominator so the bars are comparable across services.
        totals = has_winner.groupby(["prompt", "model"]).size().rename("n").reset_index()
        counts = (
            has_winner.groupby(["prompt", "model", "winner"]).size()
            .rename("n_wins").reset_index()
        )
        rates = counts.merge(totals, on=["prompt", "model"])
        rates["win_rate"] = rates["n_wins"] / rates["n"]
        rates = rates.rename(columns={"winner": "service"})

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = _grouped_bar(ax, rates, x="service", y="win_rate", hue="model",
                             ylabel="Fraction of samples picked as winner")
        _tight_ylim(ax, *_pivot_range(pivot))
        ax.set_title(f"{task} – {paradigm_label}: LLM winner rate by service{_avg_suffix(rates)}")
        fig.tight_layout()
        fig.savefig(out_dir / "winner_rate_by_service.png", dpi=150)
        plt.close(fig)

        _grouped_bar_per_prompt(
            rates, x="service", y="win_rate", hue="model",
            ylabel="Fraction of samples picked as winner",
            out_path=out_dir / "winner_rate_by_service_by_prompt.png",
            suptitle=f"{task} – {paradigm_label}: LLM winner rate by service (per prompt)",
        )

    # Consistency: does the LLM's winner match an actually-best service per
    # the human reference?
    accuracy = _read_csv_or_none(task_dir / "accuracy.csv")
    if accuracy is None:
        return
    best = _best_services_per_sample(accuracy, task)
    if not best:
        return
    correctness = _winner_correctness(paradigm_df, best, exclude_fallback=exclude_fallback)
    if correctness.empty:
        return
    per_m = correctness.groupby(["prompt", "model"])["correct"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = _grouped_bar(ax, per_m, x="model", y="correct", hue="prompt",
                         ylabel="Winner matches actually-best service (per human ref)")
    _tight_ylim(ax, *_pivot_range(pivot))
    ax.set_title(f"{task} – {paradigm_label}: winner vs human-best service")
    fig.tight_layout()
    fig.savefig(out_dir / "consistency_with_human.png", dpi=150)
    plt.close(fig)


def _plot_cost_breakdown(task_dir: Path, task: str, paradigm: str, out_dir: Path) -> None:
    cost = _read_csv_or_none(task_dir / "cost.csv")
    if cost is None:
        return
    cost = cost[(cost["task"] == task) & (cost["paradigm"] == paradigm)]
    # Drop unpriced / sentinel rows so they don't drag totals below zero.
    cost = cost[cost["cost_usd"].fillna(-1) >= 0]
    if cost.empty:
        return
    # Older cost.csv files predate the status column; treat them as all-success
    # so total and successful coincide (no failure data to split on).
    if "status" not in cost.columns:
        cost = cost.assign(status="success")
    total = cost.groupby("model")["cost_usd"].sum().rename("total")
    successful = (
        cost[cost["status"] == "success"].groupby("model")["cost_usd"].sum()
        .rename("successful")
    )
    grouped = pd.concat([total, successful], axis=1).fillna(0.0).reset_index()
    grouped["total"] = grouped["total"].clip(lower=0)
    grouped["successful"] = grouped["successful"].clip(lower=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(grouped))
    width = 0.38
    b_total = ax.bar(x_pos - width / 2, grouped["total"], width,
                     label="total (incl. failed outputs)", color="#4C72B0")
    b_succ = ax.bar(x_pos + width / 2, grouped["successful"], width,
                    label="successful only", color="#55A868")
    ax.bar_label(b_total, labels=[f"${v:.3f}" for v in grouped["total"]],
                 padding=2, fontsize=7, rotation=90)
    ax.bar_label(b_succ, labels=[f"${v:.3f}" for v in grouped["successful"]],
                 padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped["model"], rotation=30, ha="right")
    ax.set_ylabel("USD")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.legend(loc="best", fontsize="small")
    ax.set_title(f"{task}/{paradigm} – LLM cost by model (total vs successful)")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_breakdown.png", dpi=150)
    plt.close(fig)


# ---------------------------- cross-paradigm summary ----------------------------


def plot_summary_bundle(task_dir: Path, task: str) -> None:
    out_dir = _ensure_dir(task_dir / "plots" / "summary")
    _plot_paradigm_pareto(task_dir, task, out_dir)
    _plot_paradigm_consistency(task_dir, task, out_dir)
    _plot_sds_top_k_curve(task_dir, task, out_dir)
    _plot_total_cost(task_dir, task, out_dir)


def _plot_total_cost(task_dir: Path, task: str, out_dir: Path) -> None:
    """Whole-task spend in one chart: each LLM (across all paradigms) and each
    third-party service as its own bar, with the grand total in the title.

    Combines the LLM ``cost.csv`` (token-priced) and the service
    ``service_cost.csv`` (usage-priced) so the run's *total* invocation cost —
    not just the per-paradigm LLM slice — is visible at a glance.
    """
    rows: list[dict] = []

    llm = _read_csv_or_none(task_dir / "cost.csv")
    if llm is not None and {"model", "cost_usd"}.issubset(llm.columns):
        sub = llm[llm["task"] == task] if "task" in llm.columns else llm
        sub = sub[sub["cost_usd"].fillna(-1) >= 0]
        for model, amount in sub.groupby("model")["cost_usd"].sum().items():
            rows.append({"entity": f"llm:{model}", "kind": "LLM", "cost_usd": float(amount)})

    svc = _read_csv_or_none(task_dir / "service_cost.csv")
    if svc is not None and {"service", "cost_usd"}.issubset(svc.columns):
        sub = svc[svc["task"] == task] if "task" in svc.columns else svc
        sub = sub[sub["cost_usd"].fillna(-1) >= 0]
        for service, amount in sub.groupby("service")["cost_usd"].sum().items():
            rows.append({"entity": str(service), "kind": "service", "cost_usd": float(amount)})

    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["kind", "entity"]).reset_index(drop=True)
    grand_total = float(frame["cost_usd"].sum())

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(frame) + 4), 5))
    colors = ["#C44E52" if k == "LLM" else "#4C72B0" for k in frame["kind"]]
    x_pos = np.arange(len(frame))
    bars = ax.bar(x_pos, frame["cost_usd"].values, color=colors)
    ax.bar_label(bars, labels=[f"${v:.3f}" for v in frame["cost_usd"]],
                 padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(frame["entity"], rotation=30, ha="right")
    ax.set_ylabel("USD")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C44E52"),
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
    ]
    ax.legend(handles, ["LLM (all paradigms)", "third-party service"],
              loc="best", fontsize="small")
    ax.set_title(f"{task} – total invocation cost by entity (grand total: ${grand_total:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / "total_invocation_cost.png", dpi=150)
    plt.close(fig)


def _per_sample_label_accuracy(task_dir: Path, task: str) -> pd.DataFrame:
    """Per-paradigm, per-model 'label accuracy against the human reference'.

    Returns long-format rows: paradigm, model, score. Definitions:
      - oracle: 1 - mean(human metric) for ASR, mean(human metric) otherwise.
        Approximates "how good is the oracle label" via the services-vs-oracle
        metric we already store.
      - judge / human_loop: fraction of samples where the LLM's *winner* is
        an actually-best service per the human reference. For human_loop this
        excludes fallback rows so we measure the LLM judgement only.
      - majority_voting: average agreement reported by MV.
    """
    cfg = _TASK_CONFIG[task]
    rows: list[dict] = []

    accuracy = _read_csv_or_none(task_dir / "accuracy.csv")
    metric_col = cfg["metric_col"]
    per_sample_col = cfg["per_sample_col"]
    llmaas = _read_csv_or_none(task_dir / "llmaas_summary.csv")
    if (llmaas is not None
            and "model" in llmaas.columns
            and metric_col in llmaas.columns):
        # Real LLMaaS quality: the LLM's own answer scored vs the human reference
        # (the "no service outcomes" baseline to compare against judge/human-loop).
        per_m = llmaas.groupby("model")[metric_col].mean().reset_index()
        for _, r in per_m.iterrows():
            score = float(r[metric_col])
            if pd.isna(score):
                continue
            normalized = (1.0 - score) if cfg["lower_is_better"] else score
            rows.append({"paradigm": "llmaas", "model": str(r["model"]), "score": normalized})
    elif (accuracy is not None
            and {"prompt", "model"}.issubset(accuracy.columns)
            and per_sample_col in accuracy.columns):
        # Legacy proxy (pre-LLMaaS runs): average service accuracy vs human ref.
        per_pm = accuracy.groupby(["prompt", "model"])[per_sample_col].mean().reset_index()
        for _, r in per_pm.iterrows():
            score = float(r[per_sample_col])
            normalized = (1.0 - score) if cfg["lower_is_better"] else score
            rows.append({"paradigm": "oracle", "model": str(r["model"]), "score": normalized})

    best = _best_services_per_sample(accuracy, task) if accuracy is not None else {}

    for paradigm_name, path, exclude_fallback in (
        ("judge", task_dir / "judge.csv", False),
        ("human_loop", task_dir / "human_loop.csv", True),
    ):
        df = _read_csv_or_none(path)
        if df is None or df.empty or not best:
            continue
        correctness = _winner_correctness(df, best, exclude_fallback=exclude_fallback)
        if correctness.empty:
            continue
        per_m = correctness.groupby("model")["correct"].mean().reset_index()
        for _, r in per_m.iterrows():
            rows.append({"paradigm": paradigm_name, "model": str(r["model"]),
                         "score": float(r["correct"])})

    mv_path = task_dir / "majority_voting" / "majority_voting.csv"
    mv = _read_csv_or_none(mv_path)
    if mv is not None and "agreement" in mv.columns:
        rows.append({
            "paradigm": "majority_voting",
            "model": "(no LLM)",
            "score": float(mv["agreement"].mean()),
        })

    return pd.DataFrame(rows)


def _per_paradigm_cost(task_dir: Path, task: str) -> pd.DataFrame:
    cost = _read_csv_or_none(task_dir / "cost.csv")
    if cost is None:
        return pd.DataFrame(columns=["paradigm", "model", "cost_usd"])
    cost = cost[(cost["task"] == task) & (cost["cost_usd"].fillna(-1) >= 0)]
    if cost.empty:
        return pd.DataFrame(columns=["paradigm", "model", "cost_usd"])
    grouped = cost.groupby(["paradigm", "model"])["cost_usd"].sum().reset_index()
    grouped["cost_usd"] = grouped["cost_usd"].clip(lower=0)
    return grouped


def _plot_paradigm_pareto(task_dir: Path, task: str, out_dir: Path) -> None:
    accuracy = _per_sample_label_accuracy(task_dir, task)
    cost = _per_paradigm_cost(task_dir, task)
    if accuracy.empty:
        return
    merged = accuracy.merge(cost, on=["paradigm", "model"], how="left")
    merged["cost_usd"] = merged["cost_usd"].fillna(0.0).clip(lower=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    paradigms = merged["paradigm"].unique()
    cmap = plt.get_cmap("tab10")
    for i, paradigm in enumerate(paradigms):
        sub = merged[merged["paradigm"] == paradigm]
        ax.scatter(sub["cost_usd"], sub["score"], label=paradigm, color=cmap(i), s=80)
        for _, r in sub.iterrows():
            ax.annotate(str(r["model"]), (r["cost_usd"], r["score"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("USD spent on LLM calls (total, incl. billed failed-output attempts)")
    ax.set_ylabel("Label accuracy proxy (higher = better)")
    ax.set_xlim(left=0)
    ax.set_title(f"{task} – accuracy vs cost across paradigms")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "paradigm_accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


def _plot_paradigm_consistency(task_dir: Path, task: str, out_dir: Path) -> None:
    accuracy = _read_csv_or_none(task_dir / "accuracy.csv")
    if accuracy is None:
        return
    best = _best_services_per_sample(accuracy, task)
    if not best:
        return

    rows: list[dict] = []
    for paradigm_name, path, exclude_fallback in (
        ("judge", task_dir / "judge.csv", False),
        ("human_loop", task_dir / "human_loop.csv", True),
    ):
        df = _read_csv_or_none(path)
        if df is None or df.empty:
            continue
        correctness = _winner_correctness(df, best, exclude_fallback=exclude_fallback)
        if correctness.empty:
            continue
        per_m = correctness.groupby("model")["correct"].mean().reset_index()
        for _, r in per_m.iterrows():
            rows.append({"paradigm": paradigm_name, "model": str(r["model"]),
                         "agreement": float(r["correct"])})

    mv_path = task_dir / "majority_voting" / "majority_voting.csv"
    mv = _read_csv_or_none(mv_path)
    if mv is not None and "agreement" in mv.columns:
        rows.append({
            "paradigm": "majority_voting",
            "model": "(no LLM)",
            "agreement": float(mv["agreement"].mean()),
        })

    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = _grouped_bar(ax, df, x="model", y="agreement", hue="paradigm",
                         ylabel="LLM winner matches actually-best service (per human ref)")
    _tight_ylim(ax, *_pivot_range(pivot))
    n_prompts = accuracy["prompt"].dropna().astype(str).nunique() if "prompt" in accuracy.columns else 1
    prompt_note = f" (mean over {n_prompts} prompts)" if n_prompts > 1 else ""
    ax.set_title(f"{task} – winner-pick agreement with human reference{prompt_note}")
    fig.tight_layout()
    fig.savefig(out_dir / "paradigm_human_consistency.png", dpi=150)
    plt.close(fig)


def _plot_sds_top_k_curve(task_dir: Path, task: str, out_dir: Path) -> None:
    sds = _read_csv_or_none(task_dir / "sds" / "sds_ranking.csv")
    if sds is None or "discrimination" not in sds.columns:
        return
    sds = sds.sort_values("discrimination", ascending=False).reset_index(drop=True)
    n = len(sds)
    if n == 0:
        return
    ks = np.arange(1, n + 1)
    cumulative = sds["discrimination"].cumsum().values
    total = cumulative[-1] if cumulative[-1] != 0 else 1.0
    retained = cumulative / total

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, retained, marker="o")
    ax.set_xlabel("K (samples selected, ranked by SDS)")
    ax.set_ylabel("Fraction of total discrimination retained")
    ax.set_title(f"{task} – SDS top-K coverage curve")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "sds_top_k_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------- public API ----------------------------


def plot_all_for_task(task_dir: Path, task: str, paradigms: Iterable[str] | None = None) -> None:
    """Emit per-paradigm bundles + summary plots for one task directory."""
    if paradigms is None:
        paradigms = ("oracle", "llmaas", "judge", "human_loop")
    for paradigm in paradigms:
        if paradigm == "oracle":
            plot_oracle_bundle(task_dir, task)
        elif paradigm == "llmaas":
            plot_llmaas_bundle(task_dir, task)
        elif paradigm == "judge":
            plot_judge_bundle(task_dir, task)
        elif paradigm == "human_loop":
            plot_human_loop_bundle(task_dir, task)
    plot_summary_bundle(task_dir, task)


__all__ = [
    "plot_oracle_bundle",
    "plot_llmaas_bundle",
    "plot_judge_bundle",
    "plot_human_loop_bundle",
    "plot_summary_bundle",
    "plot_all_for_task",
]
