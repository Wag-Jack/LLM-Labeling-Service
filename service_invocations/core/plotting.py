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


def _grouped_bar_per_prompt(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    ylabel: str,
    out_path: Path,
    suptitle: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """One faceted PNG: a subplot per prompt, each using the standard grouped bar.

    Skipped when fewer than 2 prompts are present — the single-prompt case is
    already covered by the aggregate chart.
    """
    if "prompt" not in df.columns or df.empty:
        return
    prompts = sorted(df["prompt"].dropna().astype(str).unique())
    if len(prompts) < 2:
        return
    ncols = 1 if len(prompts) <= 2 else 2
    nrows = (len(prompts) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 4.5 * nrows), squeeze=False,
    )
    for idx, prompt in enumerate(prompts):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["prompt"].astype(str) == prompt]
        _grouped_bar(ax, sub, x=x, y=y, hue=hue, ylabel=ylabel)
        ax.margins(y=0.15)
        ax.set_title(f"prompt: {prompt}")
        if ylim is not None:
            ax.set_ylim(*ylim)
    for idx in range(len(prompts), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _grouped_bar(ax, df: pd.DataFrame, x: str, y: str, hue: str, ylabel: str) -> None:
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    if pivot.empty:
        return
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
        # Annotate each bar with its exact value (thousandths); skip NaN bars.
        value_labels = ["" if pd.isna(v) else f"{v:.3f}" for v in values]
        ax.bar_label(bars, labels=value_labels, padding=2, fontsize=6, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize="small")


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


def plot_oracle_bundle(task_dir: Path, task: str) -> None:
    summary = _read_csv_or_none(task_dir / "accuracy_summary.csv")
    if summary is None:
        print(f"[plot] {task}/oracle: no accuracy_summary.csv, skipping")
        return
    cfg = _TASK_CONFIG[task]
    out_dir = _ensure_dir(task_dir / "plots" / "oracle")

    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, summary, x="service", y=cfg["metric_col"], hue="model", ylabel=cfg["ylabel"])
    ax.margins(y=0.15)  # headroom for vertical value labels
    ax.set_title(f"{task} – service accuracy vs human reference (by oracle LLM)")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_service.png", dpi=150)
    plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, long, x="service", y="metric", hue="reference", ylabel=cfg["ylabel"])
    ax.margins(y=0.15)  # headroom for vertical value labels
    ax.set_title(f"{task} – oracle-ref vs human-ref consistency")
    fig.tight_layout()
    fig.savefig(out_dir / "consistency_oracle_vs_human.png", dpi=150)
    plt.close(fig)

    _grouped_bar_per_prompt(
        long, x="service", y="metric", hue="reference", ylabel=cfg["ylabel"],
        out_path=out_dir / "consistency_oracle_vs_human_by_prompt.png",
        suptitle=f"{task} – oracle-ref vs human-ref consistency (per prompt)",
    )

    _plot_cost_breakdown(task_dir, task, paradigm="oracle", out_dir=out_dir)


def plot_judge_bundle(task_dir: Path, task: str) -> None:
    judge = _read_csv_or_none(task_dir / "judge.csv")
    if judge is None:
        print(f"[plot] {task}/judge: no judge.csv, skipping")
        return
    out_dir = _ensure_dir(task_dir / "plots" / "judge")
    _plot_winner_and_consistency(judge, task_dir, task, out_dir,
                                 paradigm_label="judge", exclude_fallback=False)
    _plot_cost_breakdown(task_dir, task, paradigm="judge", out_dir=out_dir)


def plot_human_loop_bundle(task_dir: Path, task: str) -> None:
    hl = _read_csv_or_none(task_dir / "human_loop.csv")
    if hl is None:
        print(f"[plot] {task}/human_loop: no human_loop.csv, skipping")
        return
    out_dir = _ensure_dir(task_dir / "plots" / "human_loop")
    _plot_winner_and_consistency(hl, task_dir, task, out_dir,
                                 paradigm_label="human-loop", exclude_fallback=True)

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
        _grouped_bar(ax, fallback, x="model", y="fallback_used", hue="prompt",
                     ylabel="Fraction of samples that fell back to human")
        ax.set_title(f"{task} – human-loop fallback rate")
        ax.set_ylim(0, 1.12)  # headroom for vertical value labels
        fig.tight_layout()
        fig.savefig(out_dir / "human_fallback_rate.png", dpi=150)
        plt.close(fig)

    _plot_cost_breakdown(task_dir, task, paradigm="human_loop", out_dir=out_dir)


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
        _grouped_bar(ax, rates, x="service", y="win_rate", hue="model",
                     ylabel="Fraction of samples picked as winner")
        ax.set_ylim(0, 1.12)  # headroom for vertical value labels
        ax.set_title(f"{task} – {paradigm_label}: LLM winner rate by service")
        fig.tight_layout()
        fig.savefig(out_dir / "winner_rate_by_service.png", dpi=150)
        plt.close(fig)

        _grouped_bar_per_prompt(
            rates, x="service", y="win_rate", hue="model",
            ylabel="Fraction of samples picked as winner",
            out_path=out_dir / "winner_rate_by_service_by_prompt.png",
            suptitle=f"{task} – {paradigm_label}: LLM winner rate by service (per prompt)",
            ylim=(0, 1.12),
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
    _grouped_bar(ax, per_m, x="model", y="correct", hue="prompt",
                 ylabel="Winner matches actually-best service (per human ref)")
    ax.set_ylim(0, 1.12)  # headroom for vertical value labels
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
    grouped = cost.groupby("model")[["input_tokens", "output_tokens", "cost_usd"]].sum().reset_index()
    grouped["cost_usd"] = grouped["cost_usd"].clip(lower=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.6
    x_pos = np.arange(len(grouped))
    ax.bar(x_pos, grouped["cost_usd"], width=width)
    for i, total in enumerate(grouped["cost_usd"]):
        ax.text(x_pos[i], total, f"${total:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped["model"], rotation=30, ha="right")
    ax.set_ylabel("USD")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{task}/{paradigm} – LLM cost by model")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_breakdown.png", dpi=150)
    plt.close(fig)


# ---------------------------- cross-paradigm summary ----------------------------


def plot_summary_bundle(task_dir: Path, task: str) -> None:
    out_dir = _ensure_dir(task_dir / "plots" / "summary")
    _plot_paradigm_pareto(task_dir, task, out_dir)
    _plot_paradigm_consistency(task_dir, task, out_dir)
    _plot_sds_top_k_curve(task_dir, task, out_dir)


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
    per_sample_col = cfg["per_sample_col"]
    if (accuracy is not None
            and {"prompt", "model"}.issubset(accuracy.columns)
            and per_sample_col in accuracy.columns):
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
    ax.set_xlabel("USD spent on LLM calls")
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
    _grouped_bar(ax, df, x="model", y="agreement", hue="paradigm",
                 ylabel="LLM winner matches actually-best service (per human ref)")
    ax.set_ylim(0, 1.12)  # headroom for vertical value labels
    ax.set_title(f"{task} – winner-pick agreement with human reference")
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
        paradigms = ("oracle", "judge", "human_loop")
    for paradigm in paradigms:
        if paradigm == "oracle":
            plot_oracle_bundle(task_dir, task)
        elif paradigm == "judge":
            plot_judge_bundle(task_dir, task)
        elif paradigm == "human_loop":
            plot_human_loop_bundle(task_dir, task)
    plot_summary_bundle(task_dir, task)


__all__ = [
    "plot_oracle_bundle",
    "plot_judge_bundle",
    "plot_human_loop_bundle",
    "plot_summary_bundle",
    "plot_all_for_task",
]
