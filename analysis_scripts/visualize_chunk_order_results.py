#!/usr/bin/env python3
"""
Visualize retrieved chunk-order permutation results (ROUGE-L + optional dosage/answer stability).

Research framing: if the model reasons over evidence, reordering the same retrieved chunks
(original vs reverse vs random) should not materially change recommendations. Low ROUGE-L or
dosage flips vs baseline suggest positional / presentation bias.

Inputs:
  - analysis_results/chunk_order_rouge_l/<model>/chunk_order_rouge_l_scores_ff.csv
  - Optional: experiment_results/<model>/chunk_order_tier1/results_tier1_chunk_order_permutations_ff.csv
  - Optional: experiment_results/<model>/*tier1* baseline CSV (original chunk order)

Outputs (under analysis_results/chunk_order_rouge_l/plots/):
  - PNG figures (ROUGE distributions, vignette heatmaps, reverse-vs-random scatter)
  - chunk_order_stability_summary_ff.csv (per model × permutation aggregates)
  - chunk_order_vignette_rouge_ff.csv (per vignette means)
  - If perm CSV present: chunk_order_decision_agreement_ff.csv

Usage (from Q-Pain root):
  python analysis_scripts/visualize_chunk_order_results.py
  python analysis_scripts/visualize_chunk_order_results.py --model gpt4o_mini_retrieved
"""

from __future__ import annotations

import argparse
import glob
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PERMUTATIONS = ("reverse", "random")
ROUGE_THRESHOLDS = (0.5, 0.7, 0.9)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_models_from_results(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, name)
        scores = os.path.join(p, "chunk_order_rouge_l_scores_ff.csv")
        if os.path.isdir(p) and os.path.isfile(scores):
            out.append(name)
    return out


def find_tier1_csv(model_dir: str) -> str | None:
    patterns = [
        os.path.join(model_dir, "*tier1*_ff*.csv"),
        os.path.join(model_dir, "*tier1*.csv"),
    ]
    for pat in patterns:
        matches = [
            m
            for m in sorted(glob.glob(pat))
            if ".bak." not in os.path.basename(m)
        ]
        if matches:
            return matches[0]
    return None


def find_perm_csv(model_dir: str) -> str | None:
    p = os.path.join(
        model_dir, "chunk_order_tier1", "results_tier1_chunk_order_permutations_ff.csv"
    )
    return p if os.path.isfile(p) else None


def _detect_dosage_column(df: pd.DataFrame) -> str | None:
    if "gpt4o_dosage" in df.columns:
        return "gpt4o_dosage"
    candidates = [c for c in df.columns if str(c).lower().endswith("_dosage")]
    if len(candidates) == 1:
        return candidates[0]
    for c in sorted(candidates):
        if "gpt" in str(c).lower():
            return c
    return candidates[0] if candidates else None


def _detect_answer_column(df: pd.DataFrame) -> str | None:
    if "gpt4o_answer" in df.columns:
        return "gpt4o_answer"
    candidates = [c for c in df.columns if str(c).lower().endswith("_answer")]
    return candidates[0] if len(candidates) == 1 else None


def _normalize_dosage(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=series.index)
    out[s.str.contains("low", na=False)] = "low"
    out[s.str.contains("high", na=False)] = "high"
    out[s.str.contains("medium", na=False)] = "medium"
    out[s.str.contains("none", na=False)] = "none"
    return out


def _normalize_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.startswith("yes")


def load_rouge_scores(results_dir: str, model_folder: str) -> pd.DataFrame:
    path = os.path.join(results_dir, model_folder, "chunk_order_rouge_l_scores_ff.csv")
    df = pd.read_csv(path)
    df["model_folder"] = model_folder
    return df


def vignette_mean_rouge(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean ROUGE-L per (model, vignette, permutation) across 8 demographics."""
    g = (
        scores.groupby(["model_folder", "vignette_idx", "permutation_id"], as_index=False)[
            "rouge_l_f1"
        ]
        .agg(mean_rouge_l_f1="mean", min_rouge_l_f1="min", std_rouge_l_f1="std")
    )
    return g


def build_stability_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, perm), g in scores.groupby(["model_folder", "permutation_id"]):
        x = g["rouge_l_f1"].to_numpy(dtype=float)
        rows.append(
            {
                "model_folder": model,
                "permutation_id": perm,
                "comparison": f"{perm} vs original (baseline)",
                "n_rows": len(g),
                "mean_rouge_l_f1": float(np.nanmean(x)),
                "median_rouge_l_f1": float(np.nanmedian(x)),
                "std_rouge_l_f1": float(np.nanstd(x)),
                **{
                    f"pct_rouge_lt_{t}": float(np.mean(x < t) * 100.0)
                    for t in ROUGE_THRESHOLDS
                },
            }
        )
    return pd.DataFrame(rows)


def load_decision_agreement(
    model_folder: str, exp_dir: str
) -> pd.DataFrame | None:
    """
    Compare permuted answer/dosage to baseline Tier 1 (original chunk order).
    Returns row-level DataFrame or None if perm/baseline files missing.
    """
    model_dir = os.path.join(exp_dir, model_folder)
    perm_path = find_perm_csv(model_dir)
    base_path = find_tier1_csv(model_dir)
    if not perm_path or not base_path:
        return None

    perm = pd.read_csv(perm_path)
    base = pd.read_csv(base_path)
    dosage_col = _detect_dosage_column(perm)
    ans_col = _detect_answer_column(perm)
    if dosage_col is None or dosage_col not in base.columns:
        return None

    join_cols = ["vignette_idx", "race", "gender"]
    b = base[join_cols + [dosage_col]].copy()
    b = b.rename(columns={dosage_col: "baseline_dosage"})
    if ans_col and ans_col in base.columns:
        b[ans_col] = base[ans_col]
        b = b.rename(columns={ans_col: "baseline_answer"})

    pcols = join_cols + ["permutation_id", dosage_col]
    if ans_col:
        pcols.append(ans_col)
    p = perm[pcols].copy()
    p = p.rename(columns={dosage_col: "perm_dosage"})
    if ans_col:
        p = p.rename(columns={ans_col: "perm_answer"})

    merged = p.merge(b, on=join_cols, how="inner")
    merged["baseline_dosage_norm"] = _normalize_dosage(merged["baseline_dosage"])
    merged["perm_dosage_norm"] = _normalize_dosage(merged["perm_dosage"])
    merged["dosage_match"] = (
        merged["baseline_dosage_norm"] == merged["perm_dosage_norm"]
    ) & (merged["baseline_dosage_norm"] != "unknown")
    if "baseline_answer" in merged.columns:
        merged["baseline_yes"] = _normalize_yes(merged["baseline_answer"])
        merged["perm_yes"] = _normalize_yes(merged["perm_answer"])
        merged["yes_match"] = merged["baseline_yes"] == merged["perm_yes"]
    merged["model_folder"] = model_folder
    return merged


def plot_mean_rouge_bar(summary: pd.DataFrame, out_path: str) -> None:
    df = summary.copy()
    df["model_short"] = df["model_folder"].str.replace("_retrieved", "", regex=False)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * df["model_folder"].nunique()), 5))
    sns.barplot(
        data=df,
        x="model_short",
        y="mean_rouge_l_f1",
        hue="permutation_id",
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0.9, color="0.45", ls="--", lw=1, label="ROUGE-L = 0.9")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean ROUGE-L F1 vs original order", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(
        "Explanation similarity under permuted chunk order\n"
        "(higher = closer to original-order explanation)"
    )
    ax.legend(title="Chunk order", loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rouge_violin(all_scores: pd.DataFrame, out_path: str) -> None:
    df = all_scores.copy()
    df["model_short"] = df["model_folder"].str.replace("_retrieved", "", regex=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.violinplot(
        data=df,
        x="model_short",
        y="rouge_l_f1",
        hue="permutation_id",
        split=False,
        inner="quart",
        cut=0,
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0.5, color="0.55", ls=":", lw=1.2)
    ax.axhline(0.9, color="0.45", ls="--", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("ROUGE-L F1 vs original order")
    ax.set_xlabel("Model")
    ax.set_title("Per-prompt explanation stability (80 rows per model × permutation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_material_difference_rates(stab: pd.DataFrame, out_path: str) -> None:
    long = stab.melt(
        id_vars=["model_folder", "permutation_id"],
        value_vars=[f"pct_rouge_lt_{t}" for t in ROUGE_THRESHOLDS],
        var_name="threshold",
        value_name="pct_below",
    )
    long["threshold"] = long["threshold"].str.replace("pct_rouge_lt_", "", regex=False)
    long["model_short"] = long["model_folder"].str.replace("_retrieved", "", regex=False)

    g = sns.catplot(
        data=long,
        kind="bar",
        x="model_short",
        y="pct_below",
        hue="permutation_id",
        col="threshold",
        col_order=[str(t) for t in ROUGE_THRESHOLDS],
        height=4,
        aspect=1.1,
        palette="Set2",
        legend_out=False,
    )
    g.set_axis_labels("Model", "% prompts with ROUGE-L below threshold")
    g.set_titles("ROUGE-L < {col_name}")
    g.fig.suptitle(
        "Material explanation change vs original chunk order\n"
        "(low ROUGE-L ⇒ ordering may drive wording, not only evidence)",
        y=1.05,
        fontsize=11,
    )
    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_vignette_heatmap(vig: pd.DataFrame, out_path: str) -> None:
    """Heatmap: vignette × (model|permutation) mean ROUGE."""
    vig = vig.copy()
    vig["cell"] = (
        vig["model_folder"].str.replace("_retrieved", "", regex=False)
        + "\n"
        + vig["permutation_id"]
    )
    pivot = vig.pivot_table(
        index="vignette_idx", columns="cell", values="mean_rouge_l_f1", aggfunc="first"
    )
    col_order = sorted(pivot.columns)
    pivot = pivot[col_order]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(col_order)), 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Mean ROUGE-L F1"},
        ax=ax,
    )
    ax.set_title("Mean ROUGE-L by clinical vignette (avg over 8 demographics)")
    ax.set_xlabel("Model × chunk-order permutation")
    ax.set_ylabel("Vignette index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reverse_vs_random_scatter(vig: pd.DataFrame, out_path: str) -> None:
    """Per vignette & model: mean ROUGE under reverse vs random (both vs original baseline)."""
    rows: list[pd.DataFrame] = []
    for model, g in vig.groupby("model_folder"):
        wide = g.pivot(index="vignette_idx", columns="permutation_id", values="mean_rouge_l_f1")
        if not {"reverse", "random"}.issubset(wide.columns):
            continue
        wide = wide.reset_index()
        wide["model_folder"] = model
        rows.append(wide)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    df["model_short"] = df["model_folder"].str.replace("_retrieved", "", regex=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(
        data=df,
        x="reverse",
        y="random",
        hue="model_short",
        s=80,
        alpha=0.85,
        ax=ax,
    )
    lims = [0, 1.02]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="reverse = random")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Mean ROUGE-L: reverse vs original")
    ax.set_ylabel("Mean ROUGE-L: random vs original")
    ax.set_title(
        "Vignette-level consistency between two permutations\n"
        "(same 10 chunks; order only changes)"
    )
    ax.legend(title="Model", loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_decision_agreement(dec: pd.DataFrame, out_path: str) -> None:
    agg_spec: dict = {
        "pct_dosage_match": ("dosage_match", lambda s: 100.0 * s.mean()),
    }
    if "yes_match" in dec.columns:
        agg_spec["pct_yes_match"] = ("yes_match", lambda s: 100.0 * s.mean())
    agg = dec.groupby(["model_folder", "permutation_id"], as_index=False).agg(
        **{k: v for k, v in agg_spec.items()}
    )

    agg["model_short"] = agg["model_folder"].str.replace("_retrieved", "", regex=False)
    fig, axes = plt.subplots(1, 2 if "pct_yes_match" in agg.columns else 1, figsize=(10, 4.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    sns.barplot(
        data=agg,
        x="model_short",
        y="pct_dosage_match",
        hue="permutation_id",
        palette="Set2",
        ax=axes[0],
    )
    axes[0].set_ylabel("% rows matching baseline dosage")
    axes[0].set_xlabel("Model")
    axes[0].set_title("Dosage agreement vs original order")
    axes[0].set_ylim(0, 100)
    if "pct_yes_match" in agg.columns and len(axes) > 1:
        sns.barplot(
            data=agg,
            x="model_short",
            y="pct_yes_match",
            hue="permutation_id",
            palette="Set2",
            ax=axes[1],
        )
        axes[1].set_ylabel("% rows matching baseline Yes/No")
        axes[1].set_xlabel("Model")
        axes[1].set_title("Prescription decision (Yes/No) agreement")
        axes[1].set_ylim(0, 100)
    fig.suptitle("Recommendation stability under chunk reordering", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vignette_dosage_flip(dec: pd.DataFrame, out_path: str) -> None:
    """Per vignette: any demographic with dosage mismatch vs baseline."""
    rows: list[dict] = []
    for (model, perm), g in dec.groupby(["model_folder", "permutation_id"]):
        for v, gv in g.groupby("vignette_idx"):
            rows.append(
                {
                    "model_folder": model,
                    "permutation_id": perm,
                    "vignette_idx": v,
                    "pct_dosage_match": 100.0 * gv["dosage_match"].mean(),
                    "any_flip": not gv["dosage_match"].all(),
                }
            )
    vdf = pd.DataFrame(rows)
    vdf["model_short"] = vdf["model_folder"].str.replace("_retrieved", "", regex=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=vdf,
        x="model_short",
        y="pct_dosage_match",
        hue="permutation_id",
        palette="Set2",
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of 8 demographics with same dosage as original order")
    ax.set_xlabel("Model")
    ax.set_title(
        "Vignette-level dosage stability (chunk order shared within vignette)\n"
        "100% = all race×gender arms match baseline for that clinical case"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(model_filter: list[str] | None) -> None:
    root = _project_root()
    results_dir = os.path.join(root, "analysis_results", "chunk_order_rouge_l")
    exp_dir = os.path.join(root, "experiment_results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    models = discover_models_from_results(results_dir)
    if model_filter:
        models = [m for m in models if m in model_filter]
    if not models:
        print(f"No scored models under {results_dir}")
        return

    all_scores: list[pd.DataFrame] = []
    all_dec: list[pd.DataFrame] = []
    for model in models:
        print(f"Loading {model}...")
        all_scores.append(load_rouge_scores(results_dir, model))
        dec = load_decision_agreement(model, exp_dir)
        if dec is not None:
            all_dec.append(dec)
            print(f"  + decision agreement from perm CSV ({len(dec)} rows)")
        else:
            print("  (no perm CSV — ROUGE-only plots)")

    scores = pd.concat(all_scores, ignore_index=True)
    stab = build_stability_summary(scores)
    vig = vignette_mean_rouge(scores)

    stab.to_csv(
        os.path.join(results_dir, "chunk_order_stability_summary_ff.csv"),
        index=False,
    )
    vig.to_csv(
        os.path.join(results_dir, "chunk_order_vignette_rouge_ff.csv"),
        index=False,
    )

    plot_mean_rouge_bar(
        stab, os.path.join(plots_dir, "chunk_order_mean_rouge_bar_ff.png")
    )
    plot_rouge_violin(scores, os.path.join(plots_dir, "chunk_order_rouge_violin_ff.png"))
    plot_material_difference_rates(
        stab, os.path.join(plots_dir, "chunk_order_material_rouge_drop_ff.png")
    )
    plot_vignette_heatmap(
        vig, os.path.join(plots_dir, "chunk_order_vignette_rouge_heatmap_ff.png")
    )
    plot_reverse_vs_random_scatter(
        vig, os.path.join(plots_dir, "chunk_order_reverse_vs_random_scatter_ff.png")
    )

    if all_dec:
        dec_df = pd.concat(all_dec, ignore_index=True)
        dec_df.to_csv(
            os.path.join(results_dir, "chunk_order_decision_agreement_ff.csv"),
            index=False,
        )
        dec_agg_spec: dict = {
            "n": ("dosage_match", "count"),
            "pct_dosage_match": ("dosage_match", lambda s: 100.0 * s.mean()),
        }
        if "yes_match" in dec_df.columns:
            dec_agg_spec["pct_yes_match"] = ("yes_match", lambda s: 100.0 * s.mean())
        dec_summary = dec_df.groupby(
            ["model_folder", "permutation_id"], as_index=False
        ).agg(**dec_agg_spec)
        dec_summary.to_csv(
            os.path.join(results_dir, "chunk_order_decision_agreement_summary_ff.csv"),
            index=False,
        )
        plot_decision_agreement(
            dec_df, os.path.join(plots_dir, "chunk_order_decision_agreement_bar_ff.png")
        )
        plot_vignette_dosage_flip(
            dec_df, os.path.join(plots_dir, "chunk_order_vignette_dosage_match_ff.png")
        )

    print(f"\nSaved plots to {plots_dir}")
    print(f"Saved tables to {results_dir}")
    print("\nSummary (mean ROUGE-L vs original order):")
    print(
        stab[["model_folder", "permutation_id", "mean_rouge_l_f1", "pct_rouge_lt_0.5"]]
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize chunk-order permutation results")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Restrict to model folder(s), e.g. gpt4o_mini_retrieved",
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
