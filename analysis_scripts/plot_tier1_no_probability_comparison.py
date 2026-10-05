"""
Tier 1 P("No") summaries for closed-book models vs. Q-Pain GPT-3 reference.

Reference (GPT-3, post-operative): mean P(No) ≈ 25.6% — Logé et al., Fig. 2a.

Requires Tier 1 CSVs with a log-probability column (e.g. prob_gpt4o_no).
Models without logprobs (e.g. GPT-5.4 closed-book) are skipped.

Usage (from Q-Pain root):
  python analysis_scripts/plot_tier1_no_probability_comparison.py

Outputs (analysis_results/tier1_no_prob/):
  - tier1_no_probability_summary_all_models_ff.csv
  - tier1_no_probability_all_models_bar_ff.png
  - figure1_no_probability_comparison.png  (GPT-4o-mini vs GPT-3 only, legacy)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QPAIN_GPT3_POSTOP_MEAN_NO_PROB = 0.256  # 25.6%

MODEL_LABELS: dict[str, str] = {
    "gpt4o_mini": "GPT-4o-mini",
    "gpt41_mini": "GPT-4.1-mini",
    "llama31": "Llama 3.1",
    "gemma3n-e4b": "Gemma 3n-E4B",
    "gpt54": "GPT-5.4",
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_closed_book_models(experiment_results_dir: str) -> list[str]:
    """Model folders under experiment_results/, excluding *_retrieved."""
    if not os.path.isdir(experiment_results_dir):
        return []
    return sorted(
        n
        for n in os.listdir(experiment_results_dir)
        if os.path.isdir(os.path.join(experiment_results_dir, n))
        and not n.startswith(".")
        and not n.endswith("_retrieved")
    )


def find_tier_csv(model_dir: str, tier: int) -> str | None:
    patterns = [
        os.path.join(model_dir, f"*tier{tier}_ff*.csv"),
        os.path.join(model_dir, f"*tier{tier}*ff*.csv"),
        os.path.join(model_dir, f"*tier{tier}*.csv"),
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


def detect_no_prob_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if "prob" in cl and "no" in cl and "yes" not in cl:
            return c
    return None


def summarize_no_probabilities(prob_no: np.ndarray, eps_log: float = 1e-12) -> dict:
    p = np.asarray(prob_no, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {
            "n": 0,
            "mean_prob_no": np.nan,
            "median_prob_no": np.nan,
            "sd_prob_no": np.nan,
            "min_prob_no": np.nan,
            "max_prob_no": np.nan,
            "pct_exactly_zero": np.nan,
            "mean_log_prob_no_clipped": np.nan,
            "median_log_prob_no_clipped": np.nan,
            "eps_log_clip": eps_log,
        }
    p_clip = np.clip(p, eps_log, 1.0)
    log_p = np.log(p_clip)
    return {
        "n": int(len(p)),
        "mean_prob_no": float(np.mean(p)),
        "median_prob_no": float(np.median(p)),
        "sd_prob_no": float(np.std(p, ddof=0)),
        "min_prob_no": float(np.min(p)),
        "max_prob_no": float(np.max(p)),
        "pct_exactly_zero": float(np.mean(p <= 0.0) * 100.0),
        "mean_log_prob_no_clipped": float(np.mean(log_p)),
        "median_log_prob_no_clipped": float(np.median(log_p)),
        "eps_log_clip": eps_log,
    }


def score_model_tier1_no_prob(
    model_id: str,
    tier1_path: str,
    gpt3_ref: float,
) -> dict | None:
    df = pd.read_csv(tier1_path)
    col = detect_no_prob_column(df)
    if col is None:
        return None
    stats = summarize_no_probabilities(df[col].astype(float).values)
    stats.update(
        {
            "model_id": model_id,
            "model_label": MODEL_LABELS.get(model_id, model_id.replace("_", " ")),
            "tier1_csv": tier1_path,
            "no_prob_column": col,
            "mean_prob_no_pct": 100.0 * stats["mean_prob_no"],
            "gpt3_reference_mean_prob_no": gpt3_ref,
            "gpt3_reference_mean_prob_no_pct": 100.0 * gpt3_ref,
            "gpt3_reference_citation": "Logé et al., Q-Pain (GPT-3 post-operative, Fig. 2a)",
        }
    )
    return stats


def plot_all_models_bar(summary: pd.DataFrame, gpt3_ref: float, out_path: str) -> None:
    """Bar chart: mean P(No) per closed-book model + GPT-3 reference."""
    df = summary.sort_values("mean_prob_no", ascending=False).copy()
    labels = [f"Q-Pain\n(GPT-3)"] + df["model_label"].tolist()
    values = [gpt3_ref] + df["mean_prob_no"].tolist()
    colors = ["#4C72B0"] + ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"][: len(df)]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(labels)), 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(r"Mean $P(\mathrm{No})$", fontsize=12)
    ax.set_yscale("log")
    ymin = max(1e-8, min(v for v in values if v > 0) * 0.4)
    ax.set_ylim(bottom=ymin, top=1.0)
    ax.axhline(gpt3_ref, color="#4C72B0", linestyle="--", alpha=0.35, linewidth=1)
    for xi, v in zip(x, values):
        ax.text(
            xi,
            v,
            f"{v:.3g}" if v >= 0.001 else f"{v:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Mean deny-treatment probability (Tier 1, closed-book)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gpt4o_vs_gpt3(prob_no: np.ndarray, gpt3_ref: float, out_path: str) -> None:
    """Legacy two-bar figure (GPT-4o-mini vs GPT-3)."""
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300})
    mean_obs = float(np.mean(prob_no))
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2))
    labels = ["Q-Pain\n(GPT-3)", "GPT-4o-mini\n(Tier 1)"]
    values = [gpt3_ref, mean_obs]
    colors = ["#4C72B0", "#DD8452"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_ylabel(r"Mean $P(\mathrm{No})$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-8, min(values) * 0.5), top=1.0)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.3g}" if v >= 0.001 else f"{v:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Mean deny-treatment probability", fontweight="bold")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 1 P(No) for all closed-book models + comparison to GPT-3."
    )
    parser.add_argument(
        "--gpt3_ref",
        type=float,
        default=QPAIN_GPT3_POSTOP_MEAN_NO_PROB,
        help="Literature mean P(No) for GPT-3 post-operative (default: 0.256).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: analysis_results/tier1_no_prob).",
    )
    parser.add_argument(
        "--tier1_csv",
        default=None,
        help="Optional: score only this Tier 1 CSV (single-model mode).",
    )
    args = parser.parse_args()

    root = _project_root()
    out_dir = args.out_dir or os.path.join(root, "analysis_results", "tier1_no_prob")
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    skipped: list[str] = []

    if args.tier1_csv:
        model_id = os.path.basename(os.path.dirname(args.tier1_csv))
        row = score_model_tier1_no_prob(model_id, args.tier1_csv, args.gpt3_ref)
        if row is None:
            print("ERROR: No P(No) column in CSV.", file=sys.stderr)
            sys.exit(1)
        rows.append(row)
    else:
        exp_dir = os.path.join(root, "experiment_results")
        for model_id in discover_closed_book_models(exp_dir):
            model_dir = os.path.join(exp_dir, model_id)
            tier1_path = find_tier_csv(model_dir, 1)
            if not tier1_path:
                skipped.append(f"{model_id}: no Tier 1 CSV")
                continue
            row = score_model_tier1_no_prob(model_id, tier1_path, args.gpt3_ref)
            if row is None:
                skipped.append(f"{model_id}: no log-probability P(No) column")
                continue
            rows.append(row)
            print(
                f"{model_id}: mean P(No) = {row['mean_prob_no']:.4f} "
                f"({row['mean_prob_no_pct']:.2f}%), n={row['n']}"
            )

    if not rows:
        print("ERROR: No models scored.", file=sys.stderr)
        if skipped:
            print("Skipped:", file=sys.stderr)
            for s in skipped:
                print(f"  - {s}", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows).sort_values("mean_prob_no", ascending=False)
    summary_path = os.path.join(out_dir, "tier1_no_probability_summary_all_models_ff.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")

    if skipped:
        print("\nSkipped models:")
        for s in skipped:
            print(f"  - {s}")

    plot_all_models_bar(
        summary,
        args.gpt3_ref,
        os.path.join(out_dir, "tier1_no_probability_all_models_bar_ff.png"),
    )
    print(f"Wrote {os.path.join(out_dir, 'tier1_no_probability_all_models_bar_ff.png')}")

    gpt4o = summary[summary["model_id"] == "gpt4o_mini"]
    if len(gpt4o):
        tier1_path = gpt4o.iloc[0]["tier1_csv"]
        df = pd.read_csv(tier1_path)
        col = gpt4o.iloc[0]["no_prob_column"]
        plot_gpt4o_vs_gpt3(
            df[col].astype(float).values,
            args.gpt3_ref,
            os.path.join(out_dir, "figure1_no_probability_comparison.png"),
        )
        gpt4o.to_csv(os.path.join(out_dir, "gpt4o_mini_tier1_no_probability_summary.csv"), index=False)
        print(f"Wrote legacy figure1_no_probability_comparison.png")


if __name__ == "__main__":
    main()
