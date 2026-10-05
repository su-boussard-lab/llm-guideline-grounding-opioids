"""
Visualize Tier 2 → Tier 3 impact of adding a Medium dosage option.

Primary metric (per model): among rows where the chosen dosage changes from Tier 2
(binary Low/High) to Tier 3 (Low/Medium/High/None), what fraction select Medium in
Tier 3? (e.g. GPT-4o-mini closed-book ≈ 92% of changes go to Medium.)

Secondary metrics: overall Tier 3 Medium rate; overall decision-change rate;
breakdown of Tier 3 choices among changed rows.

Models without both Tier 2 and Tier 3 CSVs are skipped. Works for closed-book and
_retrieved folders under experiment_results/.

Usage (from Q-Pain root):
  python analysis_scripts/visualize_medium_dosage_impact.py

Outputs (analysis_results/medium_dosage_impact/):
  - medium_dosage_impact_summary_ff.csv
  - medium_dosage_impact_barplot_ff.png
  - medium_dosage_change_destination_ff.png  (stacked bar among changes)
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Plot typography (edit sizes here, then re-run this script) ---
MEDIUM_DOSAGE_PLOT_FONTS = {
    "suptitle": 14,       # fig.suptitle (two-panel bar chart only)
    "panel_title": 12,    # ax.set_title on each subplot
    "axis_label": 14,     # ax.set_ylabel / ax.set_xlabel
    "tick": 15,           # x/y tick labels (model names, axis numbers)
    "bar_value": 9,       # "%" labels above bars (main bar chart)
    "legend": 11,         # fig.legend / ax.legend
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_models(experiment_results_dir: str) -> list[str]:
    if not os.path.isdir(experiment_results_dir):
        return []
    return sorted(
        n
        for n in os.listdir(experiment_results_dir)
        if os.path.isdir(os.path.join(experiment_results_dir, n)) and not n.startswith(".")
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


def _detect_dosage_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    if "gpt4o_dosage" in cols:
        return "gpt4o_dosage"
    candidates = [c for c in cols if str(c).lower().endswith("_dosage")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=len)[0]
    if "dosage" in cols:
        return "dosage"
    raise ValueError(f"No dosage column in: {cols}")


def _canonical_dosage(raw: pd.Series) -> pd.Series:
    d = raw.fillna("").astype(str).str.lower().str.replace(".", "", regex=False).str.strip()
    out = pd.Series("Unknown", index=d.index, dtype=object)
    out[d.str.contains("none", na=False)] = "None"
    out[d.str.contains("medium", na=False)] = "Medium"
    out[d.str.contains("low", na=False) & ~d.str.contains("medium", na=False)] = "Low"
    out[d.str.contains("high", na=False)] = "High"
    return out


def _match_key_full(df: pd.DataFrame) -> pd.Series:
    return (
        df["vignette_idx"].astype(str)
        + "_"
        + df["race"].astype(str)
        + "_"
        + df["gender"].astype(str)
        + "_"
        + df["risk_op"].astype(str)
        + "_"
        + df["risk_mh"].astype(str)
        + "_"
        + df["risk_pain"].astype(str)
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_medium_impact(model_id: str, tier2_path: str, tier3_path: str) -> dict | None:
    df2 = pd.read_csv(tier2_path)
    df3 = pd.read_csv(tier3_path)

    col2 = _detect_dosage_column(df2)
    col3 = _detect_dosage_column(df3)

    t2 = pd.DataFrame(
        {
            "match_key": _match_key_full(df2),
            "tier2_chosen": _canonical_dosage(df2[col2]),
        }
    )
    t3 = pd.DataFrame(
        {
            "match_key": _match_key_full(df3),
            "tier3_chosen": _canonical_dosage(df3[col3]),
        }
    )

    merged = t2.merge(t3, on="match_key", how="inner")
    if merged.empty:
        return None

    merged["decision_changed"] = merged["tier2_chosen"] != merged["tier3_chosen"]
    changed = merged[merged["decision_changed"]].copy()

    n_matched = len(merged)
    n_changed = len(changed)
    n_to_medium = int((changed["tier3_chosen"] == "Medium").sum()) if n_changed else 0
    n_tier3_medium = int((merged["tier3_chosen"] == "Medium").sum())

    pct_changed = 100.0 * n_changed / n_matched if n_matched else np.nan
    pct_to_medium_among_changed = (
        100.0 * n_to_medium / n_changed if n_changed else np.nan
    )
    pct_tier3_medium = 100.0 * n_tier3_medium / n_matched if n_matched else np.nan

    ci_lo, ci_hi = wilson_ci(n_to_medium, n_changed)

    dest: dict[str, int] = {}
    if n_changed:
        for label, cnt in changed["tier3_chosen"].value_counts().items():
            dest[str(label)] = int(cnt)

    return {
        "model_id": model_id,
        "tier2_path": tier2_path,
        "tier3_path": tier3_path,
        "dosage_col_tier2": col2,
        "dosage_col_tier3": col3,
        "n_matched": n_matched,
        "n_decision_changed": n_changed,
        "n_changed_to_medium": n_to_medium,
        "pct_decision_changed": pct_changed,
        "pct_changed_to_medium": pct_to_medium_among_changed,
        "pct_changed_to_medium_ci_low": 100.0 * ci_lo if n_changed else np.nan,
        "pct_changed_to_medium_ci_high": 100.0 * ci_hi if n_changed else np.nan,
        "pct_tier3_medium_overall": pct_tier3_medium,
        "dest_medium": dest.get("Medium", 0),
        "dest_low": dest.get("Low", 0),
        "dest_high": dest.get("High", 0),
        "dest_none": dest.get("None", 0),
        "dest_unknown": dest.get("Unknown", 0),
        "is_retrieved": model_id.endswith("_retrieved"),
    }


def _label_y_above_ci(acc_pct: float, ci_hi_pct: float, pad: float = 2.5) -> float:
    y = (ci_hi_pct + pad) if np.isfinite(ci_hi_pct) else acc_pct + pad
    return y


def plot_main_bars(summary: pd.DataFrame, save_path: str) -> None:
    """Two-panel bar chart: % changes → Medium; overall Tier 3 Medium rate."""
    sns.set_theme(style="whitegrid")
    fonts = MEDIUM_DOSAGE_PLOT_FONTS
    df = summary.sort_values("pct_changed_to_medium", ascending=False).copy()
    df["variant"] = np.where(df["is_retrieved"], "Retrieved", "Closed-book")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    x = np.arange(len(df))
    palette = {"Closed-book": "#4C78A8", "Retrieved": "#F58518"}
    colors = [palette[v] for v in df["variant"]]

    metrics = [
        ("pct_changed_to_medium", "pct_changed_to_medium_ci_low", "pct_changed_to_medium_ci_high",
         "Among Tier 2→3 changes,\n% selecting Medium"),
        ("pct_tier3_medium_overall", None, None,
         "All matched rows,\n% selecting Medium (Tier 3)"),
    ]

    for ax, (col, ci_lo, ci_hi, title) in zip(axes, metrics):
        vals = df[col].to_numpy()
        ax.bar(x, vals, color=colors, alpha=0.9, edgecolor="0.35", linewidth=0.6)
        if ci_lo and ci_hi:
            err_lo = vals - df[ci_lo].to_numpy()
            err_hi = df[ci_hi].to_numpy() - vals
            ax.errorbar(
                x, vals,
                yerr=np.vstack([err_lo, err_hi]),
                fmt="none", ecolor="0.2", elinewidth=1.1, capsize=3,
            )
        label_pad = 2.5
        label_ys = []
        for xi, row in zip(x, df.itertuples(index=False)):
            v = getattr(row, col)
            if not np.isfinite(v):
                continue
            if ci_hi:
                y = _label_y_above_ci(v, getattr(row, ci_hi), label_pad)
            else:
                y = v + label_pad
            label_ys.append(y)
            ax.text(
                xi, y, f"{v:.1f}%",
                ha="center", va="bottom",
                fontsize=fonts["bar_value"],
                clip_on=False,
            )
        ax.set_title(title, fontsize=fonts["panel_title"], fontweight="bold")
        ax.set_ylabel("%", fontsize=fonts["axis_label"])
        ax.tick_params(axis="y", labelsize=fonts["tick"])
        ymax = max(float(np.nanmax(label_ys)) + 4 if label_ys else 100, float(np.nanmax(vals)) + 12)
        ax.set_ylim(0, min(108.0, ymax))
        ax.axhline(50, color="0.75", linestyle=":", linewidth=0.8, zorder=0)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        df["model_id"], rotation=35, ha="right", fontsize=fonts["tick"],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette]
    fig.legend(
        handles,
        palette.keys(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=fonts["legend"],
    )
    fig.suptitle(
        "Medium dosage impact: Tier 2 (binary) → Tier 3 (+ Medium option)",
        fontsize=fonts["suptitle"],
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_change_destinations(summary: pd.DataFrame, save_path: str) -> None:
    """Stacked bar: where Tier 3 choices go among Tier 2→3 changes only."""
    sns.set_theme(style="whitegrid")
    fonts = MEDIUM_DOSAGE_PLOT_FONTS
    df = summary.sort_values("pct_changed_to_medium", ascending=False).copy()
    labels = ["Medium", "Low", "High", "None", "Unknown"]
    cols = ["dest_medium", "dest_low", "dest_high", "dest_none", "dest_unknown"]
    colors = ["#59A14F", "#4C78A8", "#E15759", "#B07AA1", "#9D9D9D"]

    n_changed = df["n_decision_changed"].to_numpy(dtype=float)
    pct = np.zeros((len(df), len(labels)))
    for j, c in enumerate(cols):
        pct[:, j] = np.where(
            n_changed > 0,
            100.0 * df[c].to_numpy() / n_changed,
            np.nan,
        )

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df)), 5.5))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for j, (lab, col) in enumerate(zip(labels, cols)):
        ax.bar(x, pct[:, j], bottom=bottom, label=lab, color=colors[j], edgecolor="white", linewidth=0.4)
        bottom += np.nan_to_num(pct[:, j], nan=0.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["model_id"], rotation=35, ha="right", fontsize=fonts["tick"],
    )
    ax.set_ylabel("% of Tier 2→3 changes", fontsize=fonts["axis_label"])
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    ax.set_ylim(0, 100)
    ax.set_title(
        "Destination of dosage changes (Tier 2 → Tier 3)\n"
        "Denominator = rows with different chosen dosage",
        fontsize=fonts["panel_title"],
        fontweight="bold",
    )
    ax.legend(
        loc="upper right",
        framealpha=0.95,
        fontsize=fonts["legend"],
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = _project_root()
    exp_dir = os.path.join(root, "experiment_results")
    out_dir = os.path.join(root, "analysis_results", "medium_dosage_impact")
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    for model_id in discover_models(exp_dir):
        model_dir = os.path.join(exp_dir, model_id)
        tier2_path = find_tier_csv(model_dir, 2)
        tier3_path = find_tier_csv(model_dir, 3)
        if not (tier2_path and tier3_path):
            continue
        result = score_medium_impact(model_id, tier2_path, tier3_path)
        if result is None:
            print(f"WARNING: {model_id}: no matched Tier 2/3 rows — skipped")
            continue
        rows.append(result)
        print(
            f"{model_id}: {result['pct_changed_to_medium']:.1f}% of changes → Medium "
            f"({result['n_changed_to_medium']}/{result['n_decision_changed']}), "
            f"Tier 3 Medium overall {result['pct_tier3_medium_overall']:.1f}%"
        )

    if not rows:
        print("No models with Tier 2 and Tier 3 data found.")
        raise SystemExit(0)

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "medium_dosage_impact_summary_ff.csv")
    summary.to_csv(csv_path, index=False)

    plot_main_bars(summary, os.path.join(out_dir, "medium_dosage_impact_barplot_ff.png"))
    plot_change_destinations(
        summary, os.path.join(out_dir, "medium_dosage_change_destination_ff.png")
    )

    print(f"\nSaved summary: {csv_path}")
    print(f"Saved barplot: {os.path.join(out_dir, 'medium_dosage_impact_barplot_ff.png')}")
    print(f"Saved stacked: {os.path.join(out_dir, 'medium_dosage_change_destination_ff.png')}")


if __name__ == "__main__":
    main()
