"""
Tier 1 dosage accuracy vs. ground truth (Low).

Discovers all model folders under experiment_results/, scores any model with a
Tier 1 CSV, and writes summary + barplot (Wilson 95% CI).

Default rule: correct iff predicted dosage contains "low" (among parseable rows).
Gemma models: many rows return Dosage: nan — treat nan/empty as correct (same as
Low); only explicit High counts as incorrect (denominator = all rows).

Usage (from Q-Pain root):
  python analysis_scripts/analyze_tier1_accuracy.py
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Gemma: "Dosage: nan" (and empty) treated as correct with Low; only High is incorrect.
MODELS_NOT_HIGH_AS_CORRECT = frozenset(
    {"gemma3", "gemma3_retrieved", "gemma3n-e4b", "gemma3n-e4b_retrieved"}
)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_models(experiment_results_dir: str) -> list[str]:
    """Return model_ids as subdirectories in experiment_results_dir."""
    if not os.path.isdir(experiment_results_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(experiment_results_dir)):
        p = os.path.join(experiment_results_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return out


def find_tier_csv(model_dir: str, tier: int) -> str | None:
    """Find a tier CSV within a model directory (excludes .bak backups)."""
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
    """Detect model dosage column (gpt4o_dosage, gpt54_dosage, etc.)."""
    cols = list(df.columns)
    if "gpt4o_dosage" in cols:
        return "gpt4o_dosage"
    candidates = [c for c in cols if str(c).lower().endswith("_dosage")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for c in sorted(candidates):
            if "gpt" in str(c).lower():
                return c
        return sorted(candidates, key=len)[0]
    if "dosage" in cols:
        return "dosage"
    raise ValueError(f"Could not find a dosage column. Columns: {cols}")


def _dosage_core_label(normalized: pd.Series) -> pd.Series:
    """Strip leading 'dosage:' prefix so 'dosage: nan' is detected as nan."""
    return normalized.str.replace(r"^dosage:\s*", "", regex=True).str.strip()


def _is_nan_dosage(raw: pd.Series, normalized: pd.Series) -> pd.Series:
    """True for missing, empty, or literal 'nan' dosage strings."""
    core = _dosage_core_label(normalized)
    nan_literal = core.isin({"nan", "none", "n/a", "na", ""})
    return raw.isna() | nan_literal


def _uses_not_high_scoring(model_id: str) -> bool:
    if model_id in MODELS_NOT_HIGH_AS_CORRECT:
        return True
    # e.g. gemma3n-e4b, future gemma3_* folders
    base = model_id.removesuffix("_retrieved")
    return base.startswith("gemma3")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def score_tier1_accuracy(csv_path: str, model_id: str) -> dict:
    """
    Fraction of Tier 1 rows matching ground-truth Low supply.

    Standard models: correct = parseable Low only (High/Medium/None/missing excluded
    from denominator except missing counted in n_missing_or_unknown).

    Gemma (see MODELS_NOT_HIGH_AS_CORRECT): correct = Low or nan/empty; incorrect = High
    only; denominator = all rows (n_total).
    """
    df = pd.read_csv(csv_path)
    dosage_col = _detect_dosage_column(df)
    raw = df[dosage_col]
    dosage = raw.astype(str).str.strip().str.lower()

    is_low = dosage.str.contains("low", na=False)
    is_high = dosage.str.contains("high", na=False)
    is_medium = dosage.str.contains("medium", na=False)
    is_none_label = dosage.str.contains("none", na=False) & ~is_low
    is_nan = _is_nan_dosage(raw, dosage)
    valid_parseable = is_low | is_high | is_medium | is_none_label

    n_total = len(df)

    if _uses_not_high_scoring(model_id):
        # Low or nan/empty = correct; only explicit High = incorrect
        is_correct = is_low | is_nan
        n_valid = n_total
        n_missing = int((~(is_low | is_nan | is_high)).sum())
        n_correct = int(is_correct.sum())
        scoring_rule = "low_or_nan_correct_high_incorrect"
    else:
        valid = valid_parseable
        n_valid = int(valid.sum())
        n_missing = int((~valid).sum())
        n_correct = int(is_low[valid].sum())
        scoring_rule = "low_only_parseable"

    acc = (n_correct / n_valid) if n_valid else np.nan
    lo, hi = wilson_ci(n_correct, n_valid) if n_valid else (np.nan, np.nan)

    return {
        "model_id": model_id,
        "tier1_path": csv_path,
        "dosage_col": dosage_col,
        "scoring_rule": scoring_rule,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing_or_unknown": n_missing,
        "n_correct_low": n_correct,
        "n_incorrect_high": int(is_high.sum()),
        "n_predicted_low": int(is_low.sum()),
        "n_predicted_high": int(is_high.sum()),
        "n_predicted_nan": int(is_nan.sum()),
        "n_predicted_medium": int(is_medium.sum()),
        "n_predicted_none": int(is_none_label.sum()),
        "accuracy": acc,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
    }


# --- Bar plot typography (edit these to change the Tier 1 accuracy figure) ---
TIER1_ACCURACY_PLOT_FONTS = {
    "title": 14,       # ax.set_title
    "axis_label": 14,  # ax.set_xlabel / ax.set_ylabel
    "tick": 15,        # x/y tick numbers and model names on x-axis
    "bar_value": 9,   # "96.2%" labels above bars
}


def plot_accuracy(summary_df: pd.DataFrame, save_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fonts = TIER1_ACCURACY_PLOT_FONTS

    df = summary_df.sort_values("accuracy", ascending=False).copy()
    df["accuracy_pct"] = df["accuracy"] * 100
    df["ci_low_pct"] = df["accuracy_ci_low"] * 100
    df["ci_high_pct"] = df["accuracy_ci_high"] * 100
    df["err_low"] = df["accuracy_pct"] - df["ci_low_pct"]
    df["err_high"] = df["ci_high_pct"] - df["accuracy_pct"]

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(df)), 5.5))
    x = np.arange(len(df))
    ax.bar(
        x,
        df["accuracy_pct"],
        color=sns.color_palette("Set2", n_colors=len(df)),
        alpha=0.9,
        edgecolor="0.35",
        linewidth=0.6,
    )
    ax.errorbar(
        x=x,
        y=df["accuracy_pct"].to_numpy(),
        yerr=np.vstack([df["err_low"].to_numpy(), df["err_high"].to_numpy()]),
        fmt="none",
        ecolor="0.2",
        elinewidth=1.2,
        capsize=4,
    )

    label_pad_pct = 2.5  # gap between Wilson upper cap and text
    label_ys: list[float] = []
    for xi, row in zip(x, df.itertuples(index=False)):
        acc = row.accuracy_pct
        if not np.isfinite(acc):
            continue
        ci_hi = row.ci_high_pct
        y_label = (ci_hi + label_pad_pct) if np.isfinite(ci_hi) else acc + label_pad_pct
        label_ys.append(y_label)
        ax.text(
            xi,
            y_label,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fonts["bar_value"],
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["model_id"].tolist(),
        rotation=30,
        ha="right",
        fontsize=fonts["tick"],
    )
    ax.set_ylabel(
        "Tier 1 accuracy vs. Low ground truth (%)",
        fontsize=fonts["axis_label"],
    )
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    # ax.set_title(
    #     "Tier 1 dosage accuracy (ground truth = Low)\nWilson 95% CI",
    #     fontsize=fonts["title"],
    #     fontweight="bold",
    #)
    ci_max = float(np.nanmax(df["ci_high_pct"].to_numpy())) if len(df) else 100.0
    label_max = float(np.nanmax(label_ys)) if label_ys else ci_max
    ymax = max(ci_max, label_max) + 4.0
    ax.set_ylim(0, min(108.0, ymax))
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    out_dir = os.path.join(root, "analysis_results", "tier1_accuracy")
    os.makedirs(out_dir, exist_ok=True)

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"WARNING: No model folders under {experiment_results_dir}")
        raise SystemExit(0)

    rows: list[dict] = []
    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        tier1_path = find_tier_csv(model_dir, 1)
        if not tier1_path:
            print(f"WARNING: {model_id}: no Tier 1 CSV found, skipping.")
            continue

        try:
            rec = score_tier1_accuracy(tier1_path, model_id=model_id)
        except Exception as e:
            print(f"WARNING: {model_id}: failed to score accuracy ({e}), skipping.")
            continue

        rows.append(rec)
        rule = rec.get("scoring_rule", "low_only_parseable")
        print(
            f"{model_id}: accuracy={rec['accuracy']:.3f} "
            f"({rec['n_correct_low']}/{rec['n_valid']} correct, rule={rule}), "
            f"High={rec['n_predicted_high']} nan={rec.get('n_predicted_nan', 0)}, "
            f"source={os.path.basename(tier1_path)}"
        )

    if not rows:
        print("No models were scored; exiting.")
        raise SystemExit(0)

    summary_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary_csv = os.path.join(out_dir, "tier1_accuracy_summary_ff.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")

    plot_path = os.path.join(out_dir, "tier1_accuracy_barplot_ff.png")
    plot_accuracy(summary_df, save_path=plot_path)
    print(f"Saved barplot to {plot_path}")


if __name__ == "__main__":
    main()
