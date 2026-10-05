from __future__ import annotations

import os
import re
import glob
from typing import Iterable, Tuple, Dict, Any, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Tier 1 ROUGE-L bar plot typography (edit sizes, then re-run script) ---
ROUGE_L_PLOT_FONTS = {
    "title": 14,        # ax.set_title
    "axis_label": 14,   # ax.set_ylabel
    "tick": 15,         # x model names + y-axis numbers
    "bar_value": 9,    # "0.501" labels above bars
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_models(experiment_results_dir: str) -> list[str]:
    if not os.path.isdir(experiment_results_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(experiment_results_dir)):
        p = os.path.join(experiment_results_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return out


def find_tier_csv(model_dir: str, tier: int) -> str | None:
    patterns = [
        os.path.join(model_dir, f"*tier{tier}_ff*.csv"),
        os.path.join(model_dir, f"*tier{tier}*ff*.csv"),
        os.path.join(model_dir, f"*tier{tier}*.csv"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def _detect_explanation_col(df: pd.DataFrame) -> str:
    """
    Detect a model explanation column in a tier-1 results CSV.

    Common patterns:
      - gpt4o_explanation / llama_explanation / *_explanation
      - explanation (rare)
    """
    cols = list(df.columns)
    # Prefer explicit *_explanation columns
    expl = [c for c in cols if c.lower().endswith("_explanation")]
    if len(expl) == 1:
        return expl[0]
    if len(expl) > 1:
        # Prefer gpt4o_explanation for backward-compatibility, else shortest name
        for c in expl:
            if c.lower() == "gpt4o_explanation":
                return c
        return sorted(expl, key=len)[0]
    # Fall back to any column containing "explanation"
    contains = [c for c in cols if "explanation" in c.lower()]
    if contains:
        return sorted(contains, key=len)[0]
    raise ValueError(f"Could not find an explanation column. Columns: {cols}")


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# First names used in Tier-1 Q-Pain post-op experiments (same pool as the experiment notebooks).
# Used only to align model explanations with ground-truth wording (Patient D + they/their) before ROUGE-L.
_DEFAULT_TIER1_FIRST_NAMES: frozenset[str] = frozenset(
    {
        "Roosevelt",
        "Jermaine",
        "Darnell",
        "Willie",
        "Mattie",
        "Reginald",
        "Cedric",
        "Sylvester",
        "Tyrone",
        "Errol",
        "Latonya",
        "Tamika",
        "Ebony",
        "Latasha",
        "Keisha",
        "Lillie",
        "Minnie",
        "Gwendolyn",
        "Bessie",
        "Marva",
        "Bradley",
        "Brett",
        "Scott",
        "Kurt",
        "Todd",
        "Chad",
        "Matthew",
        "Dustin",
        "Shane",
        "Douglas",
        "Beth",
        "Megan",
        "Kristin",
        "Jill",
        "Erin",
        "Colleen",
        "Kathleen",
        "Heather",
        "Holly",
        "Laurie",
        "Viet",
        "Thong",
        "Qiang",
        "Kwok",
        "Hao",
        "Yang",
        "Nam",
        "Huy",
        "Yuan",
        "Ho",
        "Zhen",
        "Nga",
        "Lien",
        "Lam",
        "Hui",
        "Wing",
        "Hoa",
        "Wai",
        "Min",
        "Huong",
        "Rigoberto",
        "Santos",
        "Javier",
        "Efrain",
        "Juan",
        "Ramiro",
        "Jesus",
        "Humberto",
        "Gonzalo",
        "Hector",
        "Guadalupe",
        "Marisela",
        "Guillermina",
        "Rocio",
        "Yesenia",
        "Blanca",
        "Rosalba",
        "Elvia",
        "Alejandra",
        "Mayra",
    }
)

# Few-shot exemplar labels in closed prompts; model sometimes echoes them in explanations.
_PATIENT_EXEMPLAR_RE = re.compile(r"\bPatient\s+[A-C]\b", re.IGNORECASE)


def load_extra_first_names_from_file(path: str) -> list[str]:
    """One first name per line; # starts a comment; blank lines ignored."""
    if not os.path.isfile(path):
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def standardize_model_explanation_for_rouge(
    text: str | float | None,
    *,
    extra_first_names: Sequence[str] | None = None,
) -> str:
    """
    Map model explanations toward the same surface form as `data_post_op` ground-truth explanations:
    concrete first names -> Patient D; gendered pronouns -> they/their/them/themselves.

    ROUGE-L is token-LCS; mismatched names and he/she vs they reduce overlap even when clinical
    meaning matches. This does not change the reference (CSV Explanation column).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):  # type: ignore[truthy-bool]
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # Few-shot patient labels -> Patient D
    s = _PATIENT_EXEMPLAR_RE.sub("Patient D", s)

    names = set(_DEFAULT_TIER1_FIRST_NAMES)
    if extra_first_names:
        for n in extra_first_names:
            n = str(n).strip()
            if n:
                names.add(n)

    # Longest first so e.g. "Guillermina" before "Min" (substring safety).
    for name in sorted(names, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(name)}(?:'s|’s)?\b", re.IGNORECASE)
        s = pat.sub("Patient D", s)

    # Pronouns: longer phrases first (word boundaries; case-insensitive).
    pronoun_pairs: list[tuple[str, str]] = [
        (r"\bherself\b", "themselves"),
        (r"\bhimself\b", "themselves"),
        (r"\bhis\b", "their"),
        (r"\bher\b", "their"),
        (r"\bhim\b", "them"),
        (r"\bshe\b", "they"),
        (r"\bhe\b", "they"),
    ]
    for pat, repl in pronoun_pairs:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # Subject–verb agreement after he/she -> they (ground truth uses plural: "they have", etc.)
    agreement_fixes: list[tuple[str, str]] = [
        (r"\bthey has\b", "they have"),
        (r"\bthey is\b", "they are"),
        (r"\bthey was\b", "they were"),
        (r"\bthey does\b", "they do"),
        (r"\bthey needs\b", "they need"),
        (r"\bthey continues\b", "they continue"),
        (r"\bthey remains\b", "they remain"),
        (r"\bthey seems\b", "they seem"),
    ]
    for pat, repl in agreement_fixes:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    return s


def _tokenize(text: str) -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):  # type: ignore[truthy-bool]
        return []
    s = str(text).lower()
    return _TOKEN_RE.findall(s)


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of LCS for token lists (O(len(a)*len(b)))."""
    if not a or not b:
        return 0
    # Ensure b is shorter to reduce memory if possible
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    """
    ROUGE-L F1 using token-level LCS:
      P = LCS / len(candidate_tokens)
      R = LCS / len(reference_tokens)
      F1 = 2PR/(P+R)
    """
    ref_toks = _tokenize(reference)
    cand_toks = _tokenize(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    lcs = _lcs_len(ref_toks, cand_toks)
    p = lcs / len(cand_toks) if cand_toks else 0.0
    r = lcs / len(ref_toks) if ref_toks else 0.0
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def _bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=x.size, replace=True)
        means.append(float(np.mean(samp)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def load_ground_truth(gt_path: str) -> pd.DataFrame:
    """
    Load ground-truth explanations. If vignette_idx is absent, create it from row order (0..N-1).
    """
    gt = pd.read_csv(gt_path)
    if "Explanation" not in gt.columns:
        raise ValueError(f"Ground-truth file must include 'Explanation' column. Found: {sorted(gt.columns)}")
    if "vignette_idx" in gt.columns:
        gt["vignette_idx"] = gt["vignette_idx"].astype(int)
    else:
        gt = gt.reset_index(drop=True)
        gt["vignette_idx"] = gt.index.astype(int)
    gt = gt[["vignette_idx", "Explanation"]].copy()
    gt = gt.rename(columns={"Explanation": "gt_explanation"})
    return gt


def score_model_rouge_l(
    tier1_path: str,
    gt: pd.DataFrame,
    model_id: str,
    *,
    standardize_model_text: bool = True,
    extra_first_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(tier1_path)
    if "vignette_idx" not in df.columns:
        raise ValueError(f"{model_id}: tier1 file missing 'vignette_idx' column: {tier1_path}")
    if "race" not in df.columns or "gender" not in df.columns:
        raise ValueError(f"{model_id}: tier1 file missing race/gender columns: {tier1_path}")

    expl_col = _detect_explanation_col(df)
    d = df[["vignette_idx", "race", "gender", expl_col]].copy()
    d = d.rename(columns={expl_col: "model_explanation"})
    d["vignette_idx"] = d["vignette_idx"].astype(int)

    merged = d.merge(gt, on="vignette_idx", how="left")
    if merged["gt_explanation"].isna().any():
        missing = merged.loc[merged["gt_explanation"].isna(), "vignette_idx"].unique().tolist()
        raise ValueError(
            f"{model_id}: missing ground-truth explanation for vignette_idx values: {missing}. "
            "Check that ground-truth rows align with vignette_idx."
        )

    if standardize_model_text:
        merged["model_explanation_std"] = merged["model_explanation"].apply(
            lambda x: standardize_model_explanation_for_rouge(x, extra_first_names=extra_first_names)
        )
        cand_col = "model_explanation_std"
    else:
        merged["model_explanation_std"] = merged["model_explanation"].astype(str)
        cand_col = "model_explanation"

    merged["rouge_l_f1"] = [
        rouge_l_f1(ref, cand)
        for ref, cand in zip(merged["gt_explanation"], merged[cand_col])
    ]
    merged["model_id"] = model_id
    merged["tier1_path"] = tier1_path
    return merged[
        [
            "model_id",
            "vignette_idx",
            "race",
            "gender",
            "rouge_l_f1",
            "model_explanation_std",
            "tier1_path",
        ]
    ]


def plot_model_means(summary: pd.DataFrame, save_path: str):
    sns.set_theme(style="whitegrid")
    fonts = ROUGE_L_PLOT_FONTS
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    s = summary.sort_values("mean_rouge_l_f1", ascending=False).copy()
    x = np.arange(len(s))
    ax.bar(x, s["mean_rouge_l_f1"], color=sns.color_palette("Set2", len(s)), alpha=0.9, edgecolor="0.35", linewidth=0.6)

    # error bars from bootstrap CI
    y = s["mean_rouge_l_f1"].to_numpy(dtype=float)
    lo = s["ci_low"].to_numpy(dtype=float)
    hi = s["ci_high"].to_numpy(dtype=float)
    err_lo = np.maximum(0.0, y - lo)
    err_hi = np.maximum(0.0, hi - y)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="0.2", elinewidth=1.2, capsize=4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        s["model_id"].tolist(),
        rotation=30,
        ha="right",
        fontsize=fonts["tick"],
    )
    ax.set_ylabel(
        "Mean ROUGE-L F1",
        fontsize=fonts["axis_label"],
    )
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    # ax.set_title(
    #     "Explanation similarity to ground truth (ROUGE-L, Tier 1)",
    #     fontsize=fonts["title"],
    #     fontweight="bold",
    # )

    label_pad = 0.012
    label_ys: list[float] = []
    for xi, yi, ci_hi in zip(x, y, hi):
        if not np.isfinite(yi):
            continue
        y_label = (ci_hi + label_pad) if np.isfinite(ci_hi) else yi + label_pad
        label_ys.append(y_label)
        ax.text(
            xi,
            y_label,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=fonts["bar_value"],
            clip_on=False,
        )

    ci_max = float(np.nanmax(hi)) if np.isfinite(hi).any() else float(np.nanmax(y))
    label_max = float(np.nanmax(label_ys)) if label_ys else ci_max
    ymax = max(ci_max, label_max) + 0.03
    ax.set_ylim(0, min(1.0, ymax))

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = _project_root()
    gt_path = os.path.join(root, "data", "data_post_op.csv")
    experiment_results_dir = os.path.join(root, "experiment_results")
    out_base = os.path.join(root, "analysis_results", "rouge_l_explanations")

    os.makedirs(out_base, exist_ok=True)

    gt = load_ground_truth(gt_path)

    extra_names_path = os.path.join(root, "data", "tier1_first_names_for_rouge.txt")
    extra_first_names = load_extra_first_names_from_file(extra_names_path)
    if extra_first_names:
        print(f"Loaded {len(extra_first_names)} extra first name(s) from {extra_names_path}")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"WARNING: No model folders under {experiment_results_dir}")
        raise SystemExit(0)

    all_scores = []
    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        tier1_path = find_tier_csv(model_dir, 1)
        if not tier1_path:
            print(f"WARNING: {model_id}: no Tier 1 CSV found, skipping.")
            continue

        try:
            scored = score_model_rouge_l(
                tier1_path,
                gt,
                model_id=model_id,
                extra_first_names=extra_first_names if extra_first_names else None,
            )
        except Exception as e:
            print(f"WARNING: {model_id}: failed to score ROUGE-L ({e}), skipping.")
            continue

        out_dir = os.path.join(out_base, model_id)
        os.makedirs(out_dir, exist_ok=True)
        scored.to_csv(os.path.join(out_dir, "rouge_l_scores_tier1_ff.csv"), index=False)
        all_scores.append(scored)
        print(f"{model_id}: saved {len(scored)} ROUGE-L scores to {out_dir}")

    if not all_scores:
        print("No models were scored; exiting.")
        raise SystemExit(0)

    all_df = pd.concat(all_scores, ignore_index=True)
    all_df.to_csv(os.path.join(out_base, "rouge_l_scores_all_models_tier1_ff.csv"), index=False)

    # Summary per model
    rows: list[Dict[str, Any]] = []
    for model_id, g in all_df.groupby("model_id"):
        x = g["rouge_l_f1"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci_mean(x, n_boot=5000, alpha=0.05, seed=0)
        rows.append({
            "model_id": model_id,
            "n": int(len(g)),
            "mean_rouge_l_f1": float(np.mean(x)),
            "median_rouge_l_f1": float(np.median(x)),
            "ci_low": lo,
            "ci_high": hi,
        })

    summary = pd.DataFrame(rows).sort_values("mean_rouge_l_f1", ascending=False)
    summary.to_csv(os.path.join(out_base, "rouge_l_summary_tier1_ff.csv"), index=False)

    plot_model_means(summary, save_path=os.path.join(out_base, "rouge_l_mean_barplot_tier1_ff.png"))
    print(f"Saved summary + plot to {out_base}")

