"""
analyze_rouge_l_explanations_llm_as_a_judge.py
===============================================
LLM-as-a-judge evaluation of Tier-1 model explanations using
**alibayram/medgemma:27b** (served locally via Ollama).

For each (model_explanation, ground_truth_explanation) pair the judge LLM is
asked to rate clinical similarity on a 1-5 integer scale.  The raw score is
stored as `judge_score` (1–5) and the normalised value `judge_score_norm`
([0, 1]) is used for summaries and plots.

Output directory: analysis_results/llm_judge_explanations/
"""
from __future__ import annotations

import os
import re
import glob
import json
import time
import urllib.request
import urllib.error
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration  (override via environment variables)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL: str     = os.environ.get("JUDGE_MODEL",    "alibayram/medgemma:27b")

# Seconds to wait before retrying a failed Ollama call
RETRY_DELAY: float   = float(os.environ.get("RETRY_DELAY",   "2.0"))
MAX_RETRIES: int     = int(os.environ.get("MAX_RETRIES",      "3"))

# --- Plot typography ---
JUDGE_PLOT_FONTS = {
    "title":      14,
    "axis_label": 14,
    "tick":       15,
    "bar_value":  9,
}

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified anaesthesiologist and clinical-reasoning expert.
Your task is to evaluate how well a candidate clinical explanation captures
the same medical content as a reference explanation for a post-operative
pain management vignette.

Score the candidate on a scale of 1–5 using ONLY these criteria:
  5 – Excellent: all key clinical facts, reasoning, and recommendations match.
  4 – Good: most key points match; minor omissions or wording differences only.
  3 – Adequate: core reasoning is present but notable gaps or inaccuracies exist.
  2 – Poor: partial overlap; key clinical facts are missing or incorrect.
  1 – Unacceptable: the candidate fails to capture the clinical content.

Respond with a SINGLE integer (1, 2, 3, 4, or 5) and nothing else."""

USER_TEMPLATE = (
    "Reference explanation:\n"
    "'''\n"
    "{reference}\n"
    "'''\n\n"
    "Candidate explanation:\n"
    "'''\n"
    "{candidate}\n"
    "'''\n\n"
    "Score (1\u20135):"
)

# ---------------------------------------------------------------------------
# Path / model discovery helpers
# ---------------------------------------------------------------------------

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
    """Detect the model explanation column in a tier-1 results CSV."""
    cols = list(df.columns)
    expl = [c for c in cols if c.lower().endswith("_explanation")]
    if len(expl) == 1:
        return expl[0]
    if len(expl) > 1:
        for c in expl:
            if c.lower() == "gpt4o_explanation":
                return c
        return sorted(expl, key=len)[0]
    contains = [c for c in cols if "explanation" in c.lower()]
    if contains:
        return sorted(contains, key=len)[0]
    raise ValueError(f"Could not find an explanation column. Columns: {cols}")


# ---------------------------------------------------------------------------
# Ollama chat helper
# ---------------------------------------------------------------------------

def _ollama_chat(
    system: str,
    user: str,
    model: str = JUDGE_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Call POST /api/chat (Ollama) with a system + user message.
    Returns the assistant reply text.
    """
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {
            "temperature": 0.0,   # deterministic scoring
            "num_predict": 4,     # we only need a single digit
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    # Ollama returns {"message": {"role": "assistant", "content": "..."}}
    return result["message"]["content"].strip()


def _parse_score(reply: str) -> float | None:
    """
    Extract a 1-5 integer from the model reply.
    Returns None if no valid score is found (will be treated as NaN).
    """
    m = re.search(r"[1-5]", reply)
    if m:
        return float(m.group(0))
    return None


def judge_pair(
    reference: str,
    candidate: str,
    *,
    model: str = JUDGE_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> float:
    """
    Ask the judge LLM to rate (candidate vs reference) on 1-5.
    Returns the raw integer score, or NaN on repeated failure.
    """
    if not reference or not candidate:
        return float("nan")

    user_msg = USER_TEMPLATE.format(reference=reference, candidate=candidate)

    for attempt in range(1, max_retries + 1):
        try:
            reply = _ollama_chat(SYSTEM_PROMPT, user_msg, model=model, base_url=base_url)
            score = _parse_score(reply)
            if score is not None:
                return score
            # Unparseable reply — retry
            print(f"    WARNING: unparseable reply on attempt {attempt}: {repr(reply)}")
        except urllib.error.URLError as exc:
            print(f"    WARNING: Ollama request failed on attempt {attempt}: {exc}")
        if attempt < max_retries:
            time.sleep(retry_delay)

    return float("nan")


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci_mean(
    x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(x, size=x.size, replace=True))) for _ in range(n_boot)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str) -> pd.DataFrame:
    """
    Load ground-truth explanations CSV.
    Expects an 'Explanation' column; creates 'vignette_idx' from row order if absent.
    """
    gt = pd.read_csv(gt_path)
    if "Explanation" not in gt.columns:
        raise ValueError(
            f"Ground-truth file must include 'Explanation' column. Found: {sorted(gt.columns)}"
        )
    if "vignette_idx" in gt.columns:
        gt["vignette_idx"] = gt["vignette_idx"].astype(int)
    else:
        gt = gt.reset_index(drop=True)
        gt["vignette_idx"] = gt.index.astype(int)
    gt = gt[["vignette_idx", "Explanation"]].copy()
    gt = gt.rename(columns={"Explanation": "gt_explanation"})  # type: ignore[call-overload]
    return gt


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_model_llm_judge(
    tier1_path: str,
    gt: pd.DataFrame,
    model_id: str,
    *,
    judge_model: str = JUDGE_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> pd.DataFrame:
    """
    For each row in the tier-1 CSV, call the judge LLM to score
    (model_explanation, gt_explanation) on 1-5.

    Returns a DataFrame with columns:
        model_id, vignette_idx, race, gender,
        judge_score        (raw 1-5, NaN on failure)
        judge_score_norm   ([0,1] = (score-1)/4)
        model_explanation, tier1_path
    """
    df = pd.read_csv(tier1_path)
    if "vignette_idx" not in df.columns:
        raise ValueError(f"{model_id}: tier1 file missing 'vignette_idx' column: {tier1_path}")
    if "race" not in df.columns or "gender" not in df.columns:
        raise ValueError(f"{model_id}: tier1 file missing race/gender columns: {tier1_path}")

    expl_col = _detect_explanation_col(df)
    d = df[["vignette_idx", "race", "gender", expl_col]].copy()
    d = d.rename(columns={expl_col: "model_explanation"})  # type: ignore[call-overload]
    d["vignette_idx"] = d["vignette_idx"].astype(int)

    merged = d.merge(gt, on="vignette_idx", how="left")
    missing_gt = merged["gt_explanation"].isna()
    if missing_gt.any():
        missing_idxs = merged.loc[missing_gt, "vignette_idx"].unique().tolist()
        raise ValueError(
            f"{model_id}: missing ground-truth explanation for vignette_idx: {missing_idxs}. "
            "Check that ground-truth rows align with vignette_idx."
        )

    n = len(merged)
    scores: list[float] = []
    for i, row in enumerate(merged.itertuples(index=False), start=1):
        ref  = str(row.gt_explanation)  if row.gt_explanation  else ""
        cand = str(row.model_explanation) if row.model_explanation else ""
        s = judge_pair(
            ref, cand,
            model=judge_model,
            base_url=ollama_base_url,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        scores.append(s)
        status = f"{s:.0f}" if np.isfinite(s) else "NaN"
        print(f"    [{model_id}] {i}/{n}  vignette={row.vignette_idx}  score={status}")

    merged["judge_score"]      = scores
    # Normalise 1-5 → 0-1: (score - 1) / 4
    merged["judge_score_norm"] = merged["judge_score"].apply(
        lambda v: (v - 1.0) / 4.0 if np.isfinite(v) else float("nan")
    )
    merged["model_id"]   = model_id
    merged["tier1_path"] = tier1_path

    return merged[
        [
            "model_id",
            "vignette_idx",
            "race",
            "gender",
            "judge_score",
            "judge_score_norm",
            "model_explanation",
            "tier1_path",
        ]
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_model_means(summary: pd.DataFrame, save_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fonts = JUDGE_PLOT_FONTS
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    s = summary.sort_values("mean_judge_score", ascending=False).copy()
    x = np.arange(len(s))

    ax.bar(
        x,
        s["mean_judge_score"],
        color=sns.color_palette("Set2", len(s)),
        alpha=0.9,
        edgecolor="0.35",
        linewidth=0.6,
    )

    y  = s["mean_judge_score"].to_numpy(dtype=float)
    lo = s["ci_low"].to_numpy(dtype=float)
    hi = s["ci_high"].to_numpy(dtype=float)
    err_lo = np.maximum(0.0, y - lo)
    err_hi = np.maximum(0.0, hi - y)
    ax.errorbar(x, y, yerr=[err_lo, err_hi],
                fmt="none", ecolor="0.2", elinewidth=1.2, capsize=4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(s["model_id"].tolist(), rotation=30, ha="right", fontsize=fonts["tick"])
    ax.set_ylabel(f"Mean Judge Score (1–5)  [{JUDGE_MODEL}]", fontsize=fonts["axis_label"])
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    ax.set_ylim(1, 5.5)

    label_pad = 0.06
    for xi, yi, ci_hi in zip(x, y, hi):
        if not np.isfinite(yi):
            continue
        y_label = (ci_hi + label_pad) if np.isfinite(ci_hi) else yi + label_pad
        ax.text(xi, y_label, f"{yi:.2f}", ha="center", va="bottom",
                fontsize=fonts["bar_value"], clip_on=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved barplot → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = _project_root()
    gt_path = os.path.join(root, "analysis_scripts", "data", "data_post_op.csv")
    experiment_results_dir = os.path.join(root, "experiment_results")
    out_base = os.path.join(root, "analysis_results", "llm_judge_explanations")

    os.makedirs(out_base, exist_ok=True)

    print(f"Loading ground truth from: {gt_path}")
    gt = load_ground_truth(gt_path)

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"WARNING: No model folders under {experiment_results_dir}")
        raise SystemExit(0)

    print(f"Models found: {model_ids}\n")
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Ollama URL  : {OLLAMA_BASE_URL}\n")

    all_scores: list[pd.DataFrame] = []
    for model_id in model_ids:
        model_dir  = os.path.join(experiment_results_dir, model_id)
        tier1_path = find_tier_csv(model_dir, 1)
        if not tier1_path:
            print(f"WARNING: {model_id}: no Tier 1 CSV found, skipping.")
            continue

        print(f"\n=== Scoring {model_id} ===")
        try:
            scored = score_model_llm_judge(tier1_path, gt, model_id=model_id)
        except Exception as e:
            print(f"WARNING: {model_id}: failed to score ({e}), skipping.")
            continue

        out_dir = os.path.join(out_base, model_id)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "llm_judge_scores_tier1_ff.csv")
        scored.to_csv(out_csv, index=False)
        all_scores.append(scored)
        valid = scored["judge_score"].dropna()
        print(f"{model_id}: n={len(scored)}, valid={len(valid)}, "
              f"mean_score={valid.mean():.3f}  →  {out_csv}\n")

    if not all_scores:
        print("No models were scored; exiting.")
        raise SystemExit(0)

    all_df  = pd.concat(all_scores, ignore_index=True)
    all_csv = os.path.join(out_base, "llm_judge_scores_all_models_tier1_ff.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved combined scores → {all_csv}")

    # Summary per model
    rows: list[Dict[str, Any]] = []
    for mid, g in all_df.groupby("model_id"):
        x = g["judge_score"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci_mean(x, n_boot=5000, alpha=0.05, seed=0)
        rows.append({
            "model_id":            mid,
            "n":                   int(len(g)),
            "n_valid":             int(np.isfinite(x).sum()),
            "mean_judge_score":    float(np.nanmean(x)),
            "median_judge_score":  float(np.nanmedian(x)),
            "ci_low":              lo,
            "ci_high":             hi,
        })

    summary     = pd.DataFrame(rows).sort_values("mean_judge_score", ascending=False)
    summary_csv = os.path.join(out_base, "llm_judge_summary_tier1_ff.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary → {summary_csv}")

    plot_path = os.path.join(out_base, "llm_judge_mean_barplot_tier1_ff.png")
    plot_model_means(summary, save_path=plot_path)
    print(f"\nDone. All outputs under: {out_base}")
