"""
analyze_cosine_sim_qwen3_explanations.py
========================================
Replaces ROUGE-L with **cosine similarity** between Qwen3-Embedding vectors
(via a local Ollama server) for Tier-1 model explanation evaluation.

Embedding model : qwen3-embedding:latest  (Ollama)
Similarity      : cosine similarity  ∈ [-1, 1]  (typically [0, 1] for text)

Output columns match the old schema but with  rouge_l_f1 → cosine_similarity.
Output directory: analysis_results/cosine_sim_explanations/
"""
from __future__ import annotations

import os
import glob
import json
import urllib.request
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

# Ollama endpoint (change if you run Ollama on a different host/port)
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model served by Ollama
EMBEDDING_MODEL: str = os.environ.get("QWEN3_EMBED_MODEL", "qwen3-embedding:latest")

# Number of texts to embed in a single Ollama request (tune for memory / speed)
EMBEDDING_BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

# --- Plot typography (edit sizes, then re-run script) ---
COSINE_PLOT_FONTS = {
    "title": 14,
    "axis_label": 14,
    "tick": 15,
    "bar_value": 9,
}



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
# Qwen3 embeddings via Ollama HTTP API
# ---------------------------------------------------------------------------

def _ollama_embed_batch(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> np.ndarray:
    """
    Call POST /api/embed (Ollama ≥ 0.3.x) with a list of texts.
    Returns shape (len(texts), dim) float32 array.
    """
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    # Ollama returns {"embeddings": [[...], [...], ...]}
    return np.array(result["embeddings"], dtype=np.float32)


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    model: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> np.ndarray:
    """
    Embed a list of texts in batches.  Returns shape (N, dim) float32.
    Empty / NaN strings produce a zero vector (cosine similarity = 0).
    """
    cleaned: list[str] = []
    empty_mask: list[bool] = []
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)) or str(t).strip() == "":
            cleaned.append("")
            empty_mask.append(True)
        else:
            cleaned.append(str(t).strip())
            empty_mask.append(False)

    all_vecs: list[np.ndarray] = []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        vecs = _ollama_embed_batch(batch, model=model, base_url=base_url)
        all_vecs.append(vecs)

    result = np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0,), dtype=np.float32)

    for idx, is_empty in enumerate(empty_mask):
        if is_empty:
            result[idx] = 0.0

    return result


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between two (N, dim) arrays.
    Returns shape (N,) float32.  Zero vectors yield similarity = 0.
    """
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    denom = np.maximum(norm_a * norm_b, 1e-10)
    return (np.sum(a * b, axis=1, keepdims=True) / denom).squeeze(axis=1).astype(np.float32)


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

def score_model_cosine_sim(
    tier1_path: str,
    gt: pd.DataFrame,
    model_id: str,
    *,
    embed_batch_size: int = EMBEDDING_BATCH_SIZE,
    embedding_model: str = EMBEDDING_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
) -> pd.DataFrame:
    """
    Compute Qwen3-Embedding cosine similarity between each model explanation
    and the corresponding ground-truth explanation.

    Returns a DataFrame with columns:
        model_id, vignette_idx, race, gender,
        cosine_similarity, model_explanation, tier1_path
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

    gt_texts: list[str] = merged["gt_explanation"].tolist()
    model_texts: list[str] = merged["model_explanation"].tolist()

    print(f"  [{model_id}] Embedding {len(gt_texts)} ground-truth texts …")
    gt_vecs = embed_texts(
        gt_texts,
        batch_size=embed_batch_size,
        model=embedding_model,
        base_url=ollama_base_url,
    )

    print(f"  [{model_id}] Embedding {len(model_texts)} model explanation texts …")
    model_vecs = embed_texts(
        model_texts,
        batch_size=embed_batch_size,
        model=embedding_model,
        base_url=ollama_base_url,
    )

    sims = cosine_similarity_rowwise(gt_vecs, model_vecs)
    merged["cosine_similarity"] = sims.tolist()
    merged["model_id"] = model_id
    merged["tier1_path"] = tier1_path

    return merged[
        [
            "model_id",
            "vignette_idx",
            "race",
            "gender",
            "cosine_similarity",
            "model_explanation",
            "tier1_path",
        ]
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Increase font size of x and y-ticks and x and y-labels
def plot_model_means(summary: pd.DataFrame, save_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fonts = COSINE_PLOT_FONTS
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    s = summary.sort_values("mean_cosine_similarity", ascending=False).copy()
    x = np.arange(len(s))

    ax.bar(
        x,
        s["mean_cosine_similarity"],
        color=sns.color_palette("Set2", len(s)),
        alpha=0.9,
        edgecolor="0.35",
        linewidth=0.6,
    )

    y = s["mean_cosine_similarity"].to_numpy(dtype=float)
    lo = s["ci_low"].to_numpy(dtype=float)
    hi = s["ci_high"].to_numpy(dtype=float)
    err_lo = np.maximum(0.0, y - lo)
    err_hi = np.maximum(0.0, hi - y)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="0.2", elinewidth=1.2, capsize=4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(s["model_id"].tolist(), rotation=30, ha="right", fontsize=fonts["tick"])
    ax.set_ylabel("Mean Cosine Similarity (Qwen3-Embedding)", fontsize=fonts["axis_label"])
    ax.tick_params(axis="y", labelsize=fonts["tick"])

    label_ys: list[float] = []
    label_pad = 0.008
    for xi, yi, ci_hi in zip(x, y, hi):
        if not np.isfinite(yi):
            continue
        y_label = (ci_hi + label_pad) if np.isfinite(ci_hi) else yi + label_pad
        label_ys.append(y_label)
        ax.text(xi, y_label, f"{yi:.3f}", ha="center", va="bottom", fontsize=fonts["bar_value"], clip_on=False)

    ci_max = float(np.nanmax(hi)) if np.isfinite(hi).any() else float(np.nanmax(y))
    label_max = float(np.nanmax(label_ys)) if label_ys else ci_max
    ymax = max(ci_max, label_max) + 0.02
    ax.set_ylim(0, min(1.0, ymax))

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
    out_base = os.path.join(root, "analysis_results", "cosine_sim_explanations")

    os.makedirs(out_base, exist_ok=True)

    print(f"Loading ground truth from: {gt_path}")
    gt = load_ground_truth(gt_path)

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"WARNING: No model folders under {experiment_results_dir}")
        raise SystemExit(0)

    print(f"Models found: {model_ids}\n")
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Ollama URL      : {OLLAMA_BASE_URL}")
    print(f"Batch size      : {EMBEDDING_BATCH_SIZE}\n")

    all_scores: list[pd.DataFrame] = []
    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        tier1_path = find_tier_csv(model_dir, 1)
        if not tier1_path:
            print(f"WARNING: {model_id}: no Tier 1 CSV found, skipping.")
            continue

        try:
            scored = score_model_cosine_sim(tier1_path, gt, model_id=model_id)
        except Exception as e:
            print(f"WARNING: {model_id}: failed to score ({e}), skipping.")
            continue

        out_dir = os.path.join(out_base, model_id)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "cosine_sim_scores_tier1_ff.csv")
        scored.to_csv(out_csv, index=False)
        all_scores.append(scored)
        mean_sim = float(np.nanmean(scored["cosine_similarity"].to_numpy(dtype=float)))
        print(f"{model_id}: n={len(scored)}, mean cosine_sim={mean_sim:.4f}  →  {out_csv}\n")

    if not all_scores:
        print("No models were scored; exiting.")
        raise SystemExit(0)

    all_df = pd.concat(all_scores, ignore_index=True)
    all_csv = os.path.join(out_base, "cosine_sim_scores_all_models_tier1_ff.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved combined scores → {all_csv}")

    # Summary per model
    rows: list[Dict[str, Any]] = []
    for mid, g in all_df.groupby("model_id"):
        x = g["cosine_similarity"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci_mean(x, n_boot=5000, alpha=0.05, seed=0)
        rows.append(
            {
                "model_id": mid,
                "n": int(len(g)),
                "mean_cosine_similarity": float(np.nanmean(x)),
                "median_cosine_similarity": float(np.nanmedian(x)),
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_cosine_similarity", ascending=False)
    summary_csv = os.path.join(out_base, "cosine_sim_summary_tier1_ff.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary → {summary_csv}")

    plot_path = os.path.join(out_base, "cosine_sim_mean_barplot_tier1_ff.png")
    plot_model_means(summary, save_path=plot_path)
    print(f"\nDone. All outputs under: {out_base}")
