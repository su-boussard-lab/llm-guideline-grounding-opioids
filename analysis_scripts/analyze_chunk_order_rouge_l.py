#!/usr/bin/env python3
"""
ROUGE-L comparison: baseline Tier 1 retrieved vs chunk-order permutation runs.

Joins each model's original Tier 1 CSV (original chunk order) with
  experiment_results/<model>/chunk_order_tier1/results_tier1_chunk_order_permutations_ff.csv

Usage:
  python analysis_scripts/analyze_chunk_order_rouge_l.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_rouge_l_explanations import (  # noqa: E402
    _detect_explanation_col,
    rouge_l_f1,
    standardize_model_explanation_for_rouge,
)


def find_tier1_csv(model_dir: str) -> str | None:
    patterns = [
        os.path.join(model_dir, "*tier1*_ff*.csv"),
        os.path.join(model_dir, "*tier1*.csv"),
    ]
    for pat in patterns:
        matches = [m for m in sorted(glob.glob(pat)) if ".bak." not in os.path.basename(m)]
        if matches:
            return matches[0]
    return None


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def discover_retrieved_models(exp_dir: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(exp_dir)):
        if name.endswith("_retrieved"):
            p = os.path.join(exp_dir, name)
            if os.path.isdir(p):
                out.append(name)
    return out


def score_model(model_folder: str, exp_dir: str, out_base: str) -> pd.DataFrame | None:
    model_dir = os.path.join(exp_dir, model_folder)
    baseline_path = find_tier1_csv(model_dir)
    perm_path = os.path.join(
        model_dir, "chunk_order_tier1", "results_tier1_chunk_order_permutations_ff.csv"
    )
    if not baseline_path or not os.path.isfile(perm_path):
        return None

    base = pd.read_csv(baseline_path)
    perm = pd.read_csv(perm_path)
    if "permutation_id" not in perm.columns:
        raise ValueError(f"{perm_path}: missing permutation_id column")

    expl_base = _detect_explanation_col(base)
    expl_perm = _detect_explanation_col(perm)

    join_cols = ["vignette_idx", "race", "gender"]
    for c in join_cols:
        if c not in base.columns or c not in perm.columns:
            raise ValueError(f"Missing join column {c}")

    b = base[join_cols + [expl_base]].copy()
    b = b.rename(columns={expl_base: "baseline_explanation"})
    p = perm[join_cols + ["permutation_id", "chunk_order", expl_perm]].copy()
    p = p.rename(columns={expl_perm: "permuted_explanation"})

    merged = p.merge(b, on=join_cols, how="left")
    if merged["baseline_explanation"].isna().any():
        n = int(merged["baseline_explanation"].isna().sum())
        print(f"  WARNING {model_folder}: {n} rows missing baseline match")

    merged["permuted_explanation_std"] = merged["permuted_explanation"].apply(
        standardize_model_explanation_for_rouge
    )
    merged["baseline_explanation_std"] = merged["baseline_explanation"].apply(
        standardize_model_explanation_for_rouge
    )
    merged["rouge_l_f1"] = [
        rouge_l_f1(ref, cand)
        for ref, cand in zip(merged["baseline_explanation_std"], merged["permuted_explanation_std"])
    ]
    merged["model_folder"] = model_folder

    out_dir = os.path.join(out_base, model_folder)
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(os.path.join(out_dir, "chunk_order_rouge_l_scores_ff.csv"), index=False)

    summary_rows = []
    for perm_id, g in merged.groupby("permutation_id"):
        x = g["rouge_l_f1"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model_folder": model_folder,
                "permutation_id": perm_id,
                "n": len(g),
                "mean_rouge_l_f1": float(np.nanmean(x)),
                "median_rouge_l_f1": float(np.nanmedian(x)),
                "pct_rouge_ge_0.9": float(np.mean(x >= 0.9) * 100.0),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "chunk_order_rouge_l_summary_ff.csv"), index=False)
    return merged


def main() -> None:
    root = _project_root()
    exp_dir = os.path.join(root, "experiment_results")
    out_base = os.path.join(root, "analysis_results", "chunk_order_rouge_l")
    os.makedirs(out_base, exist_ok=True)

    all_summaries: list[pd.DataFrame] = []
    for model_folder in discover_retrieved_models(exp_dir):
        print(f"Scoring {model_folder}...")
        try:
            score_model(model_folder, exp_dir, out_base)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        summ_path = os.path.join(out_base, model_folder, "chunk_order_rouge_l_summary_ff.csv")
        if os.path.isfile(summ_path):
            all_summaries.append(pd.read_csv(summ_path))

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(out_base, "chunk_order_rouge_l_summary_all_models_ff.csv"),
            index=False,
        )
        print(f"Saved summaries under {out_base}")


if __name__ == "__main__":
    main()
