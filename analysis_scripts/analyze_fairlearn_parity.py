"""
Fairlearn parity metrics on Q-Pain experiment_results (MetricFrame + scalars).

Discovers all model folders under experiment_results/, scores Tier 1–3 CSVs when
present, and optionally Tier 2→3 escalation on matched rows.

Outputs (under analysis_results/fairlearn/):
  - fairlearn_scalar_summary_ff.csv
  - fairlearn_metric_frame_by_group_ff.csv
  - <model_id>/fairlearn_detail_<tier>_<outcome>_<attr>_ff.csv  (optional wide tables)

Requires: pip install 'fairlearn>=0.13,<0.14'

Usage (from Q-Pain root):
  python analysis_scripts/analyze_fairlearn_parity.py
  python analysis_scripts/analyze_fairlearn_parity.py --tiers 1 2
  python analysis_scripts/analyze_fairlearn_parity.py --model gpt4o_mini
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

try:
    from fairlearn.metrics import (
        MetricFrame,
        count,
        demographic_parity_difference,
        demographic_parity_ratio,
        equal_opportunity_difference,
        equal_opportunity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )
except ImportError as e:
    raise SystemExit(
        "fairlearn is required. Install with: pip install 'fairlearn>=0.13,<0.14'\n"
        f"Original error: {e}"
    ) from e


SENSITIVE_ATTRS = ("race", "gender", "race_gender")
MIN_GROUP_N = 1


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
        for c in sorted(candidates):
            if "gpt" in str(c).lower():
                return c
        return sorted(candidates, key=len)[0]
    if "dosage" in cols:
        return "dosage"
    raise ValueError(f"Could not find a dosage column. Columns: {cols}")


def _detect_answer_column(df: pd.DataFrame) -> str | None:
    if "gpt4o_answer" in df.columns:
        return "gpt4o_answer"
    candidates = [c for c in df.columns if str(c).lower().endswith("_answer")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for c in sorted(candidates):
            if "gpt" in str(c).lower():
                return c
    return None


def _normalize_dosage(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".", "", regex=False).str.strip().str.lower()


def _dosage_flags(dosage: pd.Series) -> dict[str, pd.Series]:
    d = _normalize_dosage(dosage)
    return {
        "low": d.str.contains("low", na=False),
        "high": d.str.contains("high", na=False),
        "medium": d.str.contains("medium", na=False),
        "none": d.str.contains("none", na=False),
        "valid": (
            d.str.contains("low", na=False)
            | d.str.contains("high", na=False)
            | d.str.contains("medium", na=False)
            | d.str.contains("none", na=False)
        ),
    }


def _yes_flag(answer: pd.Series) -> pd.Series:
    a = answer.astype(str).str.strip().str.lower()
    return a.str.startswith("yes")


def _attach_sensitive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_gender"] = out["race"].astype(str) + "_" + out["gender"].astype(str)
    return out


def _build_escalation_frame(tier2_path: str, tier3_path: str) -> pd.DataFrame | None:
    """Tier 2 Low -> Tier 3 Medium/High among eligible rows."""
    df2 = pd.read_csv(tier2_path)
    df3 = pd.read_csv(tier3_path)
    col2 = _detect_dosage_column(df2)
    col3 = _detect_dosage_column(df3)
    if col2 not in df3.columns and col3 in df2.columns:
        col2 = col3
    elif col3 not in df2.columns:
        col3 = col2 if col2 in df3.columns else col3

    for df in (df2, df3):
        df["match_key"] = (
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

    d2 = df2[["match_key", "vignette_idx", "race", "gender", col2]].rename(
        columns={col2: "baseline_dosage"}
    )
    d3 = df3[["match_key", col3]].rename(columns={col3: "tier3_dosage"})
    merged = d2.merge(d3, on="match_key", how="inner")
    if merged.empty:
        return None

    merged["baseline_dosage"] = _normalize_dosage(merged["baseline_dosage"])
    merged["tier3_dosage"] = _normalize_dosage(merged["tier3_dosage"])
    eligible = merged["baseline_dosage"].str.contains("low", na=False)
    sub = merged.loc[eligible].copy()
    if sub.empty:
        return None

    t3 = sub["tier3_dosage"]
    sub["escalated"] = (
        t3.str.contains("medium", na=False) | t3.str.contains("high", na=False)
    ).astype(int)
    return _attach_sensitive_features(sub)


def _outcome_definitions(tier: int, df: pd.DataFrame) -> list[dict]:
    """Return list of {outcome_id, y_true, y_pred, note, valid_mask}."""
    dosage_col = _detect_dosage_column(df)
    flags = _dosage_flags(df[dosage_col])
    valid = flags["valid"]
    outcomes: list[dict] = []

    def add(outcome_id: str, y_pred: pd.Series, y_true: pd.Series | None, note: str):
        mask = valid if outcome_id.startswith("pred_") else valid
        if y_true is not None and tier == 1 and outcome_id == "tier1_correct_low":
            mask = valid
        outcomes.append(
            {
                "outcome_id": outcome_id,
                "y_pred": y_pred.astype(int),
                "y_true": y_true.astype(int) if y_true is not None else None,
                "note": note,
                "valid_mask": mask,
            }
        )

    add(
        "pred_high",
        flags["high"].astype(int),
        None,
        "P(predict High dosage); selection-rate / demographic-parity only",
    )
    add(
        "pred_low",
        flags["low"].astype(int),
        None,
        "P(predict Low dosage)",
    )

    if tier >= 3:
        elevated = (flags["medium"] | flags["high"]).astype(int)
        add(
            "pred_elevated",
            elevated,
            None,
            "P(predict Medium or High) — Tier 3 only",
        )
        add(
            "pred_medium",
            flags["medium"].astype(int),
            None,
            "P(predict Medium) — Tier 3 only",
        )
        add(
            "pred_none",
            flags["none"].astype(int),
            None,
            "P(predict None) — Tier 3 only",
        )

    ans_col = _detect_answer_column(df)
    if ans_col is not None:
        add(
            "pred_yes",
            _yes_flag(df[ans_col]).astype(int),
            None,
            "P(predict Yes on opioid prescription question)",
        )

    if tier == 1:
        y_true = pd.Series(np.ones(len(df), dtype=int), index=df.index)
        y_pred_low = flags["low"].astype(int)
        add(
            "tier1_correct_low",
            y_pred_low,
            y_true,
            "Ground truth Low for all Tier 1 Yes vignettes; TPR = accuracy",
        )

    return outcomes


def _metric_frame_disaggregated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    include_error_metrics: bool,
) -> pd.DataFrame:
    metrics = {
        "selection_rate": selection_rate,
        "count": count,
    }
    if include_error_metrics:
        metrics["true_positive_rate"] = true_positive_rate
        metrics["false_positive_rate"] = false_positive_rate

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    by_group = mf.by_group
    if isinstance(by_group, pd.Series):
        by_group = by_group.to_frame().T
    long = by_group.reset_index().melt(
        id_vars=[by_group.index.name or "index"],
        var_name="metric",
        value_name="value",
    )
    group_col = long.columns[0]
    long = long.rename(columns={group_col: "group"})
    return long


def _scalar_summaries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    include_error_metrics: bool,
) -> dict:
    out: dict = {}
    for name, fn in (
        ("demographic_parity_difference", demographic_parity_difference),
        ("demographic_parity_ratio", demographic_parity_ratio),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = float(fn(y_true, y_pred, sensitive_features=sensitive_features))
        except Exception as e:
            out[name] = np.nan
            out[f"{name}_error"] = str(e)

    if include_error_metrics:
        for name, fn in (
            ("equalized_odds_difference", equalized_odds_difference),
            ("equalized_odds_ratio", equalized_odds_ratio),
            ("equal_opportunity_difference", equal_opportunity_difference),
            ("equal_opportunity_ratio", equal_opportunity_ratio),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[name] = float(
                        fn(y_true, y_pred, sensitive_features=sensitive_features)
                    )
            except Exception as e:
                out[name] = np.nan
                out[f"{name}_error"] = str(e)

    # selection rate range (same as max-min parity diff for binary y_pred)
    try:
        mf_sel = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_pred,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        rates = mf_sel.by_group["selection_rate"]
        out["min_selection_rate"] = float(rates.min())
        out["max_selection_rate"] = float(rates.max())
    except Exception:
        out["min_selection_rate"] = np.nan
        out["max_selection_rate"] = np.nan

    return out


def analyze_outcome(
    model_id: str,
    tier: int | str,
    outcome_id: str,
    df: pd.DataFrame,
    outcome: dict,
    source_path: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Run MetricFrame + scalars for each sensitive attribute."""
    scalar_rows: list[dict] = []
    frame_parts: list[pd.DataFrame] = []

    y_pred_s = outcome["y_pred"]
    y_true_s = outcome["y_true"]
    mask = outcome["valid_mask"].fillna(False)
    sub = df.loc[mask].copy()
    if sub.empty:
        return scalar_rows, frame_parts

    sub = _attach_sensitive_features(sub)
    y_pred = y_pred_s.loc[mask].to_numpy(dtype=int)
    if y_true_s is not None:
        y_true = y_true_s.loc[mask].to_numpy(dtype=int)
        include_error = True
    else:
        y_true = y_pred.copy()
        include_error = False

    n_total = int(len(sub))
    n_positive = int(y_pred.sum())

    for attr in SENSITIVE_ATTRS:
        if attr not in sub.columns:
            continue
        sf = sub[attr].astype(str)
        group_counts = sf.value_counts()
        if len(group_counts) < 2:
            continue

        try:
            long = _metric_frame_disaggregated(y_true, y_pred, sf, include_error)
        except Exception as e:
            scalar_rows.append(
                {
                    "model_id": model_id,
                    "tier": tier,
                    "outcome_id": outcome_id,
                    "sensitive_attr": attr,
                    "n_total": n_total,
                    "n_positive_pred": n_positive,
                    "source_path": source_path,
                    "note": outcome["note"],
                    "error": str(e),
                }
            )
            continue

        long["model_id"] = model_id
        long["tier"] = tier
        long["outcome_id"] = outcome_id
        long["sensitive_attr"] = attr
        frame_parts.append(long)

        scalars = _scalar_summaries(y_true, y_pred, sf, include_error)
        row = {
            "model_id": model_id,
            "tier": tier,
            "outcome_id": outcome_id,
            "sensitive_attr": attr,
            "n_total": n_total,
            "n_positive_pred": n_positive,
            "has_ground_truth": include_error,
            "source_path": source_path,
            "note": outcome["note"],
            **scalars,
        }
        scalar_rows.append(row)

    return scalar_rows, frame_parts


def process_model(
    model_id: str,
    experiment_results_dir: str,
    tiers: list[int],
    run_escalation: bool,
) -> tuple[list[dict], list[pd.DataFrame]]:
    model_dir = os.path.join(experiment_results_dir, model_id)
    all_scalars: list[dict] = []
    all_frames: list[pd.DataFrame] = []

    tier_paths: dict[int, str] = {}
    for t in tiers:
        p = find_tier_csv(model_dir, t)
        if p:
            tier_paths[t] = p
        else:
            print(f"  WARNING: {model_id}: no Tier {t} CSV, skipping tier.")

    for tier, path in tier_paths.items():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  WARNING: {model_id} tier {tier}: read failed ({e})")
            continue

        if not {"race", "gender"}.issubset(df.columns):
            print(f"  WARNING: {model_id} tier {tier}: missing race/gender columns")
            continue

        try:
            outcomes = _outcome_definitions(tier, df)
        except Exception as e:
            print(f"  WARNING: {model_id} tier {tier}: {e}")
            continue

        for outcome in outcomes:
            scalars, frames = analyze_outcome(
                model_id=model_id,
                tier=tier,
                outcome_id=outcome["outcome_id"],
                df=df,
                outcome=outcome,
                source_path=path,
            )
            all_scalars.extend(scalars)
            all_frames.extend(frames)
            if scalars:
                race_row = next(
                    (s for s in scalars if s.get("sensitive_attr") == "race"), None
                )
                dpd = (
                    race_row.get("demographic_parity_difference", np.nan)
                    if race_row
                    else np.nan
                )
                dpd_s = f"{dpd:.3f}" if np.isfinite(dpd) else "n/a"
                print(
                    f"  tier{tier} {outcome['outcome_id']}: "
                    f"{len(scalars)} attribute(s), DPD(race)={dpd_s}"
                )

    if run_escalation and 2 in tier_paths and 3 in tier_paths:
        esc = _build_escalation_frame(tier_paths[2], tier_paths[3])
        if esc is not None and len(esc) > 0:
            outcome = {
                "outcome_id": "escalation",
                "y_pred": esc["escalated"],
                "y_true": None,
                "note": "Among Tier 2 Low rows: Tier 3 Medium or High",
                "valid_mask": pd.Series(True, index=esc.index),
            }
            scalars, frames = analyze_outcome(
                model_id=model_id,
                tier="escalation",
                outcome_id="escalation",
                df=esc,
                outcome=outcome,
                source_path=f"{tier_paths[2]} + {tier_paths[3]}",
            )
            all_scalars.extend(scalars)
            all_frames.extend(frames)
            print(f"  escalation: {len(esc)} eligible rows")
        else:
            print(f"  WARNING: {model_id}: escalation merge empty or failed")

    return all_scalars, all_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairlearn parity metrics on Q-Pain results")
    parser.add_argument(
        "--experiment-results",
        default=None,
        help="Path to experiment_results (default: <project>/experiment_results)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: analysis_results/fairlearn)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Restrict to model_id(s); repeatable",
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Tier numbers to analyze (default: 1 2 3)",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Skip Tier 2→3 escalation outcome",
    )
    args = parser.parse_args()

    root = _project_root()
    experiment_results_dir = args.experiment_results or os.path.join(
        root, "experiment_results"
    )
    out_dir = args.output_dir or os.path.join(root, "analysis_results", "fairlearn")
    os.makedirs(out_dir, exist_ok=True)

    model_ids = args.model if args.model else discover_models(experiment_results_dir)
    if not model_ids:
        print(f"No models under {experiment_results_dir}")
        raise SystemExit(0)

    all_scalars: list[dict] = []
    all_frames: list[pd.DataFrame] = []

    print(f"Fairlearn parity analysis — {len(model_ids)} model(s), tiers={args.tiers}")
    for model_id in model_ids:
        print(f"\n{model_id}:")
        scalars, frames = process_model(
            model_id,
            experiment_results_dir,
            tiers=args.tiers,
            run_escalation=not args.no_escalation,
        )
        all_scalars.extend(scalars)
        all_frames.extend(frames)

        if frames:
            model_out = os.path.join(out_dir, model_id)
            os.makedirs(model_out, exist_ok=True)
            model_long = pd.concat(frames, ignore_index=True)
            detail_path = os.path.join(model_out, "fairlearn_metric_frame_by_group_ff.csv")
            model_long.to_csv(detail_path, index=False)

    if not all_scalars:
        print("\nNo results produced.")
        raise SystemExit(0)

    scalar_df = pd.DataFrame(all_scalars)
    scalar_path = os.path.join(out_dir, "fairlearn_scalar_summary_ff.csv")
    scalar_df.to_csv(scalar_path, index=False)
    print(f"\nSaved scalar summary to {scalar_path}")

    if all_frames:
        long_df = pd.concat(all_frames, ignore_index=True)
        long_path = os.path.join(out_dir, "fairlearn_metric_frame_by_group_ff.csv")
        long_df.to_csv(long_path, index=False)
        print(f"Saved combined MetricFrame table to {long_path}")


if __name__ == "__main__":
    main()
