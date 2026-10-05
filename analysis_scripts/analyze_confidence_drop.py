"""
Confidence Drop Analysis for Q-Pain Experiments

This script analyzes how model confidence changes across three experimental tiers:
- Tier 1: Demographics only, binary dosage (Low/High)
- Tier 2: Demographics + risk factors, binary dosage (Low/High)
- Tier 3: Demographics + risk factors, 3-level dosage (Low/Medium/High)

Two main comparisons:
1. Δ Tier 1→2: Does adding clinical complexity (risk factors) reduce confidence?
2. Δ Tier 2→3: Does adding dosage granularity reduce confidence in chosen option?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob
from typing import Tuple, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def _bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return lo, hi

def _bootstrap_ci_mean_of_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean(x - y), preserving pairing by resampling indices.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    diffs = (x[idx] - y[idx]).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1 - alpha))
    return lo, hi

def _paired_permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """
    Two-sided paired permutation test for mean difference between x and y.
    Uses random sign-flips on paired differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[mask]
    if d.size == 0:
        return np.nan
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size), replace=True)
    perm_means = (signs * d).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs)).mean())
    return p

def _save_confidence_trajectory_stats(
    df_12: Optional[pd.DataFrame],
    df_23: Optional[pd.DataFrame],
    output_dir: str,
    n_boot: int = 5000,
    n_perm: int = 20000,
) -> None:
    """
    Save a compact per-model summary of mean confidence across tiers plus 95% CIs and
    paired permutation p-values for tier-to-tier changes.
    """
    if df_12 is None or df_23 is None:
        return

    t1 = df_12["confidence_tier1"].to_numpy()
    t2 = df_12["confidence_tier2"].to_numpy()
    t3 = df_23["confidence_tier3"].to_numpy()

    # Tier means and CIs (means computed on the matched cohorts used in comparisons)
    t1_mean = float(np.nanmean(t1))
    t2_mean = float(np.nanmean(t2))
    t3_mean = float(np.nanmean(t3))

    t1_ci = _bootstrap_ci_mean(t1, n_boot=n_boot, seed=1)
    t2_ci = _bootstrap_ci_mean(t2, n_boot=n_boot, seed=2)
    t3_ci = _bootstrap_ci_mean(t3, n_boot=n_boot, seed=3)

    d12 = t2 - t1
    d23 = df_23["confidence_tier3"].to_numpy() - df_23["confidence_tier2"].to_numpy()

    d12_mean = float(np.nanmean(d12))
    d23_mean = float(np.nanmean(d23))
    d13_mean = t3_mean - t1_mean

    d12_ci = _bootstrap_ci_mean_of_diff(t2, t1, n_boot=n_boot, seed=11)
    d23_ci = _bootstrap_ci_mean_of_diff(df_23["confidence_tier3"].to_numpy(), df_23["confidence_tier2"].to_numpy(), n_boot=n_boot, seed=12)
    # Tier1->Tier3 uses the same weighting as Tier1->2 and Tier2->3 cohorts; we summarize as mean(T3) - mean(T1)
    # and bootstrap it by resampling indices within each cohort independently (conservative).
    d13_ci = (np.nan, np.nan)
    try:
        rng = np.random.default_rng(13)
        # bootstrap mean(T3) - mean(T1) by resampling within each vector
        t1v = np.asarray(t1, dtype=float); t1v = t1v[np.isfinite(t1v)]
        t3v = np.asarray(t3, dtype=float); t3v = t3v[np.isfinite(t3v)]
        if t1v.size > 0 and t3v.size > 0:
            idx1 = rng.integers(0, t1v.size, size=(n_boot, t1v.size))
            idx3 = rng.integers(0, t3v.size, size=(n_boot, t3v.size))
            diffs = t3v[idx3].mean(axis=1) - t1v[idx1].mean(axis=1)
            alpha = 0.025
            d13_ci = (float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha)))
    except Exception:
        pass

    # Permutation p-values for paired changes
    p12 = _paired_permutation_pvalue(t2, t1, n_perm=n_perm, seed=21)
    p23 = _paired_permutation_pvalue(df_23["confidence_tier3"].to_numpy(), df_23["confidence_tier2"].to_numpy(), n_perm=n_perm, seed=22)

    trajectory = "up" if t3_mean > t1_mean else ("down" if t3_mean < t1_mean else "flat")

    out = pd.DataFrame([{
        "tier1_mean": t1_mean,
        "tier1_ci_low": t1_ci[0],
        "tier1_ci_high": t1_ci[1],
        "tier2_mean": t2_mean,
        "tier2_ci_low": t2_ci[0],
        "tier2_ci_high": t2_ci[1],
        "tier3_mean": t3_mean,
        "tier3_ci_low": t3_ci[0],
        "tier3_ci_high": t3_ci[1],
        "delta_12_mean": d12_mean,
        "delta_12_ci_low": d12_ci[0],
        "delta_12_ci_high": d12_ci[1],
        "delta_12_perm_pvalue": p12,
        "delta_23_mean": d23_mean,
        "delta_23_ci_low": d23_ci[0],
        "delta_23_ci_high": d23_ci[1],
        "delta_23_perm_pvalue": p23,
        "delta_13_mean": d13_mean,
        "delta_13_ci_low": d13_ci[0],
        "delta_13_ci_high": d13_ci[1],
        "trajectory_t1_to_t3": trajectory,
        "notes": "CIs are percentile bootstrap (means resampled; paired resampling for deltas). p-values are paired permutation tests on mean delta."
    }])
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "confidence_trajectory_summary_ff.csv"), index=False)
    print(f"Saved confidence trajectory summary (CIs + permutation p-values) to {output_dir}/confidence_trajectory_summary_ff.csv")

def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def discover_models(experiment_results_dir):
    """Return model_ids as subdirectories in experiment_results_dir."""
    if not os.path.isdir(experiment_results_dir):
        return []
    model_ids = []
    for name in sorted(os.listdir(experiment_results_dir)):
        p = os.path.join(experiment_results_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            model_ids.append(name)
    return model_ids

def find_tier_csv(model_dir, tier):
    """Find a tier CSV within a model directory (tolerates naming differences)."""
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

def load_data(tier1_path, tier2_path, tier3_path):
    """
    Load all three tier datasets.
    
    Args:
        tier1_path: Path to Tier 1 CSV (no risk factors, binary)
        tier2_path: Path to Tier 2 CSV (with risk factors, binary)
        tier3_path: Path to Tier 3 CSV (with risk factors, 3-level)
    
    Returns:
        Tuple of three DataFrames (df_tier1, df_tier2, df_tier3)
    """
    df_tier1 = pd.read_csv(tier1_path)
    df_tier2 = pd.read_csv(tier2_path)
    df_tier3 = pd.read_csv(tier3_path)
    
    # Clean trailing periods from dosage column
    for df in [df_tier1, df_tier2, df_tier3]:
        df['gpt4o_dosage'] = df['gpt4o_dosage'].str.replace('.', '', regex=False).str.strip()
    
    print(f"Loaded Tier 1: {len(df_tier1)} records")
    print(f"Loaded Tier 2: {len(df_tier2)} records")
    print(f"Loaded Tier 3: {len(df_tier3)} records")
    
    return df_tier1, df_tier2, df_tier3


def extract_confidence_tier1_tier2(df, tier_name):
    """
    Extract confidence in the chosen dosage for Tier 1 or Tier 2 (binary choice).
    
    For binary dosage (Low/High), confidence = probability of chosen dosage.
    """
    df = df.copy()
    
    # Determine which dosage was chosen (robust to missing/NaN)
    df['dosage_clean'] = df['gpt4o_dosage'].fillna('').astype(str).str.lower().str.strip()
    
    # Extract confidence based on chosen dosage
    def get_confidence(row):
        dosage = row['dosage_clean']
        if 'low' in dosage:
            return row['prob_gpt4o_low']
        elif 'high' in dosage:
            return row['prob_gpt4o_high']
        else:
            return np.nan
    
    df['confidence'] = df.apply(get_confidence, axis=1)
    df['chosen_dosage'] = df['dosage_clean'].apply(
        lambda x: 'Low' if 'low' in x else ('High' if 'high' in x else 'Unknown')
    )
    
    print(f"\n{tier_name} Confidence Statistics:")
    print(f"  Mean: {df['confidence'].mean():.4f}")
    print(f"  Median: {df['confidence'].median():.4f}")
    print(f"  Std: {df['confidence'].std():.4f}")
    print(f"  Min: {df['confidence'].min():.4f}")
    print(f"  Max: {df['confidence'].max():.4f}")
    
    return df


def extract_confidence_tier3(df):
    """
    Extract confidence in the chosen dosage for Tier 3 (3-level choice).
    
    For 3-level dosage (Low/Medium/High/None), confidence = probability of chosen dosage.
    """
    df = df.copy()
    
    # Determine which dosage was chosen (robust to missing/NaN)
    df['dosage_clean'] = df['gpt4o_dosage'].fillna('').astype(str).str.lower().str.strip()
    
    # Extract confidence based on chosen dosage
    def get_confidence(row):
        dosage = row['dosage_clean']
        if 'none' in dosage:
            return row['prob_gpt4o_none']
        elif 'low' in dosage:
            return row['prob_gpt4o_low']
        elif 'medium' in dosage:
            return row['prob_gpt4o_medium']
        elif 'high' in dosage:
            return row['prob_gpt4o_high']
        else:
            return np.nan
    
    df['confidence'] = df.apply(get_confidence, axis=1)
    df['chosen_dosage'] = df['dosage_clean'].apply(
        lambda x: 'None' if 'none' in x else (
            'Low' if 'low' in x else (
                'Medium' if 'medium' in x else (
                    'High' if 'high' in x else 'Unknown'
                )
            )
        )
    )
    
    print(f"\nTier 3 Confidence Statistics:")
    print(f"  Mean: {df['confidence'].mean():.4f}")
    print(f"  Median: {df['confidence'].median():.4f}")
    print(f"  Std: {df['confidence'].std():.4f}")
    print(f"  Min: {df['confidence'].min():.4f}")
    print(f"  Max: {df['confidence'].max():.4f}")
    
    return df


def compare_tier1_tier2(df_tier1, df_tier2, output_dir='results'):
    """
    Compare confidence between Tier 1 (demographics only) and Tier 2 (+ risk factors).
    
    Question: Does adding clinical complexity reduce confidence?
    
    Note: Since Tier 1 doesn't have risk factors, we can only match on vignette_idx + race + gender.
    This means we'll compare average confidence changes across all risk factor combinations.
    """
    print("\n" + "="*80)
    print("COMPARISON A: Tier 1 → Tier 2 (Adding Risk Factors)")
    print("="*80)
    
    # Create matching keys (demographics only, since Tier 1 has no risk factors)
    df_tier1['match_key'] = (df_tier1['vignette_idx'].astype(str) + '_' + 
                              df_tier1['race'].astype(str) + '_' + 
                              df_tier1['gender'].astype(str))
    
    df_tier2['match_key'] = (df_tier2['vignette_idx'].astype(str) + '_' + 
                              df_tier2['race'].astype(str) + '_' + 
                              df_tier2['gender'].astype(str))
    
    # Merge
    df_merged = df_tier2[['match_key', 'vignette_idx', 'race', 'gender', 
                          'risk_mh', 'risk_op', 'risk_pain', 
                          'confidence', 'chosen_dosage']].copy()
    df_merged.columns = ['match_key', 'vignette_idx', 'race', 'gender',
                         'risk_mh', 'risk_op', 'risk_pain',
                         'confidence_tier2', 'chosen_dosage_tier2']
    
    df_tier1_subset = df_tier1[['match_key', 'confidence', 'chosen_dosage']].copy()
    df_tier1_subset.columns = ['match_key', 'confidence_tier1', 'chosen_dosage_tier1']
    
    df_comparison = df_merged.merge(df_tier1_subset, on='match_key', how='inner')
    
    if len(df_comparison) == 0:
        print("\nWARNING: No matching records found between Tier 1 and Tier 2!")
        return None
    
    print(f"\nMatched {len(df_comparison)} records between Tier 1 and Tier 2")
    
    # Calculate confidence drop
    df_comparison['confidence_drop'] = df_comparison['confidence_tier2'] - df_comparison['confidence_tier1']
    
    # Overall statistics
    print(f"\nOverall Confidence Drop (Tier 2 - Tier 1):")
    print(f"  Mean: {df_comparison['confidence_drop'].mean():.4f}")
    print(f"  Median: {df_comparison['confidence_drop'].median():.4f}")
    print(f"  Std: {df_comparison['confidence_drop'].std():.4f}")
    
    # Statistical significance (paired permutation test) + CI (paired bootstrap)
    ci_low, ci_high = _bootstrap_ci_mean_of_diff(
        df_comparison['confidence_tier2'].to_numpy(),
        df_comparison['confidence_tier1'].to_numpy(),
        n_boot=5000,
        seed=101
    )
    p_perm = _paired_permutation_pvalue(
        df_comparison['confidence_tier2'].to_numpy(),
        df_comparison['confidence_tier1'].to_numpy(),
        n_perm=20000,
        seed=102
    )
    print("\nPaired change inference (Tier2 - Tier1):")
    print(f"  Mean Δ: {df_comparison['confidence_drop'].mean():.4f}")
    print(f"  95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Paired permutation p-value (two-sided): {p_perm:.3e}")
    
    # Analyze by risk factors
    print("\n--- Confidence Drop by Risk Factors ---")
    
    for risk_factor in ['risk_mh', 'risk_op', 'risk_pain']:
        print(f"\nBy {risk_factor}:")
        grouped = df_comparison.groupby(risk_factor)['confidence_drop'].agg(['mean', 'std', 'count'])
        print(grouped.to_string())
    
    # Analyze by demographics
    print("\n--- Confidence Drop by Demographics ---")
    
    for demo in ['race', 'gender']:
        print(f"\nBy {demo}:")
        grouped = df_comparison.groupby(demo)['confidence_drop'].agg(['mean', 'std', 'count'])
        print(grouped.to_string())
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    df_comparison.to_csv(f'{output_dir}/confidence_drop_tier1_to_tier2_ff.csv', index=False)
    print(f"\nSaved detailed results to {output_dir}/confidence_drop_tier1_to_tier2_ff.csv")
    
    return df_comparison


def compare_tier2_tier3(df_tier2, df_tier3, output_dir='results'):
    """
    Compare confidence between Tier 2 (binary) and Tier 3 (3-level).
    
    Question: Does adding dosage granularity reduce confidence in the chosen option?
    
    Match on: vignette_idx + race + gender + risk factors
    """
    print("\n" + "="*80)
    print("COMPARISON B: Tier 2 → Tier 3 (Adding Dosage Granularity)")
    print("="*80)
    
    # Create matching keys (demographics + risk factors)
    for df in [df_tier2, df_tier3]:
        df['match_key'] = (df['vignette_idx'].astype(str) + '_' + 
                           df['race'].astype(str) + '_' + 
                           df['gender'].astype(str) + '_' +
                           df['risk_op'].astype(str) + '_' +
                           df['risk_mh'].astype(str) + '_' +
                           df['risk_pain'].astype(str))
    
    # Merge
    df_merged = df_tier3[['match_key', 'vignette_idx', 'race', 'gender',
                          'risk_mh', 'risk_op', 'risk_pain',
                          'confidence', 'chosen_dosage']].copy()
    df_merged.columns = ['match_key', 'vignette_idx', 'race', 'gender',
                         'risk_mh', 'risk_op', 'risk_pain',
                         'confidence_tier3', 'chosen_dosage_tier3']
    
    df_tier2_subset = df_tier2[['match_key', 'confidence', 'chosen_dosage']].copy()
    df_tier2_subset.columns = ['match_key', 'confidence_tier2', 'chosen_dosage_tier2']
    
    df_comparison = df_merged.merge(df_tier2_subset, on='match_key', how='inner')
    
    if len(df_comparison) == 0:
        print("\nWARNING: No matching records found between Tier 2 and Tier 3!")
        print("This is expected if risk factors were randomized independently.")
        print("For full factorial experiments, all combinations should match.")
        return None
    
    print(f"\nMatched {len(df_comparison)} records between Tier 2 and Tier 3")
    
    # Calculate confidence drop
    df_comparison['confidence_drop'] = df_comparison['confidence_tier3'] - df_comparison['confidence_tier2']
    
    # Identify decision consistency
    df_comparison['decision_changed'] = (
        df_comparison['chosen_dosage_tier2'] != df_comparison['chosen_dosage_tier3']
    )
    
    # Overall statistics
    print(f"\nOverall Confidence Drop (Tier 3 - Tier 2):")
    print(f"  Mean: {df_comparison['confidence_drop'].mean():.4f}")
    print(f"  Median: {df_comparison['confidence_drop'].median():.4f}")
    print(f"  Std: {df_comparison['confidence_drop'].std():.4f}")
    
    # Statistical significance (paired permutation test) + CI (paired bootstrap)
    ci_low, ci_high = _bootstrap_ci_mean_of_diff(
        df_comparison['confidence_tier3'].to_numpy(),
        df_comparison['confidence_tier2'].to_numpy(),
        n_boot=5000,
        seed=201
    )
    p_perm = _paired_permutation_pvalue(
        df_comparison['confidence_tier3'].to_numpy(),
        df_comparison['confidence_tier2'].to_numpy(),
        n_perm=20000,
        seed=202
    )
    print("\nPaired change inference (Tier3 - Tier2):")
    print(f"  Mean Δ: {df_comparison['confidence_drop'].mean():.4f}")
    print(f"  95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Paired permutation p-value (two-sided): {p_perm:.3e}")
    
    # Analyze by decision consistency
    print("\n--- Confidence Drop by Decision Consistency ---")
    for changed in [True, False]:
        subset = df_comparison[df_comparison['decision_changed'] == changed]
        if len(subset) > 0:
            label = "Decision Changed" if changed else "Decision Stayed Same"
            print(f"\n{label} (n={len(subset)}):")
            print(f"  Mean confidence drop: {subset['confidence_drop'].mean():.4f}")
            print(f"  Median confidence drop: {subset['confidence_drop'].median():.4f}")
    
    # Analyze by risk factors
    print("\n--- Confidence Drop by Risk Factors ---")
    
    for risk_factor in ['risk_mh', 'risk_op', 'risk_pain']:
        print(f"\nBy {risk_factor}:")
        grouped = df_comparison.groupby(risk_factor)['confidence_drop'].agg(['mean', 'std', 'count'])
        print(grouped.to_string())
    
    # Analyze by demographics
    print("\n--- Confidence Drop by Demographics ---")
    
    for demo in ['race', 'gender']:
        print(f"\nBy {demo}:")
        grouped = df_comparison.groupby(demo)['confidence_drop'].agg(['mean', 'std', 'count'])
        print(grouped.to_string())
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    df_comparison.to_csv(f'{output_dir}/confidence_drop_tier2_to_tier3_ff.csv', index=False)
    print(f"\nSaved detailed results to {output_dir}/confidence_drop_tier2_to_tier3_ff.csv")
    
    return df_comparison


def compare_tier1_choice_probability_in_tier2(df_tier1, df_tier2, output_dir='results'):
    """
    Compare the probability of Tier 1's chosen dosage when evaluated in Tier 2.
    
    Question: When the model chose "Low" in Tier 1 (demographics only), 
    what probability does it assign to "Low" in Tier 2 (with risk factors)?
    
    This isolates the effect of adding risk factors on the original choice's 
    probability, separate from whether the final decision changed.
    
    Args:
        df_tier1: Tier 1 data with chosen dosage and confidence
        df_tier2: Tier 2 data with probabilities for Low and High
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("ANALYSIS C1: Tier 1 Choice Probability in Tier 2")
    print("="*80)
    print("\nQuestion: When considering risk factors (Tier 2), what happens to")
    print("the probability of the dosage chosen based on demographics alone (Tier 1)?")
    print("="*80)
    
    # Create matching keys (demographics only, since Tier 1 has no risk factors)
    df_tier1['match_key_demo'] = (df_tier1['vignette_idx'].astype(str) + '_' + 
                                   df_tier1['race'].astype(str) + '_' + 
                                   df_tier1['gender'].astype(str))
    
    df_tier2['match_key_demo'] = (df_tier2['vignette_idx'].astype(str) + '_' + 
                                   df_tier2['race'].astype(str) + '_' + 
                                   df_tier2['gender'].astype(str))
    
    # Merge - keep Tier 2's risk factors for stratification
    df_merged = df_tier2[['match_key_demo', 'vignette_idx', 'race', 'gender',
                          'risk_mh', 'risk_op', 'risk_pain',
                          'chosen_dosage', 'prob_gpt4o_low', 'prob_gpt4o_high']].copy()
    df_merged.columns = ['match_key_demo', 'vignette_idx', 'race', 'gender',
                         'risk_mh', 'risk_op', 'risk_pain',
                         'tier2_chosen', 'tier2_prob_low', 'tier2_prob_high']
    
    df_tier1_subset = df_tier1[['match_key_demo', 'chosen_dosage', 'confidence']].copy()
    df_tier1_subset.columns = ['match_key_demo', 'tier1_chosen', 'tier1_confidence']
    
    df_comparison = df_merged.merge(df_tier1_subset, on='match_key_demo', how='inner')
    
    if len(df_comparison) == 0:
        print("\nWARNING: No matching records found between Tier 1 and Tier 2!")
        return None
    
    print(f"\nMatched {len(df_comparison)} records between Tier 1 and Tier 2")
    
    # Extract Tier 2 probability for the dosage chosen in Tier 1
    def get_tier2_prob_for_tier1_choice(row):
        tier1_choice = row['tier1_chosen'].lower()
        if 'low' in tier1_choice:
            return row['tier2_prob_low']
        elif 'high' in tier1_choice:
            return row['tier2_prob_high']
        else:
            return np.nan
    
    df_comparison['tier2_prob_of_tier1_choice'] = df_comparison.apply(
        get_tier2_prob_for_tier1_choice, axis=1
    )
    
    # Calculate probability drop
    df_comparison['prob_drop'] = (df_comparison['tier2_prob_of_tier1_choice'] - 
                                  df_comparison['tier1_confidence'])
    
    # Identify if decision changed
    df_comparison['decision_changed'] = (
        df_comparison['tier1_chosen'].str.lower() != 
        df_comparison['tier2_chosen'].str.lower()
    )
    
    # Overall statistics
    print(f"\nOverall Probability Drop (Tier 2 P(Tier 1 choice) - Tier 1 Confidence):")
    print(f"  Mean: {df_comparison['prob_drop'].mean():.4f}")
    print(f"  Median: {df_comparison['prob_drop'].median():.4f}")
    print(f"  Std: {df_comparison['prob_drop'].std():.4f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(df_comparison['tier2_prob_of_tier1_choice'], 
                                       df_comparison['tier1_confidence'])
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    
    # Analyze by decision change
    print("\n--- Analysis by Decision Consistency ---")
    for changed in [True, False]:
        subset = df_comparison[df_comparison['decision_changed'] == changed]
        if len(subset) > 0:
            label = "Decision Changed" if changed else "Decision Stayed Same"
            print(f"\n{label} (n={len(subset)}):")
            print(f"  Mean prob drop: {subset['prob_drop'].mean():.4f}")
            if changed:
                switches = subset.groupby(['tier1_chosen', 'tier2_chosen']).size()
                if len(switches) > 0:
                    print(f"  Most common switch: {switches.idxmax()} (n={switches.max()})")
    
    # Analyze by risk factors (KEY INSIGHT)
    print("\n--- Analysis by Risk Factors (Why did probability change?) ---")
    
    for risk_factor in ['risk_mh', 'risk_op', 'risk_pain']:
        print(f"\nBy {risk_factor}:")
        grouped = df_comparison.groupby(risk_factor).agg({
            'prob_drop': ['mean', 'std', 'count'],
            'decision_changed': 'mean'
        }).round(4)
        grouped.columns = ['Mean Prob Drop', 'Std', 'Count', '% Decision Changed']
        print(grouped.to_string())
    
    # Analyze by Tier 1 choice
    print("\n--- Analysis by Tier 1 Choice ---")
    for choice in df_comparison['tier1_chosen'].unique():
        subset = df_comparison[df_comparison['tier1_chosen'] == choice]
        print(f"\n{choice} (n={len(subset)}):")
        print(f"  Mean prob drop: {subset['prob_drop'].mean():.4f}")
        print(f"  % stayed with {choice}: {((~subset['decision_changed']).sum() / len(subset) * 100):.1f}%")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df_comparison.to_csv(f'{output_dir}/tier1_choice_probability_in_tier2_ff.csv', index=False)
    print(f"\nSaved detailed results to {output_dir}/tier1_choice_probability_in_tier2_ff.csv")
    
    return df_comparison


def compare_tier2_choice_probability_in_tier3(df_tier2, df_tier3, output_dir='results'):
    """
    Compare the probability of Tier 2's chosen dosage when evaluated in Tier 3.
    
    Question: When the model chose "Low" in Tier 2 (binary), what probability 
    does it assign to "Low" in Tier 3 (when Medium is available)?
    
    This isolates the effect of adding dosage options on the original choice's 
    probability, separate from whether the final decision changed.
    
    Args:
        df_tier2: Tier 2 data with chosen dosage and confidence
        df_tier3: Tier 3 data with probabilities for all options
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("ANALYSIS C2: Tier 2 Choice Probability in Tier 3")
    print("="*80)
    print("\nQuestion: When given more dosage options (Tier 3), what happens to the")
    print("probability of the dosage the model originally chose (Tier 2)?")
    print("="*80)
    
    # Create matching keys (demographics + risk factors)
    for df in [df_tier2, df_tier3]:
        df['match_key_full'] = (df['vignette_idx'].astype(str) + '_' + 
                                df['race'].astype(str) + '_' + 
                                df['gender'].astype(str) + '_' +
                                df['risk_op'].astype(str) + '_' +
                                df['risk_mh'].astype(str) + '_' +
                                df['risk_pain'].astype(str))
    
    # Merge
    df_merged = df_tier2[['match_key_full', 'vignette_idx', 'race', 'gender',
                          'risk_mh', 'risk_op', 'risk_pain',
                          'chosen_dosage', 'confidence']].copy()
    df_merged.columns = ['match_key_full', 'vignette_idx', 'race', 'gender',
                         'risk_mh', 'risk_op', 'risk_pain',
                         'tier2_chosen', 'tier2_confidence']
    
    # Get Tier 3 probabilities for all options
    df_tier3_subset = df_tier3[['match_key_full', 'chosen_dosage',
                                'prob_gpt4o_none', 'prob_gpt4o_low', 
                                'prob_gpt4o_medium', 'prob_gpt4o_high']].copy()
    df_tier3_subset.columns = ['match_key_full', 'tier3_chosen',
                               'tier3_prob_none', 'tier3_prob_low',
                               'tier3_prob_medium', 'tier3_prob_high']
    
    df_comparison = df_merged.merge(df_tier3_subset, on='match_key_full', how='inner')
    
    if len(df_comparison) == 0:
        print("\nWARNING: No matching records found between Tier 2 and Tier 3!")
        print("This is expected if risk factors were randomized independently.")
        print("For full factorial experiments, all combinations should match.")
        return None
    
    print(f"\nMatched {len(df_comparison)} records between Tier 2 and Tier 3")
    
    # Extract Tier 3 probability for the dosage chosen in Tier 2
    def get_tier3_prob_for_tier2_choice(row):
        tier2_choice = row['tier2_chosen'].lower()
        if 'low' in tier2_choice:
            return row['tier3_prob_low']
        elif 'high' in tier2_choice:
            return row['tier3_prob_high']
        else:
            return np.nan
    
    df_comparison['tier3_prob_of_tier2_choice'] = df_comparison.apply(
        get_tier3_prob_for_tier2_choice, axis=1
    )
    
    # Calculate probability drop
    df_comparison['prob_drop'] = (df_comparison['tier3_prob_of_tier2_choice'] - 
                                  df_comparison['tier2_confidence'])
    
    # Identify if decision changed
    df_comparison['decision_changed'] = (
        df_comparison['tier2_chosen'].str.lower() != 
        df_comparison['tier3_chosen'].str.lower()
    )
    
    # Calculate probability mass transferred to Medium
    df_comparison['prob_mass_to_medium'] = df_comparison['tier3_prob_medium']
    
    # Overall statistics
    print(f"\nOverall Probability Drop (Tier 3 P(Tier 2 choice) - Tier 2 Confidence):")
    print(f"  Mean: {df_comparison['prob_drop'].mean():.4f}")
    print(f"  Median: {df_comparison['prob_drop'].median():.4f}")
    print(f"  Std: {df_comparison['prob_drop'].std():.4f}")
    print(f"\nMean probability mass transferred to Medium: {df_comparison['prob_mass_to_medium'].mean():.4f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(df_comparison['tier3_prob_of_tier2_choice'], 
                                       df_comparison['tier2_confidence'])
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    
    # Analyze by whether decision changed
    print("\n--- Analysis by Decision Consistency ---")
    for changed in [True, False]:
        subset = df_comparison[df_comparison['decision_changed'] == changed]
        if len(subset) > 0:
            label = "Decision Changed" if changed else "Decision Stayed Same"
            print(f"\n{label} (n={len(subset)}):")
            print(f"  Mean prob drop: {subset['prob_drop'].mean():.4f}")
            print(f"  Mean P(Medium): {subset['prob_mass_to_medium'].mean():.4f}")
            if changed:
                switched_to_medium = subset['tier3_chosen'].str.contains('Medium', case=False).sum()
                print(f"  % switched to Medium: {(switched_to_medium / len(subset) * 100):.1f}%")
    
    # Analyze by Tier 2 choice
    print("\n--- Analysis by Tier 2 Choice ---")
    for choice in df_comparison['tier2_chosen'].unique():
        subset = df_comparison[df_comparison['tier2_chosen'] == choice]
        print(f"\n{choice} (n={len(subset)}):")
        print(f"  Mean prob drop: {subset['prob_drop'].mean():.4f}")
        print(f"  Mean P(Medium) in Tier 3: {subset['prob_mass_to_medium'].mean():.4f}")
        print(f"  % stayed with {choice}: {((~subset['decision_changed']).sum() / len(subset) * 100):.1f}%")
    
    # Analyze by risk factors
    print("\n--- Analysis by Risk Factors ---")
    
    for risk_factor in ['risk_mh', 'risk_op', 'risk_pain']:
        print(f"\nBy {risk_factor}:")
        grouped = df_comparison.groupby(risk_factor).agg({
            'prob_drop': ['mean', 'std', 'count'],
            'prob_mass_to_medium': 'mean',
            'decision_changed': 'mean'
        }).round(4)
        grouped.columns = ['Mean Prob Drop', 'Std', 'Count', 'Mean P(Med)', '% Decision Changed']
        print(grouped.to_string())
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df_comparison.to_csv(f'{output_dir}/tier2_choice_probability_in_tier3_ff.csv', index=False)
    print(f"\nSaved detailed results to {output_dir}/tier2_choice_probability_in_tier3_ff.csv")
    
    return df_comparison


def create_confidence_distribution_plots(df_tier1, df_tier2, df_tier3, output_dir='results'):
    """
    Create histograms showing confidence distributions across all three tiers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Tier 1
    axes[0].hist(df_tier1['confidence'].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(df_tier1['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_tier1["confidence"].mean():.3f}')
    axes[0].set_xlabel('Confidence', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Tier 1: Demographics Only\n(Binary Dosage)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Tier 2
    axes[1].hist(df_tier2['confidence'].dropna(), bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1].axvline(df_tier2['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_tier2["confidence"].mean():.3f}')
    axes[1].set_xlabel('Confidence', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Tier 2: + Risk Factors\n(Binary Dosage)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Tier 3
    axes[2].hist(df_tier3['confidence'].dropna(), bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[2].axvline(df_tier3['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_tier3["confidence"].mean():.3f}')
    axes[2].set_xlabel('Confidence', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Tier 3: + Risk Factors\n(4-Level Dosage)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f'{output_dir}/confidence_distributions_all_tiers_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved confidence distribution plot to {save_path}")
    plt.close()


def create_confidence_drop_heatmap_tier1_to_tier2(df_comparison, output_dir='results'):
    """
    Create heatmap showing confidence drop from Tier 1 to Tier 2 by demographics.
    """
    if df_comparison is None or len(df_comparison) == 0:
        print("Skipping Tier 1→2 heatmap (no data)")
        return
    
    # Aggregate by race and gender
    pivot = df_comparison.groupby(['race', 'gender'])['confidence_drop'].mean().reset_index()
    pivot_table = pivot.pivot(index='race', columns='gender', values='confidence_drop')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine color scale range (symmetric around 0)
    max_abs = max(abs(pivot_table.min().min()), abs(pivot_table.max().max()))
    vmin, vmax = -max_abs, max_abs
    
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdBu_r',
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Mean Confidence Drop (Tier 2 - Tier 1)'},
                linewidths=1, linecolor='gray', ax=ax)
    
    ax.set_title('Confidence Drop: Tier 1 → Tier 2 (Adding Risk Factors)\n' +
                 'Blue = Confidence Decreased | Red = Confidence Increased',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Gender', fontweight='bold', fontsize=11)
    ax.set_ylabel('Race', fontweight='bold', fontsize=11)
    
    save_path = f'{output_dir}/confidence_drop_heatmap_tier1_to_tier2_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Tier 1→2 heatmap to {save_path}")
    plt.close()


def create_confidence_drop_heatmap_tier2_to_tier3(df_comparison, output_dir='results'):
    """
    Create heatmap showing confidence drop from Tier 2 to Tier 3 by demographics.
    """
    if df_comparison is None or len(df_comparison) == 0:
        print("Skipping Tier 2→3 heatmap (no data)")
        return
    
    # Aggregate by race and gender
    pivot = df_comparison.groupby(['race', 'gender'])['confidence_drop'].mean().reset_index()
    pivot_table = pivot.pivot(index='race', columns='gender', values='confidence_drop')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine color scale range (symmetric around 0)
    max_abs = max(abs(pivot_table.min().min()), abs(pivot_table.max().max()))
    vmin, vmax = -max_abs, max_abs
    
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdBu_r',
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Mean Confidence Drop (Tier 3 - Tier 2)'},
                linewidths=1, linecolor='gray', ax=ax)
    
    ax.set_title('Confidence Drop: Tier 2 → Tier 3 (Adding Dosage Granularity)\n' +
                 'Blue = Confidence Decreased | Red = Confidence Increased',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Gender', fontweight='bold', fontsize=11)
    ax.set_ylabel('Race', fontweight='bold', fontsize=11)
    
    save_path = f'{output_dir}/confidence_drop_heatmap_tier2_to_tier3_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Tier 2→3 heatmap to {save_path}")
    plt.close()


def create_confidence_drop_by_risk_factors(df_comparison, comparison_name, output_dir='results'):
    """
    Create bar plots showing confidence drop by each risk factor.
    """
    if df_comparison is None or len(df_comparison) == 0:
        print(f"Skipping risk factor plots for {comparison_name} (no data)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    risk_factors = [
        ('risk_mh', 'Mental Health'),
        ('risk_op', 'Opioid Status'),
        ('risk_pain', 'Preoperative Pain')
    ]
    
    for idx, (col, label) in enumerate(risk_factors):
        if col not in df_comparison.columns:
            continue
            
        grouped = df_comparison.groupby(col)['confidence_drop'].agg(['mean', 'sem']).reset_index()
        
        axes[idx].bar(range(len(grouped)), grouped['mean'], 
                      yerr=grouped['sem'], capsize=5,
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[idx].set_xticks(range(len(grouped)))
        axes[idx].set_xticklabels(grouped[col], rotation=45, ha='right')
        axes[idx].set_ylabel('Mean Confidence Drop', fontweight='bold')
        axes[idx].set_xlabel(label, fontweight='bold')
        axes[idx].set_title(f'Confidence Drop by {label}', fontweight='bold')
        axes[idx].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = f'{output_dir}/confidence_drop_by_risk_factors_{comparison_name}_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved risk factor plot to {save_path}")
    plt.close()


def create_confidence_drop_boxplot(df_tier1_to_tier2, df_tier2_to_tier3, output_dir='results'):
    """
    Create side-by-side boxplots comparing confidence drops across both transitions.
    """
    # Prepare data
    data_list = []
    
    if df_tier1_to_tier2 is not None and len(df_tier1_to_tier2) > 0:
        for _, row in df_tier1_to_tier2.iterrows():
            data_list.append({
                'Transition': 'Tier 1 → 2\n(+Risk Factors)',
                'Confidence Drop': row['confidence_drop']
            })
    
    if df_tier2_to_tier3 is not None and len(df_tier2_to_tier3) > 0:
        for _, row in df_tier2_to_tier3.iterrows():
            data_list.append({
                'Transition': 'Tier 2 → 3\n(+Dosage Granularity)',
                'Confidence Drop': row['confidence_drop']
            })
    
    if len(data_list) == 0:
        print("Skipping boxplot (no data)")
        return
    
    df_plot = pd.DataFrame(data_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df_plot, x='Transition', y='Confidence Drop', 
                palette=['steelblue', 'darkorange'], ax=ax)
    sns.stripplot(data=df_plot, x='Transition', y='Confidence Drop',
                  color='black', alpha=0.3, size=2, ax=ax)
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.set_ylabel('Confidence Drop', fontweight='bold', fontsize=12)
    ax.set_xlabel('Experimental Transition', fontweight='bold', fontsize=12)
    ax.set_title('Confidence Drop Across Experimental Tiers\n' +
                 'Negative = Confidence Decreased | Positive = Confidence Increased',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = f'{output_dir}/confidence_drop_boxplot_comparison_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved boxplot comparison to {save_path}")
    plt.close()


def _prob_drop_grouped_op_mh(df):
    """Mean ± SEM of prob_drop by opioid status and by mental health."""
    g_op = df.groupby('risk_op').agg({'prob_drop': ['mean', 'sem']}).reset_index()
    g_op.columns = ['risk_op', 'mean', 'sem']
    g_mh = df.groupby('risk_mh').agg({'prob_drop': ['mean', 'sem']}).reset_index()
    g_mh.columns = ['risk_mh', 'mean', 'sem']
    return g_op, g_mh


def _shared_ylim_mean_sem_bars(*groupeds, pad_frac=0.08):
    """
    Shared y-limits for any number of bar groupings (columns mean, sem).
    Used within one figure (opioid vs MH) or across Tier1→2 and Tier2→3 figures.
    """
    if not groupeds:
        return -0.1, 0.1

    def _extents(g):
        sem = g['sem'].fillna(0)
        low = float((g['mean'] - sem).min())
        high = float((g['mean'] + sem).max())
        return low, high

    lows, highs = [], []
    for g in groupeds:
        lo, hi = _extents(g)
        lows.append(lo)
        highs.append(hi)
    lo = min(min(lows), 0.0)
    hi = max(max(highs), 0.0)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = 0.1
    pad = span * pad_frac
    return lo - pad, hi + pad


def create_tier1_choice_probability_plot(df_comparison, output_dir='results', ylim_bar_panels=None):
    """
    Visualize how Tier 1 choice probabilities change in Tier 2.
    Shows the effect of adding risk factors on original choice probability.
    """
    if df_comparison is None or len(df_comparison) == 0:
        print("Skipping Tier 1 choice probability plot (no data)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Scatter plot - decision change
    ax1 = axes[0, 0]
    
    stayed = df_comparison[~df_comparison['decision_changed']]
    changed = df_comparison[df_comparison['decision_changed']]
    
    ax1.scatter(stayed['tier1_confidence'], 
               stayed['tier2_prob_of_tier1_choice'],
               s=100, alpha=0.6, c='steelblue', 
               edgecolors='black', linewidth=1,
               label=f'Decision Stayed (n={len(stayed)})')
    
    ax1.scatter(changed['tier1_confidence'], 
               changed['tier2_prob_of_tier1_choice'],
               s=100, alpha=0.6, c='coral', 
               edgecolors='black', linewidth=1,
               label=f'Decision Changed (n={len(changed)})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No Change')
    
    ax1.set_xlabel('Tier 1 Confidence\n(Demographics Only)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Tier 2 P(Tier 1 Choice)\n(With Risk Factors)', fontweight='bold', fontsize=11)
    ax1.set_title('Tier 1→2: How Risk Factors Affect\nOriginal Choice Probability',
                  fontweight='bold', fontsize=12, pad=10)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Box plot by Tier 1 choice
    ax2 = axes[0, 1]
    
    plot_data = []
    for _, row in df_comparison.iterrows():
        plot_data.append({
            'Tier 1 Choice': row['tier1_chosen'],
            'Probability Drop': row['prob_drop'],
            'Decision': 'Changed' if row['decision_changed'] else 'Same'
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    sns.boxplot(data=df_plot, x='Tier 1 Choice', y='Probability Drop',
                hue='Decision', palette=['steelblue', 'coral'], ax=ax2)
    
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tier 1 Chosen Dosage', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Probability Drop (Tier 2 - Tier 1)', fontweight='bold', fontsize=11)
    ax2.set_title('Probability Drop by Tier 1 Choice',
                  fontweight='bold', fontsize=12, pad=10)
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(title='Final Decision')
    
    # Plot 3: Probability drop by opioid status
    ax3 = axes[1, 0]
    
    grouped_op = df_comparison.groupby('risk_op').agg({
        'prob_drop': ['mean', 'sem']
    }).reset_index()
    grouped_op.columns = ['risk_op', 'mean', 'sem']
    
    ax3.bar(range(len(grouped_op)), grouped_op['mean'], 
            yerr=grouped_op['sem'], capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xticks(range(len(grouped_op)))
    ax3.set_xticklabels(grouped_op['risk_op'], rotation=45, ha='right')
    ax3.set_ylabel('Mean Probability Drop', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Opioid Status', fontweight='bold', fontsize=11)
    ax3.set_title('Effect of Opioid Status on Probability Drop',
                  fontweight='bold', fontsize=12, pad=10)
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Probability drop by mental health
    ax4 = axes[1, 1]
    
    grouped_mh = df_comparison.groupby('risk_mh').agg({
        'prob_drop': ['mean', 'sem']
    }).reset_index()
    grouped_mh.columns = ['risk_mh', 'mean', 'sem']
    
    ax4.bar(range(len(grouped_mh)), grouped_mh['mean'], 
            yerr=grouped_mh['sem'], capsize=5,
            color='darkorange', alpha=0.7, edgecolor='black')
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xticks(range(len(grouped_mh)))
    ax4.set_xticklabels(grouped_mh['risk_mh'], rotation=45, ha='right')
    ax4.set_ylabel('Mean Probability Drop', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Mental Health Status', fontweight='bold', fontsize=11)
    ax4.set_title('Effect of Mental Health on Probability Drop',
                  fontweight='bold', fontsize=12, pad=10)
    ax4.grid(alpha=0.3, axis='y')

    if ylim_bar_panels is not None:
        y_lo, y_hi = ylim_bar_panels
    else:
        y_lo, y_hi = _shared_ylim_mean_sem_bars(grouped_op, grouped_mh)
    ax3.set_ylim(y_lo, y_hi)
    ax4.set_ylim(y_lo, y_hi)
    
    plt.tight_layout()
    
    save_path = f'{output_dir}/tier1_choice_probability_in_tier2_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Tier 1 choice probability plot to {save_path}")
    plt.close()


def create_tier2_choice_probability_plot(df_comparison, output_dir='results', ylim_bar_panels=None):
    """
    Visualize how Tier 2 choice probabilities change in Tier 3.
    Fixed: Variable names now match the output of Analysis C2.
    """
    if df_comparison is None or len(df_comparison) == 0:
        print("Skipping Tier 2 choice probability plot (no data)")
        return
    
    # Check required columns exist
    required_cols = ['tier2_confidence', 'tier3_prob_of_tier2_choice', 'decision_changed', 
                     'prob_drop', 'tier2_chosen', 'risk_op', 'risk_mh']
    missing_cols = [col for col in required_cols if col not in df_comparison.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df_comparison.columns)}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Scatter plot with decision change
    ax1 = axes[0, 0]
    
    stayed = df_comparison[~df_comparison['decision_changed']]
    changed = df_comparison[df_comparison['decision_changed']]
    
    ax1.scatter(stayed['tier2_confidence'], 
               stayed['tier3_prob_of_tier2_choice'],
               s=100, alpha=0.6, c='steelblue', 
               edgecolors='black', linewidth=1,
               label=f'Decision Stayed (n={len(stayed)})')
    
    ax1.scatter(changed['tier2_confidence'], 
               changed['tier3_prob_of_tier2_choice'],
               s=100, alpha=0.6, c='coral', 
               edgecolors='black', linewidth=1,
               label=f'Decision Changed (n={len(changed)})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No Change')
    ax1.set_xlabel('Tier 2 Confidence (Binary Choice)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Tier 3 P(Tier 2 Choice)\n(With Medium Option)', fontweight='bold', fontsize=11)
    ax1.set_title('Tier 2→3: How Dosage Granularity Affects\nConfidence', fontweight='bold', fontsize=12, pad=10)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Box plot by Tier 2 choice
    ax2 = axes[0, 1]
    
    plot_data = []
    for _, row in df_comparison.iterrows():
        plot_data.append({
            'Tier 2 Choice': row['tier2_chosen'],
            'Probability Drop': row['prob_drop'],
            'Decision': 'Changed' if row['decision_changed'] else 'Same'
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    sns.boxplot(data=df_plot, x='Tier 2 Choice', y='Probability Drop',
                hue='Decision', palette=['steelblue', 'coral'], ax=ax2)
    
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tier 2 Chosen Dosage', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Probability Drop (Tier 3 - Tier 2)', fontweight='bold', fontsize=11)
    ax2.set_title('Probability Drop by Tier 2 Choice',
                  fontweight='bold', fontsize=12, pad=10)
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend(title='Final Decision')
    
    # Plot 3: Probability drop by opioid status
    ax3 = axes[1, 0]
    
    grouped_op = df_comparison.groupby('risk_op').agg({
        'prob_drop': ['mean', 'sem']
    }).reset_index()
    grouped_op.columns = ['risk_op', 'mean', 'sem']
    
    ax3.bar(range(len(grouped_op)), grouped_op['mean'], 
            yerr=grouped_op['sem'], capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xticks(range(len(grouped_op)))
    ax3.set_xticklabels(grouped_op['risk_op'], rotation=45, ha='right')
    ax3.set_ylabel('Mean Probability Drop', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Opioid Status', fontweight='bold', fontsize=11)
    ax3.set_title('Effect of Opioid Status on Probability Drop',
                  fontweight='bold', fontsize=12, pad=10)
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Probability drop by mental health
    ax4 = axes[1, 1]
    
    grouped_mh = df_comparison.groupby('risk_mh').agg({
        'prob_drop': ['mean', 'sem']
    }).reset_index()
    grouped_mh.columns = ['risk_mh', 'mean', 'sem']
    
    ax4.bar(range(len(grouped_mh)), grouped_mh['mean'], 
            yerr=grouped_mh['sem'], capsize=5,
            color='darkorange', alpha=0.7, edgecolor='black')
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xticks(range(len(grouped_mh)))
    ax4.set_xticklabels(grouped_mh['risk_mh'], rotation=45, ha='right')
    ax4.set_ylabel('Mean Probability Drop', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Mental Health Status', fontweight='bold', fontsize=11)
    ax4.set_title('Effect of Mental Health on Probability Drop',
                  fontweight='bold', fontsize=12, pad=10)
    ax4.grid(alpha=0.3, axis='y')

    if ylim_bar_panels is not None:
        y_lo, y_hi = ylim_bar_panels
    else:
        y_lo, y_hi = _shared_ylim_mean_sem_bars(grouped_op, grouped_mh)
    ax3.set_ylim(y_lo, y_hi)
    ax4.set_ylim(y_lo, y_hi)
    
    plt.tight_layout()
    save_path = f'{output_dir}/tier2_choice_probability_in_tier3_ff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("CONFIDENCE DROP ANALYSIS")
    print("="*80)
    
    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    analysis_results_base = os.path.join(root, "analysis_results", "confidence_drop")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"\nWARNING: No model folders found under: {experiment_results_dir}")
        return

    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        output_dir = os.path.join(analysis_results_base, model_id)
        os.makedirs(output_dir, exist_ok=True)

        # File paths (auto-discovered within model folder)
        tier1_path = find_tier_csv(model_dir, 1)
        tier2_path = find_tier_csv(model_dir, 2)
        tier3_path = find_tier_csv(model_dir, 3)

        print("\n" + "="*80)
        print(f"MODEL: {model_id}")
        print("="*80)
        print(f"Tier 1 input: {tier1_path}")
        print(f"Tier 2 input: {tier2_path}")
        print(f"Tier 3 input: {tier3_path}")
        print(f"Output dir:   {output_dir}")

        if not (tier1_path and tier2_path and tier3_path):
            print("WARNING: Missing one or more tier CSVs (need Tier 1/2/3). Skipping this model.")
            continue

        # Load data
        df_tier1, df_tier2, df_tier3 = load_data(tier1_path, tier2_path, tier3_path)
        
        # Extract confidence for each tier
        df_tier1 = extract_confidence_tier1_tier2(df_tier1, "Tier 1")
        df_tier2 = extract_confidence_tier1_tier2(df_tier2, "Tier 2")
        df_tier3 = extract_confidence_tier3(df_tier3)
        
        # Create overall confidence distribution plots
        create_confidence_distribution_plots(df_tier1, df_tier2, df_tier3, output_dir)
    
    # ========================================================================
    # TYPE A ANALYSES: Confidence in Chosen Dosage
    # ========================================================================
    
    # Comparison A1: Tier 1 → Tier 2 (Adding Risk Factors)
        df_tier1_to_tier2_A = compare_tier1_tier2(df_tier1, df_tier2, output_dir)
    
        if df_tier1_to_tier2_A is not None:
            create_confidence_drop_heatmap_tier1_to_tier2(df_tier1_to_tier2_A, output_dir)
            create_confidence_drop_by_risk_factors(df_tier1_to_tier2_A, 'tier1_to_tier2', output_dir)
    
    # Comparison A2: Tier 2 → Tier 3 (Adding Dosage Granularity)
        df_tier2_to_tier3_A = compare_tier2_tier3(df_tier2, df_tier3, output_dir)
    
        if df_tier2_to_tier3_A is not None:
            create_confidence_drop_heatmap_tier2_to_tier3(df_tier2_to_tier3_A, output_dir)
            create_confidence_drop_by_risk_factors(df_tier2_to_tier3_A, 'tier2_to_tier3', output_dir)
    
    # Combined boxplot for Type A
        create_confidence_drop_boxplot(df_tier1_to_tier2_A, df_tier2_to_tier3_A, output_dir)

        # Save a compact trajectory summary (means + CIs + permutation p-values)
        _save_confidence_trajectory_stats(df_tier1_to_tier2_A, df_tier2_to_tier3_A, output_dir)
    
    # ========================================================================
    # TYPE C ANALYSES: Probability of Original Choice
    # ========================================================================
    
    # Comparison C1: Tier 1 Choice Probability in Tier 2
        df_tier1_to_tier2_C = compare_tier1_choice_probability_in_tier2(df_tier1, df_tier2, output_dir)
    
    # Comparison C2: Tier 2 Choice Probability in Tier 3
        df_tier2_to_tier3_C = compare_tier2_choice_probability_in_tier3(df_tier2, df_tier3, output_dir)

        # Same y-axis on bar subplots (panels 3–4) across both Type-C figures
        ylim_bar_panels_cross = None
        if df_tier1_to_tier2_C is not None and df_tier2_to_tier3_C is not None:
            g12_op, g12_mh = _prob_drop_grouped_op_mh(df_tier1_to_tier2_C)
            g23_op, g23_mh = _prob_drop_grouped_op_mh(df_tier2_to_tier3_C)
            ylim_bar_panels_cross = _shared_ylim_mean_sem_bars(
                g12_op, g12_mh, g23_op, g23_mh
            )

        if df_tier1_to_tier2_C is not None:
            create_tier1_choice_probability_plot(
                df_tier1_to_tier2_C, output_dir, ylim_bar_panels=ylim_bar_panels_cross
            )

        if df_tier2_to_tier3_C is not None:
            create_tier2_choice_probability_plot(
                df_tier2_to_tier3_C, output_dir, ylim_bar_panels=ylim_bar_panels_cross
            )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
        print("\n" + "="*80)
        print(f"ANALYSIS COMPLETE FOR MODEL: {model_id}")
        print("="*80)
        print(f"\nAll results saved to {output_dir}/")
    print("\nGenerated files:")
    print("\n  Confidence Distributions:")
    print("    - confidence_distributions_all_tiers_ff.png")
    print("\n  Type A Analyses (Confidence in Chosen Dosage):")
    print("    - confidence_drop_tier1_to_tier2_ff.csv")
    print("    - confidence_drop_heatmap_tier1_to_tier2_ff.png")
    print("    - confidence_drop_by_risk_factors_tier1_to_tier2_ff.png")
    print("    - confidence_drop_tier2_to_tier3_ff.csv")
    print("    - confidence_drop_heatmap_tier2_to_tier3_ff.png")
    print("    - confidence_drop_by_risk_factors_tier2_to_tier3_ff.png")
    print("    - confidence_drop_boxplot_comparison_ff.png")
    print("\n  Type C Analyses (Probability of Original Choice):")
    print("    - tier1_choice_probability_in_tier2_ff.csv")
    print("    - tier1_choice_probability_in_tier2_ff.png")
    print("    - tier2_choice_probability_in_tier3_ff.csv")
    print("    - tier2_choice_probability_in_tier3_ff.png")
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("  Type A: Measures overall confidence changes (what was ultimately chosen)")
    print("  Type C: Measures how original preferences weaken/strengthen")
    print("="*80)


if __name__ == "__main__":
    main()

