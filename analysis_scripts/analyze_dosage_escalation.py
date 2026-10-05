import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import glob

# --- Paired escalation/de-escalation figure (edit sizes, then re-run script) ---
PAIRED_ESCALATION_PLOT_FONTS = {
    "suptitle": 14,        # fig.suptitle
    "panel_title": 12,   # "By gender" / "By race"
    "axis_label": 14,    # Gender, Race, Rate (%)
    "tick": 15,          # x category names + y-axis numbers
    "bar_annotate": 9,   # n= above bars
    "legend": 11,        # Escalation / De-escalation legend
}


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def _bh_fdr(pvals: np.ndarray):
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(pvals))
    if m == 0:
        return pvals
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]
    ranks = np.arange(1, len(sorted_p) + 1)
    adj = sorted_p * (len(sorted_p) / ranks)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[idx] = adj
    return out

def _two_proportion_ztest(k1: int, n1: int, k2: int, n2: int):
    """Two-sided z-test p-value for difference in proportions."""
    if n1 <= 0 or n2 <= 0:
        return np.nan
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 1.0 if p1 == p2 else 0.0
    z = (p1 - p2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))

def _fisher_exact_pvalue(k1: int, n1: int, k2: int, n2: int):
    """Two-sided Fisher exact p-value for a 2x2 table."""
    if n1 <= 0 or n2 <= 0:
        return np.nan
    a = int(k1)
    b = int(n1 - k1)
    c = int(k2)
    d = int(n2 - k2)
    if min(a, b, c, d) < 0:
        return np.nan
    _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)

def global_rate_chi2_by_group(df: pd.DataFrame, metric_col: str, group_col: str, opioid_status: str | None = None):
    """
    Global chi-squared test: group_col × outcome (0/1) association.
    Returns dict with chi2, p_value, dof, cramers_v.
    """
    d = df.copy()
    if opioid_status is not None:
        d = d[d["opioid_status"] == opioid_status].copy()
    if len(d) == 0:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan}
    if group_col not in d.columns:
        raise KeyError(f"group_col '{group_col}' not found in df columns: {sorted(d.columns)}")
    contingency = pd.crosstab(d[metric_col].astype(int), d[group_col].astype(str))
    # Ensure both 0 and 1 rows exist
    for v in (0, 1):
        if v not in contingency.index:
            contingency.loc[v] = 0
    contingency = contingency.sort_index()
    try:
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    except ValueError as e:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan, "error": str(e)}
    n = contingency.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else np.nan
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof), "cramers_v": float(cramers_v)}

def global_rate_chi2(df: pd.DataFrame, metric_col: str, opioid_status: str | None = None):
    """
    Global chi-squared test: subgroup × outcome (0/1) association.
    Returns dict with chi2, p_value, dof, cramers_v.
    """
    d = df.copy()
    if opioid_status is not None:
        d = d[d["opioid_status"] == opioid_status].copy()
    if len(d) == 0:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan}
    d["subgroup"] = d["race"].astype(str) + "_" + d["gender"].astype(str)
    contingency = pd.crosstab(d[metric_col].astype(int), d["subgroup"])
    # Ensure both 0 and 1 rows exist
    for v in (0, 1):
        if v not in contingency.index:
            contingency.loc[v] = 0
    contingency = contingency.sort_index()
    try:
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    except ValueError as e:
        # Can occur when expected frequencies contain zeros (extreme sparsity).
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan, "error": str(e)}
    n = contingency.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else np.nan
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof), "cramers_v": float(cramers_v)}

def gender_difference_test(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Hypothesis test for difference in rates between genders (2-group).
    Outputs both a two-proportion z-test p-value and Fisher exact p-value.
    """
    if len(df) == 0:
        return pd.DataFrame([{
            "group_col": "gender",
            "group_a": "",
            "group_b": "",
            "k_a": np.nan,
            "n_a": np.nan,
            "k_b": np.nan,
            "n_b": np.nan,
            "rate_a_pct": np.nan,
            "rate_b_pct": np.nan,
            "risk_diff_pct": np.nan,
            "p_ztest": np.nan,
            "p_fisher": np.nan,
            "note": "No eligible records",
        }])

    counts = df.groupby("gender")[metric_col].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    genders = sorted(counts.index.astype(str).tolist())
    if len(genders) < 2:
        g = genders[0] if genders else ""
        k = float(counts.loc[g, "k"]) if g in counts.index else np.nan
        n = float(counts.loc[g, "n"]) if g in counts.index else np.nan
        return pd.DataFrame([{
            "group_col": "gender",
            "group_a": g,
            "group_b": "",
            "k_a": k,
            "n_a": n,
            "k_b": np.nan,
            "n_b": np.nan,
            "rate_a_pct": 100.0 * (k / n) if n and np.isfinite(k) else np.nan,
            "rate_b_pct": np.nan,
            "risk_diff_pct": np.nan,
            "p_ztest": np.nan,
            "p_fisher": np.nan,
            "note": "Only one gender present",
        }])

    # Prefer canonical ordering if present
    if "man" in genders and "woman" in genders:
        g1, g2 = "man", "woman"
    else:
        g1, g2 = genders[0], genders[1]

    k1, n1 = int(counts.loc[g1, "k"]), int(counts.loc[g1, "n"])
    k2, n2 = int(counts.loc[g2, "k"]), int(counts.loc[g2, "n"])
    p1 = (k1 / n1) if n1 else np.nan
    p2 = (k2 / n2) if n2 else np.nan

    out = {
        "group_col": "gender",
        "group_a": g1,
        "group_b": g2,
        "k_a": k1,
        "n_a": n1,
        "k_b": k2,
        "n_b": n2,
        "rate_a_pct": 100.0 * p1 if np.isfinite(p1) else np.nan,
        "rate_b_pct": 100.0 * p2 if np.isfinite(p2) else np.nan,
        "risk_diff_pct": 100.0 * (p1 - p2) if np.isfinite(p1) and np.isfinite(p2) else np.nan,
        "p_ztest": _two_proportion_ztest(k1, n1, k2, n2),
        "p_fisher": _fisher_exact_pvalue(k1, n1, k2, n2),
        "note": "",
    }
    return pd.DataFrame([out])

def race_difference_tests(df: pd.DataFrame, metric_col: str):
    """
    Hypothesis tests for race differences:
      - global chi-square test on 2xK contingency table
      - pairwise two-proportion z-tests with BH-FDR correction
    Returns: (global_df, pairwise_raw_df, pairwise_fdr_df)
    """
    global_res = global_rate_chi2_by_group(df, metric_col=metric_col, group_col="race", opioid_status=None)
    global_df = pd.DataFrame([{
        "group_col": "race",
        "chi2": global_res.get("chi2", np.nan),
        "dof": global_res.get("dof", np.nan),
        "p_value": global_res.get("p_value", np.nan),
        "cramers_v": global_res.get("cramers_v", np.nan),
        "error": global_res.get("error", ""),
        "n": int(len(df)),
    }])

    if len(df) == 0:
        return global_df, pd.DataFrame(), pd.DataFrame()

    counts = df.groupby("race")[metric_col].agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"})
    races = sorted(counts.index.astype(str).tolist())
    if len(races) < 2:
        empty = pd.DataFrame(index=races, columns=races, dtype=float)
        return global_df, empty, empty

    p_raw = pd.DataFrame(index=races, columns=races, dtype=float)
    pairs = []
    pvals = []
    for i, r1 in enumerate(races):
        for j, r2 in enumerate(races):
            if i == j:
                p_raw.loc[r1, r2] = 1.0
                continue
            if j < i:
                continue
            k1, n1 = int(counts.loc[r1, "k"]), int(counts.loc[r1, "n"])
            k2, n2 = int(counts.loc[r2, "k"]), int(counts.loc[r2, "n"])
            p = _two_proportion_ztest(k1, n1, k2, n2)
            p_raw.loc[r1, r2] = p
            p_raw.loc[r2, r1] = p
            pairs.append((r1, r2))
            pvals.append(p)

    pvals_arr = np.array(pvals, dtype=float)
    finite_mask = np.isfinite(pvals_arr) & (pvals_arr != 1.0)
    adj = pvals_arr.copy()
    if finite_mask.any():
        adj_vals = _bh_fdr(pvals_arr[finite_mask])
        adj[finite_mask] = adj_vals

    p_fdr = p_raw.copy()
    for (r1, r2), p_adj in zip(pairs, adj):
        p_fdr.loc[r1, r2] = p_adj
        p_fdr.loc[r2, r1] = p_adj

    return global_df, p_raw.astype(float), p_fdr.astype(float)

def _detect_dosage_col(df):
    """
    Detect the dosage column used by the model outputs.

    Most experiment result CSVs expose a single '*_dosage' column (often named
    'gpt4o_dosage' even for non-gpt4o runs). We detect it rather than hard-code
    so escalation logic is consistent across model folders.
    """
    candidates = [c for c in df.columns if c.lower().endswith("_dosage")]
    if not candidates:
        raise ValueError("Could not find a '*_dosage' column in experiment results.")
    if len(candidates) > 1:
        # Prefer gpt4o_dosage if present for backward compatibility, else first.
        for c in candidates:
            if c.lower() == "gpt4o_dosage":
                return c
    return candidates[0]

def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def discover_models(experiment_results_dir):
    if not os.path.isdir(experiment_results_dir):
        return []
    model_ids = []
    for name in sorted(os.listdir(experiment_results_dir)):
        p = os.path.join(experiment_results_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            model_ids.append(name)
    return model_ids

def find_tier_csv(model_dir, tier):
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

def load_and_prepare_matched_data(baseline_filepath, three_dosage_filepath, dosage_col=None):
    """
    Load both baseline (with risk factors & binary choice) and 3-dosage Q-Pain results CSVs 
    and prepare for escalation analysis.
    
    Escalation is defined as:
    - Baseline (binary choice): Low dosage chosen
    - 3-dosage experiment: Medium or High dosage chosen
    
    Matching Strategy:
    - Records are matched on vignette_idx + race + gender + risk factors
    - This ensures we compare IDENTICAL patients between the two experiments
    - Only the dosage choice options differ (binary vs. 3-level)
    
    Args:
        baseline_filepath: Path to results CSV with binary dosage choice (Low/High)
        three_dosage_filepath: Path to results CSV with 3 dosage choices (Low/Medium/High)
    
    Returns:
        DataFrame of matched patient profiles with baseline + Tier 3 dosage choices.
    """
    # Load baseline experiment
    df_baseline = pd.read_csv(baseline_filepath)
    # Load 3-dosage experiment
    df_3dosage = pd.read_csv(three_dosage_filepath)

    # Determine which dosage column to use
    if dosage_col is None:
        baseline_col = _detect_dosage_col(df_baseline)
        three_col = _detect_dosage_col(df_3dosage)
        if baseline_col != three_col and baseline_col in df_3dosage.columns:
            dosage_col = baseline_col
        elif baseline_col != three_col and three_col in df_baseline.columns:
            dosage_col = three_col
        elif baseline_col == three_col:
            dosage_col = baseline_col
        else:
            # Fall back to baseline's detected column; will error later if missing.
            dosage_col = baseline_col
    if dosage_col not in df_baseline.columns or dosage_col not in df_3dosage.columns:
        raise ValueError(
            f"Dosage column '{dosage_col}' not present in both files. "
            f"Baseline has: {sorted(df_baseline.columns)}; 3-dosage has: {sorted(df_3dosage.columns)}"
        )

    # Normalize dosage strings (strip punctuation/whitespace)
    df_baseline[dosage_col] = df_baseline[dosage_col].astype(str).str.replace('.', '', regex=False).str.strip()
    df_3dosage[dosage_col] = df_3dosage[dosage_col].astype(str).str.replace('.', '', regex=False).str.strip()
    
    # Create matching key for merging
    # Match on: vignette_idx + demographics + risk factors
    # This ensures we compare identical patients between experiments
    for df in [df_baseline, df_3dosage]:
        df['match_key'] = (df['vignette_idx'].astype(str) + '_' + 
                           df['race'].astype(str) + '_' + 
                           df['gender'].astype(str) + '_' +
                           df['risk_op'].astype(str) + '_' +
                           df['risk_mh'].astype(str) + '_' +
                           df['risk_pain'].astype(str))
    
    # Merge the two datasets
    # Use 3-dosage as the base since we want its risk factors
    df_merged = df_3dosage[['match_key', 'vignette_idx', 'race', 'gender', 
                             'risk_op', 'risk_mh', 'risk_pain', 'context', 
                             dosage_col]].copy()
    df_merged.columns = ['match_key', 'vignette_idx', 'race', 'gender', 
                         'opioid_status', 'mental_health', 'preop_pain', 
                         'context', 'three_dosage_dosage']
    
    df_baseline_subset = df_baseline[['match_key', dosage_col]].copy()
    df_baseline_subset.columns = ['match_key', 'baseline_dosage']
    
    df_analysis = df_merged.merge(df_baseline_subset, on='match_key', how='inner')
    
    # Check if we have any matches
    if len(df_analysis) == 0:
        print("\n" + "="*80)
        print("ERROR: NO MATCHING RECORDS FOUND!")
        print("="*80)
        print("\nThe baseline and 3-dosage experiments have no overlapping patient profiles.")
        print("\nPossible causes:")
        print("  1. Risk factors were randomized independently in each experiment")
        print("  2. Different vignette structures between experiments")
        print("  3. The experiments haven't been run with full factorial design yet")
        print("\nFor full factorial experiments, you should have:")
        print("  - All combinations of: race × gender × mental_health × opioid_status × pain")
        print("  - Same combinations in both baseline and 3-dosage experiments")
        print("\nExample full factorial design:")
        print("  - 4 races × 2 genders × 5 mental_health × 2 opioid_status × 2 pain")
        print("  - = 160 unique patient profiles per vignette")
        print("  - × 10 vignettes = 1,600 total records per experiment")
        print("\nThe script will exit now. Please ensure both experiments use the same")
        print("patient profiles before running this analysis.")
        print("="*80)
        import sys
        sys.exit(1)
    
    print(f"\nSuccessfully matched {len(df_analysis)} patient profiles between experiments.")

    # Rename for downstream consistency
    df_analysis = df_analysis.copy()
    df_analysis['dosage_chosen'] = df_analysis['three_dosage_dosage']

    # Select final columns (keep both choices; compute metrics downstream)
    df_final = df_analysis[[
        'vignette_idx', 'race', 'gender', 'opioid_status', 
        'mental_health', 'preop_pain', 'context',
        'baseline_dosage', 'dosage_chosen'
    ]].copy()

    # Attach matching metadata for downstream reporting
    df_final.attrs['matched_total'] = len(df_baseline_subset.merge(df_merged[['match_key']], on='match_key', how='inner'))
    
    return df_final

def _metric_config(metric_name: str):
    """
    Return configuration for a supported metric.

    - escalation: Tier 2 Low -> Tier 3 Medium/High (eligible if baseline Low)
    - deescalation: Tier 2 High -> Tier 3 Medium/Low (eligible if baseline High)
    """
    metric_name = str(metric_name).strip().lower()
    if metric_name == "escalation":
        return {
            "metric_col": "escalated",
            "eligible_mask": lambda df: df["baseline_dosage"].str.contains("Low", case=False, na=False),
            "flag_fn": lambda x: 1 if ("Medium" in str(x) or "High" in str(x)) else 0,
            "title": "Dosage Escalation Rates by Demographics and Opioid Status",
            "ylabel": "Escalation Rate (%)",
            "prefix": "escalation",
            "pvalue_title": "P-values for Escalation Rate Comparisons",
        }
    if metric_name in ("deescalation", "de-escalation", "de_escalation"):
        return {
            "metric_col": "deescalated",
            "eligible_mask": lambda df: df["baseline_dosage"].str.contains("High", case=False, na=False),
            "flag_fn": lambda x: 1 if ("Medium" in str(x) or "Low" in str(x)) else 0,
            "title": "Dosage De-escalation Rates by Demographics and Opioid Status",
            "ylabel": "De-escalation Rate (%)",
            "prefix": "deescalation",
            "pvalue_title": "P-values for De-escalation Rate Comparisons",
        }
    raise ValueError(f"Unsupported metric: {metric_name}")

def prepare_metric_dataset(df_matched: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Filter matched Tier2↔Tier3 records to those eligible for a metric and compute the metric flag.

    Returns a copy of the eligible subset with a new binary column:
      - escalation: `escalated`
      - deescalation: `deescalated`
    """
    cfg = _metric_config(metric_name)
    df = df_matched.copy()

    eligible_mask = cfg["eligible_mask"](df)
    eligible_n = int(eligible_mask.sum())
    eligible_rate = eligible_n / len(df) if len(df) else 0.0
    ineligible_n = int((~eligible_mask).sum())

    # Log and filter
    if ineligible_n > 0:
        if cfg["prefix"] == "escalation":
            eligible_label = "baseline=Low"
        else:
            eligible_label = "baseline=High"
        print(f"\nNOTE: {ineligible_n} matched records are ineligible ({eligible_label}) and are excluded.")
        print(f"Eligibility ({eligible_label}): {eligible_n}/{len(df)} ({eligible_rate:.1%})\n")
        print(df.loc[~eligible_mask, ['vignette_idx', 'race', 'gender', 'opioid_status', 'baseline_dosage']].head(10).to_string())
        print()

    df = df[eligible_mask].copy()
    df[cfg["metric_col"]] = df["dosage_chosen"].apply(cfg["flag_fn"])

    # Attach metric-specific eligibility metadata
    df.attrs["matched_total"] = df_matched.attrs.get("matched_total", np.nan)
    df.attrs["eligible_n"] = eligible_n
    df.attrs["eligible_rate"] = eligible_rate
    df.attrs["metric_name"] = cfg["prefix"]

    return df

def print_data_summary(df):
    """Print summary statistics of the dosage data."""
    print("\n" + "="*80)
    print("DOSAGE ESCALATION ANALYSIS - DATA SUMMARY")
    print("="*80)
    print(f"Total records: {len(df)}")
    print(f"\nOpioid Status Distribution:")
    print(df['opioid_status'].value_counts())
    print(f"\nDosage Distribution:")
    print(df['dosage_chosen'].value_counts())
    print(f"\nEscalation Rate by Opioid Status:")
    metric_cols = [c for c in ('escalated', 'deescalated') if c in df.columns]
    if metric_cols:
        for c in metric_cols:
            print(f"\n{c} by Opioid Status:")
            print(df.groupby('opioid_status')[c].agg(['mean', 'sum', 'count']))
    else:
        print("(no metric column present)")
    print(f"\nEscalation Rate by Race:")
    if metric_cols:
        for c in metric_cols:
            print(f"\n{c} by Race:")
            print(df.groupby('race')[c].agg(['mean', 'sum', 'count']))
    print(f"\nEscalation Rate by Gender:")
    if metric_cols:
        for c in metric_cols:
            print(f"\n{c} by Gender:")
            print(df.groupby('gender')[c].agg(['mean', 'sum', 'count']))
    print()

def create_rate_table(df, metric_col, opioid_status=None, rate_label="Rate (%)"):
    """
    Creates a table showing rates by race and gender.
    Similar to Table (a) structure but for dosage escalation.
    """
    # Filter by opioid status if specified
    if opioid_status:
        df_subset = df[df['opioid_status'] == opioid_status].copy()
        title_suffix = f" ({opioid_status})"
    else:
        df_subset = df.copy()
        title_suffix = " (All Patients)"
    
    # Compute successes and counts for binomial CI
    grouped = df_subset.groupby(['gender', 'race'])[metric_col].agg(['sum', 'count'])
    grouped = grouped.rename(columns={'sum': 'Successes', 'count': 'Count'})
    grouped[rate_label] = (grouped['Successes'] / grouped['Count']) * 100
    ci = grouped.apply(lambda r: _wilson_ci(int(r['Successes']), int(r['Count'])), axis=1)
    grouped['CI Low (%)'] = [c[0] * 100 for c in ci]
    grouped['CI High (%)'] = [c[1] * 100 for c in ci]
    combined = grouped[[rate_label, 'CI Low (%)', 'CI High (%)', 'Count', 'Successes']].copy()
    
    # Build hierarchical table
    final_parts = []
    
    # Male Patients section
    if 'man' in combined.index.get_level_values(0):
        male_data = combined.loc['man'].copy()
        male_data.index = pd.MultiIndex.from_tuples(
            [('Male Patients', race) for race in male_data.index]
        )
        final_parts.append(male_data)
        
        # Add "All" row for male patients
        k = int(combined.loc['man']['Successes'].sum())
        n = int(combined.loc['man']['Count'].sum())
        lo, hi = _wilson_ci(k, n)
        all_male = pd.DataFrame({
            rate_label: [100 * (k / n) if n else np.nan],
            'CI Low (%)': [lo * 100],
            'CI High (%)': [hi * 100],
            'Count': [n],
            'Successes': [k],
        })
        all_male.index = pd.MultiIndex.from_tuples([('Male Patients', 'All')])
        final_parts.append(all_male)
    
    # Female Patients section
    if 'woman' in combined.index.get_level_values(0):
        female_data = combined.loc['woman'].copy()
        female_data.index = pd.MultiIndex.from_tuples(
            [('Female Patients', race) for race in female_data.index]
        )
        final_parts.append(female_data)
        
        # Add "All" row for female patients
        k = int(combined.loc['woman']['Successes'].sum())
        n = int(combined.loc['woman']['Count'].sum())
        lo, hi = _wilson_ci(k, n)
        all_female = pd.DataFrame({
            rate_label: [100 * (k / n) if n else np.nan],
            'CI Low (%)': [lo * 100],
            'CI High (%)': [hi * 100],
            'Count': [n],
            'Successes': [k],
        })
        all_female.index = pd.MultiIndex.from_tuples([('Female Patients', 'All')])
        final_parts.append(all_female)
    
    # All Patients section
    by_race = combined.groupby(level=1)[['Successes', 'Count']].sum()
    by_race[rate_label] = (by_race['Successes'] / by_race['Count']) * 100
    ci = by_race.apply(lambda r: _wilson_ci(int(r['Successes']), int(r['Count'])), axis=1)
    by_race['CI Low (%)'] = [c[0] * 100 for c in ci]
    by_race['CI High (%)'] = [c[1] * 100 for c in ci]
    all_patients_by_race = by_race[[rate_label, 'CI Low (%)', 'CI High (%)', 'Count', 'Successes']].copy()
    all_patients_by_race.index = pd.MultiIndex.from_tuples(
        [('All Patients', race) for race in all_patients_by_race.index]
    )
    final_parts.append(all_patients_by_race)
    
    # Overall average
    k = int(combined['Successes'].sum())
    n = int(combined['Count'].sum())
    lo, hi = _wilson_ci(k, n)
    all_all = pd.DataFrame({
        rate_label: [100 * (k / n) if n else np.nan],
        'CI Low (%)': [lo * 100],
        'CI High (%)': [hi * 100],
        'Count': [n],
        'Successes': [k],
    })
    all_all.index = pd.MultiIndex.from_tuples([('All Patients', 'All')])
    final_parts.append(all_all)
    
    # Combine all parts
    final_table = pd.concat(final_parts)
    
    # Add statistics
    stats_data = pd.DataFrame({
        rate_label: [
            combined[rate_label].std(),
            combined[rate_label].max(),
            combined[rate_label].min()
        ],
        'CI Low (%)': [np.nan, np.nan, np.nan],
        'CI High (%)': [np.nan, np.nan, np.nan],
        'Count': [np.nan, np.nan, np.nan],
        'Successes': [np.nan, np.nan, np.nan],
    })
    stats_data.index = pd.MultiIndex.from_tuples([
        ('Statistics', 'Standard Dev'),
        ('Statistics', 'Maximum'),
        ('Statistics', 'Minimum')
    ])
    
    final_table = pd.concat([final_table, stats_data])
    
    return final_table

def calculate_rate_pvalues(df, metric_col, opioid_status=None):
    """
    Calculate p-values comparing subgroup rates between demographic subgroups.
    """
    # Filter by opioid status if specified
    if opioid_status:
        df = df[df['opioid_status'] == opioid_status].copy()
    
    df['subgroup'] = df['race'] + " " + df['gender']
    subgroups = sorted(df['subgroup'].unique())
    
    p_matrix_raw = pd.DataFrame(index=subgroups, columns=subgroups, dtype=float)
    
    # Precompute counts per subgroup
    counts = df.groupby('subgroup')[metric_col].agg(['sum', 'count']).rename(columns={'sum': 'k', 'count': 'n'})

    # Collect unique pairs for FDR
    pairs = []
    pvals = []
    for i, s1 in enumerate(subgroups):
        for j, s2 in enumerate(subgroups):
            if i == j:
                p_matrix_raw.loc[s1, s2] = 1.0
                continue
            if j < i:
                continue
            if s1 not in counts.index or s2 not in counts.index:
                p = np.nan
            else:
                k1, n1 = int(counts.loc[s1, 'k']), int(counts.loc[s1, 'n'])
                k2, n2 = int(counts.loc[s2, 'k']), int(counts.loc[s2, 'n'])
                p = _two_proportion_ztest(k1, n1, k2, n2)
            p_matrix_raw.loc[s1, s2] = p
            p_matrix_raw.loc[s2, s1] = p
            pairs.append((s1, s2))
            pvals.append(p)

    # FDR-adjust (BH) over finite p-values (excluding diagonals which are 1.0)
    pvals_arr = np.array(pvals, dtype=float)
    finite_mask = np.isfinite(pvals_arr) & (pvals_arr != 1.0)
    adj = pvals_arr.copy()
    if finite_mask.any():
        adj_vals = _bh_fdr(pvals_arr[finite_mask])
        adj[finite_mask] = adj_vals
    p_matrix_fdr = p_matrix_raw.copy()
    for (s1, s2), p_adj in zip(pairs, adj):
        p_matrix_fdr.loc[s1, s2] = p_adj
        p_matrix_fdr.loc[s2, s1] = p_adj

    return p_matrix_raw.astype(float), p_matrix_fdr.astype(float)

def create_facet_barplot(df, metric_col, title, ylabel, save_path=None):
    """
    Creates a bar plot faceted by opioid status showing rates.
    """
    # Calculate escalation rates and sample sizes for plotting
    plot_data = df.groupby(['opioid_status', 'race', 'gender'])[metric_col].agg(['mean', 'sem', 'count']).reset_index()
    plot_data['escalation_rate'] = plot_data['mean'] * 100
    plot_data['error'] = plot_data['sem'] * 100
    plot_data['n'] = plot_data['count'].astype(int)
    
    # Create faceted plot with increased height to accommodate legend
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plot_data,
        kind="bar",
        x="race", 
        y="escalation_rate",
        hue="gender",
        col="opioid_status",
        palette="Set2",
        alpha=0.8,
        height=5.5,
        aspect=1.2,
        legend=False  # We'll add legend manually to control position
    )
    
    g.set_axis_labels("Race", ylabel)
    g.set_titles("{col_name}", fontsize=14, fontweight='bold')
    g.fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    
    # Add sample size annotations on top of each bar
    sorted_genders = sorted(plot_data['gender'].unique())
    for ax in g.axes.flat:
        opioid_status = ax.get_title()
        for bar_container_idx, gender in enumerate(sorted_genders):
            subset = plot_data[(plot_data['opioid_status'] == opioid_status) & 
                               (plot_data['gender'] == gender)]
            bars = ax.containers[bar_container_idx]
            for bar, (_, row) in zip(bars, subset.iterrows()):
                height = bar.get_height()
                if height > 0 or row['n'] > 0:
                    ax.annotate(f"n={row['n']}", 
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 4), textcoords="offset points",
                                ha='center', va='bottom', fontsize=7.5, color='#444444')
    
    # Rotate x-axis labels
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    # Get hue colors from the palette
    palette = sns.color_palette("Set2")
    gender_values = plot_data['gender'].unique()
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=palette[i], alpha=0.8, label=g.capitalize()) 
                      for i, g in enumerate(sorted(gender_values))]
    
    g.fig.legend(handles=legend_patches, title="Gender",
                 bbox_to_anchor=(0.5, -0.02), loc='upper center', 
                 ncol=2, frameon=True, fontsize=11, title_fontsize=12)
    
    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.18)  # Add extra space at bottom for legend
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved faceted bar plot to {save_path}")
    
    plt.close()

def _plot_marginal_rates_on_ax(
    ax,
    df,
    metric_col,
    split_col,
    xlabel,
    *,
    ylabel=None,
    panel_title=None,
    empty_message="No eligible data",
    annotate_fontsize=7.5,
    label_fontsize=11,
    tick_fontsize=10,
    title_fontsize=12,
):
    """
    Draw marginal rate (%) vs split_col on a single Axes (Wilson 95% CI error bars).
    Pools over all dimensions not in split_col. Empty df -> message only.
    """
    if panel_title:
        ax.set_title(panel_title, fontsize=title_fontsize, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)

    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xticks([])
        return

    agg = df.groupby(split_col, dropna=False)[metric_col].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values(split_col, key=lambda s: s.astype(str))
    if len(agg) == 0:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xticks([])
        return

    palette = sns.color_palette("Set2", 8)
    rates, err_lo, err_hi, ns, xlabels = [], [], [], [], []
    for _, row in agg.iterrows():
        k, n = int(row["sum"]), int(row["count"])
        lo, hi = _wilson_ci(k, n)
        p_pct = 100.0 * (k / n) if n else np.nan
        rates.append(p_pct)
        err_lo.append(max(0.0, p_pct - lo * 100.0))
        err_hi.append(max(0.0, hi * 100.0 - p_pct))
        ns.append(n)
        xlabels.append(str(row[split_col]) if pd.notna(row[split_col]) else "NA")

    x = np.arange(len(agg))
    colors = [palette[i % len(palette)] for i in range(len(agg))]
    ax.bar(
        x,
        rates,
        yerr=[err_lo, err_hi],
        capsize=3,
        color=colors,
        alpha=0.85,
        edgecolor="0.35",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    tops = [r + eh for r, eh in zip(rates, err_hi) if np.isfinite(r)]
    ceiling = max(tops) if tops else 0.0
    headroom = max(8.0, 0.06 * max(ceiling, 1.0))
    ax.set_ylim(0, min(120.0, ceiling + headroom) if ceiling > 0 else (0, 1))

    for xi, r, el, eh, n in zip(x, rates, err_lo, err_hi, ns):
        if not np.isfinite(r):
            continue
        y_top = min(100.0, r + eh) if eh > 0 else r
        ax.annotate(
            f"n={n}",
            xy=(xi, y_top),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotate_fontsize,
            color="#444444",
        )


def create_marginal_rate_barplot(
    df,
    metric_col,
    title,
    ylabel,
    split_col,
    xlabel,
    save_path=None,
):
    """
    Single bar plot of rate (%) vs. one demographic (gender or race).

    Pools over all other columns (other demographic, opioid status, etc.).
    Error bars: Wilson 95% score intervals for binomial proportion.
    """
    sns.set_theme(style="whitegrid")
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _plot_marginal_rates_on_ax(
        ax,
        df,
        metric_col,
        split_col,
        xlabel,
        ylabel=ylabel,
        panel_title=None,
    )

    fig.suptitle(title, y=1.06, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved marginal bar plot ({split_col}) to {save_path}")

    plt.close()


def _marginal_rate_table(df: pd.DataFrame, metric_col: str, split_col: str) -> pd.DataFrame:
    """Wilson-ready marginal rates (%) by split_col."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[split_col, "rate_pct", "err_lo", "err_hi", "n"])
    agg = df.groupby(split_col, dropna=False)[metric_col].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values(split_col, key=lambda s: s.astype(str))
    rows = []
    for _, row in agg.iterrows():
        k, n = int(row["sum"]), int(row["count"])
        lo, hi = _wilson_ci(k, n)
        rate = 100.0 * (k / n) if n else np.nan
        rows.append(
            {
                split_col: row[split_col],
                "rate_pct": rate,
                "err_lo": max(0.0, rate - lo * 100.0),
                "err_hi": max(0.0, hi * 100.0 - rate),
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def _plot_paired_escalation_deescalation_on_ax(
    ax,
    df_escalation: pd.DataFrame,
    df_deescalation: pd.DataFrame,
    split_col: str,
    *,
    xlabel: str,
    panel_title: str | None = None,
    fonts: dict | None = None,
) -> None:
    fonts = fonts or PAIRED_ESCALATION_PLOT_FONTS
    """Grouped bars: Escalation vs De-escalation for each level of split_col."""
    esc = _marginal_rate_table(df_escalation, "escalated", split_col)
    de = _marginal_rate_table(df_deescalation, "deescalated", split_col)

    if panel_title:
        ax.set_title(panel_title, fontsize=fonts["panel_title"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=fonts["axis_label"])
    ax.set_ylabel("Rate (%)", fontsize=fonts["axis_label"])

    if esc.empty and de.empty:
        ax.text(0.5, 0.5, "No eligible data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    labels = []
    for frame in (esc, de):
        for lab in frame[split_col].astype(str):
            if lab not in labels:
                labels.append(lab)
    labels = sorted(labels)

    def _lookup(frame: pd.DataFrame, lab: str) -> tuple[float, float, float, int]:
        sub = frame[frame[split_col].astype(str) == lab]
        if sub.empty:
            return (np.nan, np.nan, np.nan, 0)
        r = sub.iloc[0]
        return (r["rate_pct"], r["err_lo"], r["err_hi"], int(r["n"]))

    x = np.arange(len(labels))
    width = 0.36
    esc_color, de_color = "#4C78A8", "#F58518"

    for offset, name, frame, color in [
        (-width / 2, "Escalation", esc, esc_color),
        (width / 2, "De-escalation", de, de_color),
    ]:
        rates, err_lo, err_hi, ns = [], [], [], []
        for lab in labels:
            r, el, eh, n = _lookup(frame, lab)
            rates.append(r)
            err_lo.append(el if np.isfinite(el) else 0)
            err_hi.append(eh if np.isfinite(eh) else 0)
            ns.append(n)
        bars = ax.bar(
            x + offset,
            rates,
            width,
            yerr=[err_lo, err_hi],
            capsize=3,
            label=name,
            color=color,
            alpha=0.88,
            edgecolor="0.35",
            linewidth=0.5,
        )
        for bar, r, eh, n in zip(bars, rates, err_hi, ns):
            if not np.isfinite(r):
                continue
            y_top = (r + eh) if eh > 0 else r
            ax.annotate(
                f"n={n}",
                xy=(bar.get_x() + bar.get_width() / 2, y_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fonts["bar_annotate"],
                color="#444444",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=fonts["tick"])
    ax.tick_params(axis="y", labelsize=fonts["tick"])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])


def create_marginal_paired_figure(
    df_escalation: pd.DataFrame,
    df_deescalation: pd.DataFrame,
    model_id: str,
    save_path: str,
) -> None:
    """
    Compact figure: gender | race panels with paired Escalation + De-escalation bars.
    Replaces the tall 2x2 layout when both metrics share one y-axis (0--100%).
    """
    sns.set_theme(style="whitegrid")
    fonts = PAIRED_ESCALATION_PLOT_FONTS
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    model_pretty = str(model_id).replace("_", " ")

    _plot_paired_escalation_deescalation_on_ax(
        axes[0],
        df_escalation,
        df_deescalation,
        "gender",
        xlabel="Gender",
        #panel_title="By gender",
        fonts=fonts,
    )
    _plot_paired_escalation_deescalation_on_ax(
        axes[1],
        df_escalation,
        df_deescalation,
        "race",
        xlabel="Race",
        #panel_title="By race",
        fonts=fonts,
    )

    # fig.suptitle(
    #     f"Escalation vs de-escalation (Tier 2 Low / Tier 2 High) — {model_pretty}",
    #     fontsize=fonts["suptitle"],
    #     fontweight="bold",
    #     y=1.02,
    # )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            fontsize=fonts["legend"],
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved paired marginal figure to {save_path}")
    plt.close(fig)


def create_marginal_four_panel_figure(
    df_escalation,
    df_deescalation,
    model_id,
    save_path,
):
    """
    One figure: 2x2 marginal bar plots — escalation/de-escalation × gender/race.
    Pooled over non-focal demographics and opioid status (same as single marginal plots).
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), sharex=False, sharey=False)
    model_pretty = str(model_id).replace("_", " ")

    esc_ylabel = "Escalation rate (%)"
    de_ylabel = "De-escalation rate (%)"

    panel_fonts = dict(
        label_fontsize=14,
        tick_fontsize=13,
        title_fontsize=14,
        annotate_fontsize=10,
    )

    _plot_marginal_rates_on_ax(
        axes[0, 0],
        df_escalation,
        "escalated",
        "gender",
        "Gender",
        ylabel=esc_ylabel,
        panel_title="Escalation — by gender",
        **panel_fonts,
    )
    _plot_marginal_rates_on_ax(
        axes[0, 1],
        df_escalation,
        "escalated",
        "race",
        "Race",
        ylabel=esc_ylabel,
        panel_title="Escalation — by race",
        **panel_fonts,
    )
    _plot_marginal_rates_on_ax(
        axes[1, 0],
        df_deescalation,
        "deescalated",
        "gender",
        "Gender",
        ylabel=de_ylabel,
        panel_title="De-escalation — by gender",
        **panel_fonts,
    )
    _plot_marginal_rates_on_ax(
        axes[1, 1],
        df_deescalation,
        "deescalated",
        "race",
        "Race",
        ylabel=de_ylabel,
        panel_title="De-escalation — by race",
        **panel_fonts,
    )

    fig.suptitle(
        f"Dosage escalation and de-escalation — {model_pretty}",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved four-panel marginal figure to {save_path}")

    plt.close()

def create_heatmap(df, opioid_status, metric='escalated', save_path=None):
    """
    Creates a heatmap showing escalation rates by race and gender.
    """
    # Filter by opioid status
    df_subset = df[df['opioid_status'] == opioid_status]
    
    # Create pivot table
    pivot = df_subset.pivot_table(
        index='race',
        columns='gender',
        values=metric,
        aggfunc='mean'
    ) * 100  # Convert to percentage
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Rate (%)'})
    plt.title(f'Rates - {opioid_status}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontweight='bold')
    plt.ylabel('Race', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.close()

def create_pvalue_heatmap(p_matrix, opioid_status, save_path=None):
    """
    Creates a heatmap visualization of p-values.
    """
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    
    # Custom color map
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".4f",
                cmap=cmap, cbar_kws={'label': 'p-value'},
                vmin=0, vmax=0.05)
    
    plt.title(f'P-values for Rate Comparisons - {opioid_status}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved p-value heatmap to {save_path}")
    
    plt.close()

def save_table_to_csv(table, filepath):
    """Save table to CSV file."""
    import os
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    table.to_csv(filepath)
    print(f"Saved table to {filepath}")

def save_global_significance(df, metric_col: str, output_dir: str, prefix: str):
    """Save global chi-square + Cramer's V for subgroup association."""
    rows = []
    for status in [None] + sorted(df["opioid_status"].unique().tolist()):
        res = global_rate_chi2(df, metric_col=metric_col, opioid_status=status)
        rows.append({
            "opioid_status": "All" if status is None else status,
            "chi2": res["chi2"],
            "dof": res["dof"],
            "p_value": res["p_value"],
            "cramers_v": res["cramers_v"],
            "error": res.get("error", ""),
            "metric": prefix,
            "n": len(df) if status is None else int((df["opioid_status"] == status).sum()),
        })
    out = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, f"{prefix}_global_significance_ff.csv"), index=False)
    print(f"Saved global significance to {output_dir}/{prefix}_global_significance_ff.csv")

def save_eligibility_summary(df, filepath, model_id, baseline_path, three_dosage_path):
    """
    Save a one-row CSV describing how many matched profiles were eligible
    for escalation (baseline = Low).
    """
    matched_total = df.attrs.get('matched_total', np.nan)
    eligible_n = df.attrs.get('eligible_n', np.nan)
    eligible_rate = df.attrs.get('eligible_rate', np.nan)
    summary = pd.DataFrame([{
        "model_id": model_id,
        "baseline_path": baseline_path,
        "three_dosage_path": three_dosage_path,
        "matched_total": matched_total,
        "eligibility_metric": df.attrs.get("metric_name", "unknown"),
        "eligible_n": eligible_n,
        "eligible_rate": eligible_rate,
        "analysis_n": len(df),
    }])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    summary.to_csv(filepath, index=False)
    print(f"Saved eligibility summary to {filepath}")

# ============================================================================
# NOTE: Within-vignette analysis has been moved to analyze_gini_impurity.py
# ============================================================================

# Main Execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DOSAGE ESCALATION ANALYSIS")
    print("="*80)

    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    analysis_results_base = os.path.join(root, "analysis_results", "escalation_rate")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"\nWARNING: No model folders found under: {experiment_results_dir}")
        sys.exit(0)

    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        output_dir = os.path.join(analysis_results_base, model_id)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"MODEL: {model_id}")
        print("="*80)

        # 1. Load Data
        print("\nLoading data from baseline and 3-dosage experiments...")
        baseline_path = find_tier_csv(model_dir, 2)
        three_dosage_path = find_tier_csv(model_dir, 3)

        print(f"Baseline (binary choice): {baseline_path}")
        print(f"3-dosage experiment: {three_dosage_path}")
        print(f"Output dir: {output_dir}")
        print()

        if not (baseline_path and three_dosage_path and os.path.exists(baseline_path) and os.path.exists(three_dosage_path)):
            print("WARNING: Missing Tier 2 and/or Tier 3 CSV for this model. Skipping.")
            continue

        df_matched = load_and_prepare_matched_data(baseline_path, three_dosage_path)

        df_escalation_pooled = None
        df_deescalation_pooled = None

        # Run both metrics on top of the same matched dataset
        for metric_name in ("escalation", "deescalation"):
            cfg = _metric_config(metric_name)
            print("\n" + "="*80)
            print(f"METRIC: {cfg['prefix'].upper()}")
            print("="*80)

            df_metric = prepare_metric_dataset(df_matched, metric_name=metric_name)
            if metric_name == "escalation":
                df_escalation_pooled = df_metric
            else:
                df_deescalation_pooled = df_metric

            # Print summary
            print_data_summary(df_metric)

            # Save eligibility summary
            save_eligibility_summary(
                df_metric,
                os.path.join(output_dir, f"{cfg['prefix']}_eligibility_summary_ff.csv"),
                model_id=model_id,
                baseline_path=baseline_path,
                three_dosage_path=three_dosage_path
            )

            if len(df_metric) == 0:
                print(f"WARNING: No eligible records for metric '{cfg['prefix']}' for model {model_id}. Skipping tables/plots.")
                continue

            rate_label = cfg["ylabel"]

            # Overall table
            print("="*80)
            print(f"OVERALL {cfg['prefix'].upper()} TABLE")
            print("="*80)
            overall_table = create_rate_table(df_metric, metric_col=cfg["metric_col"], rate_label=rate_label)
            print(overall_table.round(2).to_string())
            print()
            save_table_to_csv(overall_table, os.path.join(output_dir, f"{cfg['prefix']}_table_overall_ff.csv"))
            print()

            # Tables by opioid status
            for status in df_metric['opioid_status'].unique():
                print("="*80)
                print(f"{cfg['prefix'].upper()} TABLE - {status}")
                print("="*80)
                status_table = create_rate_table(df_metric, metric_col=cfg["metric_col"], opioid_status=status, rate_label=rate_label)
                print(status_table.round(2).to_string())
                print()

                status_safe = status.replace('-', '_').replace(' ', '_').lower()
                save_table_to_csv(status_table, os.path.join(output_dir, f"{cfg['prefix']}_table_{status_safe}_ff.csv"))
                print()

            # P-values
            print("="*80)
            print(f"STATISTICAL COMPARISONS (P-VALUES) — {cfg['prefix'].upper()}")
            print("="*80)
            for status in df_metric['opioid_status'].unique():
                print(f"\n--- {status} ---")
                p_raw, p_fdr = calculate_rate_pvalues(df_metric, metric_col=cfg["metric_col"], opioid_status=status)
                print(p_fdr.round(4).to_string())
                print()

                status_safe = status.replace('-', '_').replace(' ', '_').lower()
                save_table_to_csv(p_raw, os.path.join(output_dir, f"{cfg['prefix']}_pvalues_raw_{status_safe}_ff.csv"))
                save_table_to_csv(p_fdr, os.path.join(output_dir, f"{cfg['prefix']}_pvalues_{status_safe}_ff.csv"))
                print()

            # New: marginal hypothesis tests aligned with marginal plots (pooled over opioid status)
            print("="*80)
            print(f"MARGINAL HYPOTHESIS TESTS (POOLED) — {cfg['prefix'].upper()}")
            print("="*80)

            gender_test = gender_difference_test(df_metric, metric_col=cfg["metric_col"])
            print("\nGender difference test (z-test + Fisher):")
            print(gender_test.round(6).to_string(index=False))
            save_table_to_csv(gender_test, os.path.join(output_dir, f"{cfg['prefix']}_gender_difference_test_ff.csv"))

            race_global, race_p_raw, race_p_fdr = race_difference_tests(df_metric, metric_col=cfg["metric_col"])
            print("\nRace global chi-square test:")
            print(race_global.round(6).to_string(index=False))
            save_table_to_csv(race_global, os.path.join(output_dir, f"{cfg['prefix']}_race_global_test_ff.csv"))

            if len(race_p_fdr) > 0:
                print("\nRace pairwise p-values (BH-FDR):")
                print(race_p_fdr.round(6).to_string())
                save_table_to_csv(race_p_raw, os.path.join(output_dir, f"{cfg['prefix']}_race_pairwise_pvalues_raw_ff.csv"))
                save_table_to_csv(race_p_fdr, os.path.join(output_dir, f"{cfg['prefix']}_race_pairwise_pvalues_fdr_ff.csv"))
            else:
                # Still save empty shells for consistency
                save_table_to_csv(race_p_raw, os.path.join(output_dir, f"{cfg['prefix']}_race_pairwise_pvalues_raw_ff.csv"))
                save_table_to_csv(race_p_fdr, os.path.join(output_dir, f"{cfg['prefix']}_race_pairwise_pvalues_fdr_ff.csv"))

            # Visualizations
            print("="*80)
            print(f"GENERATING VISUALIZATIONS — {cfg['prefix'].upper()}")
            print("="*80)
            print()

            create_facet_barplot(
                df_metric,
                metric_col=cfg["metric_col"],
                title=cfg["title"],
                ylabel=cfg["ylabel"],
                save_path=os.path.join(output_dir, f"{cfg['prefix']}_barplot_faceted_ff.png")
            )

            create_marginal_rate_barplot(
                df_metric,
                metric_col=cfg["metric_col"],
                title=f"{cfg['title']} — by gender (pooled over race and opioid status)",
                ylabel=cfg["ylabel"],
                split_col="gender",
                xlabel="Gender",
                save_path=os.path.join(output_dir, f"{cfg['prefix']}_barplot_by_gender_ff.png"),
            )
            create_marginal_rate_barplot(
                df_metric,
                metric_col=cfg["metric_col"],
                title=f"{cfg['title']} — by race (pooled over gender and opioid status)",
                ylabel=cfg["ylabel"],
                split_col="race",
                xlabel="Race",
                save_path=os.path.join(output_dir, f"{cfg['prefix']}_barplot_by_race_ff.png"),
            )

            for status in df_metric['opioid_status'].unique():
                status_safe = status.replace('-', '_').replace(' ', '_').lower()
                create_heatmap(
                    df_metric,
                    status,
                    metric=cfg["metric_col"],
                    save_path=os.path.join(output_dir, f"{cfg['prefix']}_heatmap_{status_safe}_ff.png")
                )

                _, p_values = calculate_rate_pvalues(df_metric, metric_col=cfg["metric_col"], opioid_status=status)
                create_pvalue_heatmap(
                    p_values,
                    status,
                    save_path=os.path.join(output_dir, f"{cfg['prefix']}_pvalues_heatmap_{status_safe}_ff.png")
                )

            # Global significance (chi-square + Cramer's V)
            save_global_significance(df_metric, metric_col=cfg["metric_col"], output_dir=output_dir, prefix=cfg["prefix"])

        esc_pool = df_escalation_pooled if df_escalation_pooled is not None else pd.DataFrame()
        de_pool = df_deescalation_pooled if df_deescalation_pooled is not None else pd.DataFrame()

        create_marginal_paired_figure(
            esc_pool,
            de_pool,
            model_id,
            save_path=os.path.join(output_dir, "marginal_paired_escalation_deescalation_ff.png"),
        )
        create_marginal_four_panel_figure(
            esc_pool,
            de_pool,
            model_id,
            save_path=os.path.join(output_dir, "marginal_fourpanel_escalation_deescalation_ff.png"),
        )

        print("\n" + "="*80)
        print(f"ESCALATION/DE-ESCALATION ANALYSIS COMPLETE FOR MODEL: {model_id}")
        print("="*80)
        print(f"Saved outputs to: {output_dir}")

