import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.special import rel_entr, kl_div
from scipy import stats
import sys
import os
import glob

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


def _detect_dosage_column(df: pd.DataFrame) -> str:
    """
    Tier result CSVs use model-specific dosage columns (gpt4o_dosage, gpt54_dosage, etc.).
    Prefer gpt4o_dosage for backward compatibility, else the sole *_dosage column.
    """
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
    raise ValueError(
        "Could not find a dosage column (expected gpt4o_dosage or *_dosage). "
        f"Columns: {cols}"
    )


def load_three_dosage_data(filepath):
    """
    Load the 3-dosage experiment results.
    
    Args:
        filepath: Path to the 3-dosage CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    df = pd.read_csv(filepath)

    dosage_col = _detect_dosage_column(df)
    raw_dosage = df[dosage_col].astype(str).str.replace(".", "", regex=False).str.strip()

    # Standardize dosage labels to categories
    def standardize_dosage(dosage_str):
        dosage_lower = str(dosage_str).lower()
        if 'none' in dosage_lower:
            return 'None'
        elif 'low' in dosage_lower:
            return 'Low'
        elif 'medium' in dosage_lower:
            return 'Medium'
        elif 'high' in dosage_lower:
            return 'High'
        else:
            return None

    df['dosage_category'] = raw_dosage.apply(standardize_dosage)
    
    # Create demographic subgroup identifier
    df['demo_subgroup'] = df['race'] + '_' + df['gender']
    
    # Prepare analysis dataframe
    df_analysis = pd.DataFrame({
        'vignette_idx': df['vignette_idx'],
        'race': df['race'],
        'gender': df['gender'],
        'demo_subgroup': df['demo_subgroup'],
        'opioid_status': df['risk_op'],
        'mental_health': df['risk_mh'],
        'preop_pain': df['risk_pain'],
        'context': df['context'],
        'dosage_category': df['dosage_category']
    })
    
    # Remove any rows with invalid dosages
    df_analysis = df_analysis.dropna(subset=['dosage_category'])
    
    return df_analysis

def calculate_dosage_distribution(df, dosage_categories=['None', 'Low', 'Medium', 'High']):
    """
    Calculate the probability distribution over dosage categories.
    
    Args:
        df: DataFrame with dosage_category column
        dosage_categories: List of dosage category names in order
        
    Returns:
        numpy array: Probability distribution [P(None), P(Low), P(Medium), P(High)]
    """
    # Count occurrences of each dosage
    dosage_counts = df['dosage_category'].value_counts()
    
    # Create probability distribution
    distribution = np.zeros(len(dosage_categories))
    for i, dosage in enumerate(dosage_categories):
        if dosage in dosage_counts.index:
            distribution[i] = dosage_counts[dosage]
    
    # Normalize to probabilities
    total = distribution.sum()
    if total > 0:
        distribution = distribution / total
    
    # Add small epsilon to avoid log(0) in KL divergence
    epsilon = 1e-10
    distribution = distribution + epsilon
    distribution = distribution / distribution.sum()
    
    return distribution

def calculate_kl_divergence_cross_vignette(df):
    """
    Calculate KL divergence for each demographic subgroup across ALL vignettes.
    
    This measures: "Does this demographic group have a systematically different
    dosage distribution pattern compared to the overall population?"
    
    KL divergence D_KL(P || Q) measures how much P (subgroup distribution) 
    differs from Q (reference distribution).
    
    Args:
        df: DataFrame with dosage data
        
    Returns:
        DataFrame with KL divergence scores and distributions per subgroup
    """
    dosage_categories = ['None', 'Low', 'Medium', 'High']
    
    # Step 1: Calculate reference distribution Q (all patients, all vignettes)
    print("Calculating reference distribution Q (overall population)...")
    Q = calculate_dosage_distribution(df, dosage_categories)
    
    print(f"Reference distribution Q:")
    for i, dosage in enumerate(dosage_categories):
        print(f"  {dosage}: {Q[i]:.4f}")
    print()
    
    # Step 2: Calculate P and KL divergence for each demographic subgroup
    results = []
    
    for demo_subgroup in sorted(df['demo_subgroup'].unique()):
        subgroup_data = df[df['demo_subgroup'] == demo_subgroup]
        
        # Calculate P (this subgroup's distribution across all vignettes)
        P = calculate_dosage_distribution(subgroup_data, dosage_categories)
        
        # Calculate KL divergence D_KL(P || Q)
        # This measures how much information is lost when Q is used to 
        # approximate P, or how "surprised" we'd be if we expected Q but got P
        kl_divergence = np.sum(rel_entr(P, Q))
        
        # Also calculate symmetric Jensen-Shannon divergence for comparison
        M = 0.5 * (P + Q)
        js_divergence = 0.5 * np.sum(rel_entr(P, M)) + 0.5 * np.sum(rel_entr(Q, M))
        
        # Extract race and gender
        race = subgroup_data['race'].iloc[0]
        gender = subgroup_data['gender'].iloc[0]
        
        # Count samples
        n_samples = len(subgroup_data)
        
        results.append({
            'race': race,
            'gender': gender,
            'demo_subgroup': demo_subgroup,
            'kl_divergence': kl_divergence,
            'js_divergence': js_divergence,
            'n_samples': n_samples,
            'P_None': P[0],
            'P_Low': P[1],
            'P_Medium': P[2],
            'P_High': P[3],
            'Q_None': Q[0],
            'Q_Low': Q[1],
            'Q_Medium': Q[2],
            'Q_High': Q[3]
        })
    
    return pd.DataFrame(results)

def calculate_kl_by_opioid_status(df):
    """
    Calculate KL divergence separately for Opioid-Naive and Opioid-Tolerant patients.
    
    This controls for the major clinical factor (opioid status) when comparing demographics.
    
    Args:
        df: DataFrame with dosage data
        
    Returns:
        DataFrame with KL divergence by opioid status
    """
    results = []
    
    for opioid_status in df['opioid_status'].unique():
        print(f"\nAnalyzing {opioid_status} patients...")
        df_subset = df[df['opioid_status'] == opioid_status]
        
        # Calculate KL divergence within this opioid status
        kl_results = calculate_kl_divergence_cross_vignette(df_subset)
        kl_results['opioid_status'] = opioid_status
        
        results.append(kl_results)
    
    return pd.concat(results, ignore_index=True)

def create_kl_summary_table(kl_df, chi2_result=None):
    """
    Create summary statistics for KL divergence analysis.
    
    Args:
        kl_df: DataFrame with KL divergence results
        chi2_result: Optional dict with chi2, p_value, cramers_v from statistical test
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = {
        'Metric': [
            'Mean KL Divergence',
            'Median KL Divergence',
            'Std Dev KL Divergence',
            'Min KL Divergence',
            'Max KL Divergence',
            '',
            'Max KL Divergence 95% CI (bootstrap)',
            'Mean JS Divergence',
            'Median JS Divergence',
            '',
            'Total Subgroups Analyzed',
            'Subgroups with KL > 0.01 (notable divergence)',
            'Subgroups with KL > 0.05 (high divergence)',
            '',
            'Most Divergent Group (by KL)',
            'Least Divergent Group (by KL)'
        ],
        'Value': [
            f"{kl_df['kl_divergence'].mean():.6f}",
            f"{kl_df['kl_divergence'].median():.6f}",
            f"{kl_df['kl_divergence'].std():.6f}",
            f"{kl_df['kl_divergence'].min():.6f}",
            f"{kl_df['kl_divergence'].max():.6f}",
            '',
            f"{chi2_result.get('max_kl_ci', '')}" if chi2_result is not None else '',
            f"{kl_df['js_divergence'].mean():.6f}",
            f"{kl_df['js_divergence'].median():.6f}",
            '',
            f"{len(kl_df)}",
            f"{(kl_df['kl_divergence'] > 0.01).sum()} / {len(kl_df)}",
            f"{(kl_df['kl_divergence'] > 0.05).sum()} / {len(kl_df)}",
            '',
            kl_df.loc[kl_df['kl_divergence'].idxmax(), 'demo_subgroup'],
            kl_df.loc[kl_df['kl_divergence'].idxmin(), 'demo_subgroup']
        ]
    }
    
    # Add statistical test results if provided
    if chi2_result is not None:
        summary_data['Metric'].extend([
            '',
            'Chi-squared Test',
            'Chi-squared statistic',
            'p-value',
            'Significance (α=0.05)',
            'Cramer\'s V (Effect Size)'
        ])
        summary_data['Value'].extend([
            '',
            '',
            f"{chi2_result['chi2']:.4f}",
            f"{chi2_result['p_value']:.6e}",
            'Significant' if chi2_result['p_value'] < 0.05 else 'Not significant',
            f"{chi2_result['cramers_v']:.4f}"
        ])
    
    return pd.DataFrame(summary_data)

def bootstrap_max_kl_ci(df, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> str:
    """
    Bootstrap a 95% CI for the *maximum* KL divergence across the 8 subgroups.
    Resamples individual rows (patient profiles) with replacement and recomputes KLs.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return ""
    max_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx]
        kl_b = calculate_kl_divergence_cross_vignette(df_b)
        max_vals.append(float(kl_b["kl_divergence"].max()))
    alpha = (1 - ci) / 2
    lo = float(np.quantile(max_vals, alpha))
    hi = float(np.quantile(max_vals, 1 - alpha))
    return f"[{lo:.6f}, {hi:.6f}]"

def create_kl_divergence_barplot(kl_df, save_path=None):
    """
    Create a bar plot showing KL divergence for each demographic subgroup.
    
    Args:
        kl_df: DataFrame with KL divergence results
        save_path: Optional path to save the figure
    """
    # Sort by KL divergence
    kl_df_sorted = kl_df.sort_values('kl_divergence', ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(kl_df_sorted)), kl_df_sorted['kl_divergence'], 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Color bars by gender
    colors = ['lightcoral' if 'woman' in subgroup else 'steelblue' 
              for subgroup in kl_df_sorted['demo_subgroup']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xticks(range(len(kl_df_sorted)), kl_df_sorted['demo_subgroup'], 
               rotation=45, ha='right')
    plt.ylabel('KL Divergence D(P || Q)', fontweight='bold', fontsize=12)
    plt.xlabel('Demographic Subgroup', fontweight='bold', fontsize=12)
    plt.title('Cross-Vignette KL Divergence by Demographic Group\n' +
              '(Higher = More Different from Overall Dosage Distribution)',
              fontweight='bold', fontsize=14)
    
    # Add horizontal line at 0.01 (notable divergence threshold)
    plt.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, 
                label='Notable Divergence (0.01)', alpha=0.7)
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                label='High Divergence (0.05)', alpha=0.7)
    
    # Add legend for gender colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Male'),
        Patch(facecolor='lightcoral', label='Female'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, 
                   label='Notable (0.01)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
                   label='High (0.05)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved KL divergence bar plot to {save_path}")
    
    plt.close()

def create_kl_heatmap(kl_df, save_path=None):
    """
    Create a heatmap showing KL divergence by race and gender.
    
    Args:
        kl_df: DataFrame with KL divergence results
        save_path: Optional path to save the figure
    """
    # Pivot to create race × gender matrix
    pivot = kl_df.pivot(index='race', columns='gender', values='kl_divergence')
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'KL Divergence D(P || Q)'}, 
                linewidths=1, linecolor='white')
    
    plt.title('Cross-Vignette KL Divergence by Demographics\n' +
              '(Higher = More Different from Overall Distribution)',
              fontweight='bold', fontsize=14)
    plt.xlabel('Gender', fontweight='bold', fontsize=12)
    plt.ylabel('Race', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved KL divergence heatmap to {save_path}")
    
    plt.close()

def create_distribution_comparison_plot(kl_df, save_path=None):
    """
    Create a grouped bar plot comparing P distributions for each subgroup vs Q.
    
    Args:
        kl_df: DataFrame with KL divergence results and distributions
        save_path: Optional path to save the figure
    """
    dosage_categories = ['None', 'Low', 'Medium', 'High']
    
    # Sort by KL divergence
    kl_df_sorted = kl_df.sort_values('kl_divergence', ascending=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(kl_df_sorted.iterrows()):
        if idx >= 8:  # Only show top 8
            break
        
        ax = axes[idx]
        
        # Extract P and Q distributions
        P = [row['P_None'], row['P_Low'], row['P_Medium'], row['P_High']]
        Q = [row['Q_None'], row['Q_Low'], row['Q_Medium'], row['Q_High']]
        
        x = np.arange(len(dosage_categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, P, width, label='Subgroup (P)', 
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, Q, width, label='Reference (Q)', 
                       color='lightgray', alpha=0.8)
        
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_title(f"{row['demo_subgroup']}\nKL={row['kl_divergence']:.4f}", 
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dosage_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(kl_df_sorted), 8):
        axes[idx].set_visible(False)
    
    plt.suptitle('Dosage Distribution Comparison: Each Subgroup (P) vs. Reference (Q)\n' +
                 'Top 8 Most Divergent Groups',
                 fontweight='bold', fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution comparison plot to {save_path}")
    
    plt.close()

def create_kl_by_opioid_status_plot(kl_by_opioid_df, save_path=None):
    """
    Create a faceted plot showing KL divergence by opioid status.
    
    Args:
        kl_by_opioid_df: DataFrame with KL divergence by opioid status
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, opioid_status in enumerate(sorted(kl_by_opioid_df['opioid_status'].unique())):
        ax = axes[idx]
        df_subset = kl_by_opioid_df[kl_by_opioid_df['opioid_status'] == opioid_status]
        df_subset = df_subset.sort_values('kl_divergence', ascending=False)
        
        # Create bar plot
        colors = ['lightcoral' if 'woman' in subgroup else 'steelblue' 
                  for subgroup in df_subset['demo_subgroup']]
        
        bars = ax.bar(range(len(df_subset)), df_subset['kl_divergence'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xticks(range(len(df_subset)))
        ax.set_xticklabels(df_subset['demo_subgroup'], rotation=45, ha='right')
        ax.set_ylabel('KL Divergence', fontweight='bold', fontsize=12)
        ax.set_title(f'{opioid_status}', fontweight='bold', fontsize=14)
        ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Male'),
        Patch(facecolor='lightcoral', label='Female')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.suptitle('Cross-Vignette KL Divergence by Opioid Status',
                 fontweight='bold', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved KL by opioid status plot to {save_path}")
    
    plt.close()

def save_table_to_csv(table, filepath):
    """Save table to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    table.to_csv(filepath, index=False)
    print(f"Saved table to {filepath}")

def run_kl_analysis_for_model(csv_path, output_dir):
    # 1. Load Data
    print("\nLoading 3-dosage experiment data...")
    df = load_three_dosage_data(csv_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Unique vignettes: {df['vignette_idx'].nunique()}")
    print(f"Demographic subgroups: {df['demo_subgroup'].nunique()}")
    print(f"\nDosage distribution overall:")
    print(df['dosage_category'].value_counts())
    print()
    
    # 2. Calculate KL Divergence (Overall)
    print("="*80)
    print("CALCULATING KL DIVERGENCE - OVERALL ANALYSIS")
    print("="*80)
    
    kl_df = calculate_kl_divergence_cross_vignette(df)
    
    print("\nKL Divergence Results (sorted by divergence):")
    print(kl_df.sort_values('kl_divergence', ascending=False)[
        ['demo_subgroup', 'kl_divergence', 'js_divergence', 'n_samples']
    ].to_string(index=False))
    print()
    
    # 2.5. Statistical Significance Test
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*80)
    
    # Create contingency table: dosage_category × demo_subgroup
    contingency = pd.crosstab(df['dosage_category'], df['demo_subgroup'])
    print("\nContingency Table (Dosage × Demographic Subgroup):")
    print(contingency)
    print()
    
    # Chi-squared test of independence
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"Chi-squared Test of Independence:")
    print(f"  Chi-squared statistic: {chi2:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.6e}")
    print(f"  Significance level (α = 0.05): {'Significant' if p_value < 0.05 else 'Not significant'}")
    
    if p_value < 0.001:
        sig_level = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "** (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "* (p < 0.05)"
    else:
        sig_level = "ns (not significant)"
    
    print(f"  Result: {sig_level}")
    print()
    
    # Effect size: Cramer's V
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    print(f"Effect Size (Cramer's V): {cramers_v:.4f}")
    print("  Interpretation:")
    if cramers_v < 0.1:
        print("    Negligible effect (V < 0.1)")
    elif cramers_v < 0.3:
        print("    Small effect (0.1 ≤ V < 0.3)")
    elif cramers_v < 0.5:
        print("    Medium effect (0.3 ≤ V < 0.5)")
    else:
        print("    Large effect (V ≥ 0.5)")
    print()
    
    # Store chi2 results for summary table
    chi2_result = {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v
    }

    # 2.6. Bootstrap CI for max KL (descriptive uncertainty)
    try:
        max_kl_ci = bootstrap_max_kl_ci(df, n_boot=2000, ci=0.95, seed=0)
    except Exception as e:
        print(f"WARNING: Failed bootstrap max KL CI: {e}")
        max_kl_ci = ""
    chi2_result["max_kl_ci"] = max_kl_ci
    
    # 3. Create Summary Table
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary_table = create_kl_summary_table(kl_df, chi2_result=chi2_result)
    print(summary_table.to_string(index=False))
    save_table_to_csv(summary_table, os.path.join(output_dir, "kl_summary_ff.csv"))
    print()
    
    # Save detailed results
    save_table_to_csv(kl_df, os.path.join(output_dir, "kl_divergence_details_ff.csv"))
    
    # 4. Calculate KL Divergence by Opioid Status
    print("="*80)
    print("CALCULATING KL DIVERGENCE - BY OPIOID STATUS")
    print("="*80)
    
    kl_by_opioid_df = calculate_kl_by_opioid_status(df)
    save_table_to_csv(kl_by_opioid_df, os.path.join(output_dir, "kl_divergence_by_opioid_status_ff.csv"))
    
    # 5. Generate Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_kl_divergence_barplot(kl_df, save_path=os.path.join(output_dir, "kl_divergence_barplot_ff.png"))
    create_kl_heatmap(kl_df, save_path=os.path.join(output_dir, "kl_divergence_heatmap_ff.png"))
    create_distribution_comparison_plot(kl_df, save_path=os.path.join(output_dir, "distribution_comparison_ff.png"))
    create_kl_by_opioid_status_plot(kl_by_opioid_df, save_path=os.path.join(output_dir, "kl_by_opioid_status_ff.png"))
    
    # 6. Key Findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    most_divergent = kl_df.loc[kl_df['kl_divergence'].idxmax()]
    least_divergent = kl_df.loc[kl_df['kl_divergence'].idxmin()]
    
    print(f"\nMost divergent group: {most_divergent['demo_subgroup']}")
    print(f"  KL Divergence: {most_divergent['kl_divergence']:.6f}")
    print(f"  Distribution: None={most_divergent['P_None']:.3f}, Low={most_divergent['P_Low']:.3f}, " +
          f"Med={most_divergent['P_Medium']:.3f}, High={most_divergent['P_High']:.3f}")
    
    print(f"\nLeast divergent group: {least_divergent['demo_subgroup']}")
    print(f"  KL Divergence: {least_divergent['kl_divergence']:.6f}")
    print(f"  Distribution: None={least_divergent['P_None']:.3f}, Low={least_divergent['P_Low']:.3f}, " +
          f"Med={least_divergent['P_Medium']:.3f}, High={least_divergent['P_High']:.3f}")
    
    print(f"\nGroups with notable divergence (KL > 0.01): {(kl_df['kl_divergence'] > 0.01).sum()} / {len(kl_df)}")
    print(f"Groups with high divergence (KL > 0.05): {(kl_df['kl_divergence'] > 0.05).sum()} / {len(kl_df)}")

# Main Execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CROSS-VIGNETTE KL DIVERGENCE ANALYSIS")
    print("="*80)
    print("\nThis analysis measures whether demographic subgroups have systematically")
    print("different dosage distribution patterns across all clinical scenarios.")
    print("\nKL Divergence D(P || Q) quantifies how much the subgroup's dosage distribution")
    print("(P) differs from the overall population distribution (Q).")
    print("="*80)
    
    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    analysis_results_base = os.path.join(root, "analysis_results", "kl_divergence")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"\nWARNING: No model folders found under: {experiment_results_dir}")
        sys.exit(0)

    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        output_dir = os.path.join(analysis_results_base, model_id)
        os.makedirs(output_dir, exist_ok=True)

        csv_path = find_tier_csv(model_dir, 3)
        print("\n" + "="*80)
        print(f"MODEL: {model_id}")
        print("="*80)
        print(f"Tier 3 input: {csv_path}")
        print(f"Output dir:   {output_dir}")

        if not csv_path or not os.path.exists(csv_path):
            print("WARNING: Tier 3 CSV not found for this model. Skipping.")
            continue

        run_kl_analysis_for_model(csv_path, output_dir)

