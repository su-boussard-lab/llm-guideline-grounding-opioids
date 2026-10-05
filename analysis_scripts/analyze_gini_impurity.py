import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import glob

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

def _fmt_count_rate_ci(k: int, n: int) -> str:
    """Format 'k / n (p%; 95% CI [lo, hi])'."""
    if n <= 0:
        return f"{k} / {n} (NA)"
    p = 100.0 * (k / n)
    lo, hi = _wilson_ci(k, n)
    return f"{k} / {n} ({p:.1f}%; 95% CI [{lo*100:.1f}, {hi*100:.1f}])"

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
    """
    Find a tier CSV within a model directory.
    Accepts minor naming differences (e.g., llama tier1 withlogprobs).
    """
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


def calculate_gini_impurity(dosage_counts):
    """
    Calculate Gini Impurity for a set of dosage decisions.
    
    Gini Impurity (IG) = 1 - Σ(p_i^2)
    where p_i is the fraction of subgroups receiving dosage i
    
    IG = 0: Perfect agreement (all subgroups chose same dosage)
    IG ≈ 0.75: Maximum disagreement for 4 categories (uniform distribution)
    
    Args:
        dosage_counts: Series with dosage categories as index and counts as values
        
    Returns:
        float: Gini Impurity score (0 = perfect agreement)
    """
    total = dosage_counts.sum()
    if total == 0:
        return 0.0
    
    # Calculate proportion for each dosage category
    proportions = dosage_counts / total
    
    # Gini Impurity = 1 - sum of squared proportions
    gini = 1 - (proportions ** 2).sum()
    
    return gini

def load_tier_data(filepath, tier=3):
    """
    Load experiment results for Gini Impurity analysis.
    
    Args:
        filepath: Path to the CSV file
        tier: Experiment tier (1, 2, or 3)
        
    Returns:
        DataFrame with cleaned data
    """
    df = pd.read_csv(filepath)

    dosage_col = _detect_dosage_column(df)
    # Clean up dosage labels (strip trailing period from "Yes." style artifacts if any)
    raw_dosage = df[dosage_col].astype(str).str.replace(".", "", regex=False).str.strip()

    # Standardize dosage categories based on tier
    def standardize_dosage(dosage_str):
        dosage_lower = str(dosage_str).lower()
        if 'low' in dosage_lower:
            return 'Low'
        elif 'medium' in dosage_lower:
            return 'Medium'
        elif 'high' in dosage_lower:
            return 'High'
        elif 'none' in dosage_lower:
            return 'None'
        else:
            return 'Other'

    df['dosage_category'] = raw_dosage.apply(standardize_dosage)
    
    # Rename columns for clarity
    if tier == 1:
        # Tier 1: demographics only, no risk factors; use placeholders for groupby compatibility
        df_analysis = pd.DataFrame({
            'vignette_idx': df['vignette_idx'],
            'race': df['race'],
            'gender': df['gender'],
            'opioid_status': 'N/A',
            'mental_health': 'N/A',
            'preop_pain': 'N/A',
            'context': df['context'] if 'context' in df.columns else '',
            'dosage_category': df['dosage_category']
        })
    elif tier == 3:
        df_analysis = pd.DataFrame({
            'vignette_idx': df['vignette_idx'],
            'race': df['race'],
            'gender': df['gender'],
            'opioid_status': df['risk_op'],
            'mental_health': df['risk_mh'],
            'preop_pain': df['risk_pain'],
            'context': df['context'],
            'dosage_category': df['dosage_category']
        })
    elif tier == 2:
        df_analysis = pd.DataFrame({
            'vignette_idx': df['vignette_idx'],
            'race': df['race'],
            'gender': df['gender'],
            'opioid_status': df['risk_op'],
            'mental_health': df['risk_mh'],
            'preop_pain': df['risk_pain'],
            'context': df['context'],
            'dosage_category': df['dosage_category']
        })
    else:
        raise ValueError(f"Unsupported tier: {tier}. Only tier 1, 2, and 3 are supported.")
    
    return df_analysis

def analyze_vignette_gini_impurity(df):
    """
    Calculate Gini Impurity for each vignette to measure dosage decision consistency
    across race/gender subgroups.
    
    For each vignette with identical clinical presentation and risk factors,
    we have 8 subgroups (4 races × 2 genders). Gini Impurity measures how
    much disagreement exists in dosage decisions across these subgroups.
    
    Args:
        df: DataFrame with vignette data
        
    Returns:
        DataFrame with Gini Impurity scores per vignette
    """
    results = []
    
    # Group by vignette and risk factors (each unique clinical scenario)
    vignette_groups = df.groupby(['vignette_idx', 'opioid_status', 'mental_health', 'preop_pain'])
    
    for (vignette_idx, opioid_status, mental_health, preop_pain), vignette_data in vignette_groups:
        # Count dosage decisions across all 8 demographic subgroups
        dosage_counts = vignette_data['dosage_category'].value_counts()
        
        # Calculate Gini Impurity
        gini = calculate_gini_impurity(dosage_counts)
        
        # Get dosage breakdown by race and gender
        by_race = vignette_data.groupby('race')['dosage_category'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        )
        by_gender = vignette_data.groupby('gender')['dosage_category'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        )
        
        # Count unique dosages chosen
        n_unique_dosages = vignette_data['dosage_category'].nunique()
        
        # Get the most common dosage
        modal_dosage = dosage_counts.idxmax() if len(dosage_counts) > 0 else 'N/A'
        modal_count = dosage_counts.max() if len(dosage_counts) > 0 else 0
        
        results.append({
            'vignette_idx': vignette_idx,
            'opioid_status': opioid_status,
            'mental_health': mental_health,
            'preop_pain': preop_pain,
            'gini_impurity': gini,
            'n_subgroups': len(vignette_data),
            'n_unique_dosages': n_unique_dosages,
            'modal_dosage': modal_dosage,
            'modal_count': modal_count,
            'dosage_distribution': dosage_counts.to_dict(),
            # Dosage by race
            'asian_dosage': by_race.get('Asian', 'N/A'),
            'black_dosage': by_race.get('Black', 'N/A'),
            'hispanic_dosage': by_race.get('Hispanic', 'N/A'),
            'white_dosage': by_race.get('White', 'N/A'),
            # Dosage by gender
            'man_dosage': by_gender.get('man', 'N/A'),
            'woman_dosage': by_gender.get('woman', 'N/A')
        })
    
    return pd.DataFrame(results)

def create_gini_summary_table(gini_df, tier=3):
    """
    Create a summary table of Gini Impurity statistics.
    
    Args:
        gini_df: DataFrame with Gini Impurity scores
        tier: Experiment tier (1, 2, or 3) - affects max theoretical text
        
    Returns:
        DataFrame with summary statistics
    """
    # Interpretation thresholds
    perfect_agreement = (gini_df['gini_impurity'] == 0).sum()
    low_disagreement = ((gini_df['gini_impurity'] > 0) & (gini_df['gini_impurity'] <= 0.25)).sum()
    moderate_disagreement = ((gini_df['gini_impurity'] > 0.25) & (gini_df['gini_impurity'] <= 0.5)).sum()
    high_disagreement = (gini_df['gini_impurity'] > 0.5).sum()
    n_v = len(gini_df)
    
    max_theo_label = "Max Theoretical Gini (uniform 4-category)" if tier == 3 else "Max Theoretical Gini (uniform 2-category)"
    max_theo_val = "0.75" if tier == 3 else "0.50"
    
    summary = pd.DataFrame({
        'Metric': [
            'Mean Gini Impurity (0-1 scale)',
            'Median Gini Impurity',
            'Max Gini Impurity',
            'Min Gini Impurity',
            'Std Dev Gini Impurity',
            '',
            'Vignettes with Perfect Agreement (GI = 0)',
            'Vignettes with Low Disagreement (0 < GI ≤ 0.25)',
            'Vignettes with Moderate Disagreement (0.25 < GI ≤ 0.5)',
            'Vignettes with High Disagreement (GI > 0.5)',
            '',
            'Mean Number of Unique Dosages per Vignette',
            max_theo_label,
            'Total Vignettes Analyzed'
        ],
        'Value': [
            f"{gini_df['gini_impurity'].mean():.4f}",
            f"{gini_df['gini_impurity'].median():.4f}",
            f"{gini_df['gini_impurity'].max():.4f}",
            f"{gini_df['gini_impurity'].min():.4f}",
            f"{gini_df['gini_impurity'].std():.4f}",
            '',
            _fmt_count_rate_ci(int(perfect_agreement), int(n_v)),
            _fmt_count_rate_ci(int(low_disagreement), int(n_v)),
            _fmt_count_rate_ci(int(moderate_disagreement), int(n_v)),
            _fmt_count_rate_ci(int(high_disagreement), int(n_v)),
            '',
            f"{gini_df['n_unique_dosages'].mean():.2f}",
            max_theo_val,
            f"{n_v}"
        ]
    })
    
    return summary

def create_gini_barplot(gini_df, tier=3, save_path=None):
    """
    Create a bar plot showing Gini Impurity for vignettes with disagreement (GI > 0).
    Includes a summary annotation for the full dataset context.
    
    Args:
        gini_df: DataFrame with Gini Impurity scores
        tier: Experiment tier (1, 2, or 3) - affects max theoretical and title
        save_path: Path to save the plot
    """
    # Abbreviation helpers
    def abbreviate_opioid(status):
        if 'Naive' in status: return 'ON'
        elif 'Tolerant' in status: return 'OT'
        return status[:2]
    
    def abbreviate_mh(mh):
        mh_map = {
            'Anxiety Disorder': 'Anx',
            'Bipolar Disorder': 'Bip',
            'Major Depressive Disorder': 'MDD',
            'Schizophrenia': 'Sch',
            'no known mental health history': 'None'
        }
        return mh_map.get(mh, mh[:3])
    
    def abbreviate_pain(pain):
        if 'chronic' in pain.lower(): return 'CP'
        elif 'no' in pain.lower(): return 'NP'
        return pain[:2]
    
    # Set max theoretical based on tier
    max_theoretical = 0.75 if tier == 3 else 0.50
    tier_label = f"Tier {tier}"
    dosage_choices = "4 dosage choices" if tier == 3 else "2 dosage choices (Low/High)"
    
    # Filter to only vignettes with disagreement
    total_vignettes = len(gini_df)
    disagree_df = gini_df[gini_df['gini_impurity'] > 0].sort_values('gini_impurity', ascending=False).copy()
    n_agree = total_vignettes - len(disagree_df)
    
    if len(disagree_df) == 0:
        print("  No vignettes with disagreement — skipping Gini bar plot.")
        return
    
    # Create descriptive x-axis labels (Tier 1 has no risk factors)
    if tier == 1:
        x_labels = [f"V{row['vignette_idx']}" for _, row in disagree_df.iterrows()]
    else:
        x_labels = [
            f"V{row['vignette_idx']}-{abbreviate_opioid(row['opioid_status'])}-"
            f"{abbreviate_mh(row['mental_health'])}-{abbreviate_pain(row['preop_pain'])}"
            for _, row in disagree_df.iterrows()
        ]
    
    # Color code by disagreement level
    colors = []
    for gi in disagree_df['gini_impurity']:
        if gi <= 0.25:
            colors.append('#66c2a5')  # teal-green
        elif gi <= 0.5:
            colors.append('#fc8d62')  # orange
        else:
            colors.append('#e31a1c')  # red
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(disagree_df)), disagree_df['gini_impurity'],
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
    
    # FIXED: Add value labels on bars only if bar count is low to avoid overlap
    if len(disagree_df) <= 15:
        for i, val in enumerate(disagree_df['gini_impurity']):
            ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    xlabel = 'Vignette' if tier == 1 else 'Vignette (Idx-Opioid-MentalHealth-PreopPain)'
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
    ax.set_ylabel('Gini Impurity', fontweight='bold', fontsize=12)
    ax.set_title(f'{tier_label}: Within-Vignette Dosage Disagreement (Gini Impurity)\n'
                 f'Showing {len(disagree_df)} vignette(s) with GI > 0  |  '
                 f'{n_agree}/{total_vignettes} vignettes had perfect agreement',
                 fontweight='bold', fontsize=13, pad=15)
    
    # Reference lines
    ax.axhline(y=0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.6,
               label='Low Disagreement Threshold (0.25)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
               label='Moderate Disagreement Threshold (0.50)')
    if tier == 3:
        ax.axhline(y=0.75, color='darkred', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='Max Theoretical (0.75 for 4 categories)')
    elif tier == 1:
        ax.axhline(y=0.50, color='darkred', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='Max Theoretical (0.50 for 2 categories)')
    
    # Fixed X-axis labels: rotated and ha='right' to prevent overlap
    ax.set_xticks(range(len(disagree_df)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylim(0, max(disagree_df['gini_impurity'].max() * 1.3, 0.6))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Abbreviation key legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='ON=Opioid-Naive  OT=Opioid-Tolerant'),
        Patch(facecolor='white', edgecolor='black', label='MH: Anx=Anxiety, MDD=Major Depressive, Sch=Schizophrenia, None=No MH'),
        Patch(facecolor='white', edgecolor='black', label='Pain: CP=Chronic Preop, NP=No Preop'),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements,
              loc='upper right', fontsize=8, frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Gini bar plot to {save_path}")
    
    plt.close()

def create_gini_heatmap_by_risk(gini_df, save_path=None):
    """
    Create a heatmap showing Gini Impurity by opioid status and mental health.
    This helps identify which risk factor combinations lead to more disagreement.
    """
    # Abbreviate opioid status
    gini_df_copy = gini_df.copy()
    def abbreviate_opioid_status(status):
        if 'Naive' in status:
            return 'ON'
        elif 'Tolerant' in status:
            return 'OT'
        else:
            return status[:2]
    
    gini_df_copy['opioid_abbrev'] = gini_df_copy['opioid_status'].apply(abbreviate_opioid_status)
    
    # Pivot table: opioid status × mental health
    pivot = gini_df_copy.pivot_table(
        index='mental_health',
        columns='opioid_abbrev',
        values='gini_impurity',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Mean Gini Impurity'},
                vmin=0, vmax=0.75, ax=ax)
    
    ax.set_title('Mean Gini Impurity by Risk Factors\n(Lower = More Agreement)',
                fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Opioid Status', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mental Health Status', fontweight='bold', fontsize=12)
    
    # Add legend for opioid status abbreviations
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='ON = Opioid-Naive'),
        Patch(facecolor='white', edgecolor='black', label='OT = Opioid-Tolerant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.08), 
              ncol=2, frameon=True, fontsize=10, title='Opioid Status')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Gini heatmap to {save_path}")
    
    plt.close()

def create_dosage_distribution_heatmap(df, gini_df=None, tier=3, save_path=None):
    """
    Create a heatmap showing the dosage distribution for each vignette × demographic group.
    Only shows vignettes with disagreement (Gini > 0) to focus on meaningful differences.
    
    Args:
        df: Raw data DataFrame with individual records
        gini_df: DataFrame with Gini Impurity scores (used to filter for disagreement)
        tier: Experiment tier (2 or 3) - affects dosage encoding and legend
        save_path: Path to save the plot
    """
    df = df.copy()
    
    # Abbreviation helpers
    def abbreviate_opioid(status):
        if 'Naive' in status:
            return 'ON'
        elif 'Tolerant' in status:
            return 'OT'
        return status[:2]
    
    def abbreviate_mental_health(mh):
        mh_map = {
            'Anxiety Disorder': 'Anx',
            'Bipolar Disorder': 'Bip',
            'Major Depressive Disorder': 'MDD',
            'Schizophrenia': 'Sch',
            'no known mental health history': 'None'
        }
        return mh_map.get(mh, mh[:3])
    
    def abbreviate_preop_pain(pain):
        if 'chronic' in pain.lower():
            return 'CP'
        elif 'no' in pain.lower():
            return 'NP'
        return pain[:2]
    
    # Create demographic group label
    df['demo_group'] = df['race'] + '\n' + df['gender']
    
    # Create FULL unique vignette identifier (Tier 1: no risk factors; Tier 2/3: full clinical dimensions)
    if tier == 1:
        df['vignette_label'] = df['vignette_idx'].apply(lambda x: f"V{x}")
    else:
        df['vignette_label'] = df.apply(
            lambda r: f"V{r['vignette_idx']}-{abbreviate_opioid(r['opioid_status'])}-"
                      f"{abbreviate_mental_health(r['mental_health'])}-"
                      f"{abbreviate_preop_pain(r['preop_pain'])}",
            axis=1
        )
    
    # Filter to only vignettes with disagreement (Gini > 0) if gini_df is provided
    if gini_df is not None:
        disagree_df = gini_df[gini_df['gini_impurity'] > 0]
        if len(disagree_df) == 0:
            print("  No vignettes with disagreement found — skipping dosage distribution heatmap.")
            return
        
        # Build matching labels for the disagreeing vignettes
        if tier == 1:
            disagree_labels = disagree_df['vignette_idx'].apply(lambda x: f"V{x}").values
        else:
            disagree_labels = disagree_df.apply(
                lambda r: f"V{r['vignette_idx']}-{abbreviate_opioid(r['opioid_status'])}-"
                          f"{abbreviate_mental_health(r['mental_health'])}-"
                          f"{abbreviate_preop_pain(r['preop_pain'])}",
                axis=1
            ).values
        
        df = df[df['vignette_label'].isin(disagree_labels)]
        n_disagree = len(disagree_labels)
        print(f"  Filtering to {n_disagree} vignette(s) with disagreement (Gini > 0)")
    
    # Encode dosages based on tier
    if tier == 3:
        dosage_encoding = {'Low': 1, 'Medium': 2, 'High': 3, 'None': 0, 'Other': -1}
        colors = ['gray', 'lightgreen', 'gold', 'orangered']
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
        dosage_labels = {0: 'N', 1: 'L', 2: 'M', 3: 'H', -1: '?'}
        cbar_ticklabels = ['None', 'Low', 'Med', 'High']
        cbar_ticks = [0, 1, 2, 3]
    elif tier in (1, 2):
        dosage_encoding = {'Low': 1, 'High': 2, 'None': 0, 'Other': -1}
        colors = ['gray', 'lightgreen', 'orangered']
        boundaries = [-0.5, 0.5, 1.5, 2.5]
        dosage_labels = {0: 'N', 1: 'L', 2: 'H', -1: '?'}
        cbar_ticklabels = ['None', 'Low', 'High']
        cbar_ticks = [0, 1, 2]
    else:
        raise ValueError(f"Unsupported tier: {tier}")
    
    df['dosage_encoded'] = df['dosage_category'].map(dosage_encoding)
    
    # FIXED: Logic for colorbar
    if tier == 3:
        # For Tier 3, ensure all 4 categories are visible in the legend
        final_colors, final_ticklabels, final_ticks, final_boundaries = colors, cbar_ticklabels, cbar_ticks, boundaries
    else:
        # For Tier 2, only show what's present (removes unused 'None')
        present_values = sorted([v for v in df['dosage_encoded'].unique() if v in cbar_ticks])
        if len(present_values) > 0:
            final_colors = [colors[cbar_ticks.index(v)] for v in present_values]
            final_ticklabels = [cbar_ticklabels[cbar_ticks.index(v)] for v in present_values]
            final_ticks = present_values
            final_boundaries = [v - 0.5 for v in present_values] + [present_values[-1] + 0.5]
        else:
            final_colors, final_ticklabels, final_ticks, final_boundaries = colors, cbar_ticklabels, cbar_ticks, boundaries

    # Pivot: each row is a unique clinical scenario, each column is a demographic group
    pivot = df.pivot_table(
        index='vignette_label',
        columns='demo_group',
        values='dosage_encoded',
        aggfunc='first'
    )
    
    # Reorder columns
    race_order = ['Asian', 'Black', 'Hispanic', 'White']
    gender_order = ['man', 'woman']
    column_order = [f"{race}\n{gender}" for race in race_order for gender in gender_order]
    column_order = [col for col in column_order if col in pivot.columns]
    if len(column_order) > 0:
        pivot = pivot[column_order]
    
    # Dynamic figure height based on row count
    n_rows = len(pivot)
    fig_height = max(6, n_rows * 1.2 + 4)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Custom discrete colormap using only final colors
    from matplotlib.colors import BoundaryNorm, ListedColormap
    norm = BoundaryNorm(final_boundaries, len(final_colors))
    cmap = ListedColormap(final_colors)
    
    sns.heatmap(pivot, cmap=cmap, norm=norm,
                cbar_kws={'label': 'Dosage Level', 'ticks': final_ticks},
                linewidths=0.5, linecolor='white', ax=ax,
                annot=True, fmt='.0f')
    
    # Fix colorbar labels to match final categories
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(final_ticklabels)
    
    # Replace numeric annotations with dosage abbreviations
    for text in ax.texts:
        val = text.get_text()
        try:
            text.set_text(dosage_labels.get(int(float(val)), val))
            text.set_fontsize(11)
            text.set_fontweight('bold')
        except ValueError:
            pass
    
    tier_label = f"Tier {tier}"
    title_suffix = f" — {n_rows} vignette(s) with disagreement" if gini_df is not None else ""
    ax.set_title(f'{tier_label}: Dosage Decisions by Vignette and Demographics{title_suffix}\n'
                 '(Each row = identical clinical scenario, showing only discrepancies)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Demographic Group', fontweight='bold', fontsize=12)
    ax.set_ylabel('Vignette (Idx-Opioid-MentalHealth-PreopPain)', fontweight='bold', fontsize=11,
                  labelpad=10)
    
    # Y-axis label font size — readable now with fewer rows
    ax.tick_params(axis='y', labelsize=11)
    plt.yticks(rotation=0)
    
    # Legend for abbreviations
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='ON = Opioid-Naive'),
        Patch(facecolor='white', edgecolor='black', label='OT = Opioid-Tolerant'),
        Patch(facecolor='white', edgecolor='black', label='MH: Anx=Anxiety, Bip=Bipolar, MDD=Major Depressive, Sch=Schizophrenia, None=No MH'),
        Patch(facecolor='white', edgecolor='black', label='Pain: CP=Chronic Preop Pain, NP=No Preop Pain'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.15),
              ncol=1, frameon=True, fontsize=9, title='Abbreviation Key')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30, left=0.18)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved dosage distribution heatmap to {save_path}")
    
    plt.close()

def create_gini_violin_plot(gini_df, save_path=None):
    """
    Create violin plots showing Gini Impurity distribution by opioid status.
    Good for comparing overall disagreement patterns.
    """
    # Abbreviate opioid status
    gini_df_copy = gini_df.copy()
    def abbreviate_opioid_status(status):
        if 'Naive' in status:
            return 'ON'
        elif 'Tolerant' in status:
            return 'OT'
        else:
            return status[:2]
    
    gini_df_copy['opioid_abbrev'] = gini_df_copy['opioid_status'].apply(abbreviate_opioid_status)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.violinplot(data=gini_df_copy, x='opioid_abbrev', y='gini_impurity',
                   palette='Set2', ax=ax)
    sns.swarmplot(data=gini_df_copy, x='opioid_abbrev', y='gini_impurity',
                  color='black', alpha=0.5, size=6, ax=ax)
    
    ax.set_xlabel('Opioid Status', fontweight='bold', fontsize=12)
    ax.set_ylabel('Gini Impurity', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Gini Impurity by Opioid Status\n' +
                 '(Each point = one vignette)',
                 fontweight='bold', fontsize=14, pad=15)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5,
              label='Perfect Agreement')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5,
              label='High Disagreement')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for opioid status abbreviations
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='ON = Opioid-Naive'),
        Patch(facecolor='white', edgecolor='black', label='OT = Opioid-Tolerant')
    ]
    ax2 = ax.twiny()
    ax2.set_xticks([])
    ax2.legend(handles=legend_elements, loc='upper left', 
              ncol=2, frameon=True, fontsize=10, title='Opioid Status')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Gini violin plot to {save_path}")
    
    plt.close()

def save_table_to_csv(table, filepath):
    """Save table to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    table.to_csv(filepath, index=False)
    print(f"Saved table to {filepath}")

# Main Execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GINI IMPURITY ANALYSIS - WITHIN-VIGNETTE DOSAGE AGREEMENT")
    print("="*80)
    print("\nThis analysis measures dosage decision consistency within vignettes")
    print("using Gini Impurity (0 = perfect agreement, higher = more disagreement)")
    print("="*80)
    
    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    analysis_results_base = os.path.join(root, "analysis_results", "gini_impurity")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"\nWARNING: No model folders found under: {experiment_results_dir}")
        sys.exit(0)

    tiers = [
        (1, "Tier 1 (demographics only, 2 dosage choices: Low/High)"),
        (2, "Tier 2 (2 dosage choices: Low/High)"),
        (3, "Tier 3 (4 dosage choices: None/Low/Medium/High)"),
    ]

    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        output_dir = os.path.join(analysis_results_base, model_id)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"MODEL: {model_id}")
        print("="*80)

        for tier, tier_name in tiers:
            csv_path = find_tier_csv(model_dir, tier)
            suffix = f"_tier{tier}_ff"
        
            print("\n" + "="*80)
            print(f"ANALYZING {tier_name}")
            print("="*80)
        
            # 1. Load Data
            print(f"\nLoading {tier_name} data...")
        
            if not csv_path or not os.path.exists(csv_path):
                print(f"\nWARNING: File not found for {model_id} Tier {tier}")
                print("  Expected something like: results_post_op_*_tier{tier}_ff*.csv")
                print(f"  Searched under: {model_dir}")
                print(f"Skipping {tier_name} analysis.")
                continue
        
            print(f"Using input CSV: {csv_path}")
            df = load_tier_data(csv_path, tier=tier)
            print(f"Loaded {len(df)} records")
        
            # Print dosage distribution
            print("\nDosage category distribution:")
            print(df['dosage_category'].value_counts())
        
            # 2. Calculate Gini Impurity
            print("\n" + "-"*80)
            print("CALCULATING GINI IMPURITY FOR EACH VIGNETTE")
            print("-"*80)
        
            gini_df = analyze_vignette_gini_impurity(df)
            print(f"\nAnalyzed {len(gini_df)} unique vignettes")
        
            # 3. Summary Statistics
            print("\n" + "-"*80)
            print("GINI IMPURITY SUMMARY")
            print("-"*80)
        
            summary = create_gini_summary_table(gini_df, tier=tier)
            print("\n" + summary.to_string(index=False))
            save_table_to_csv(summary, os.path.join(output_dir, f"gini_impurity_summary{suffix}.csv"))
        
            # 4. Detailed Results
            print("\n" + "-"*80)
            print("DETAILED GINI IMPURITY BY VIGNETTE")
            print("-"*80)
        
            # Display top disagreements
            print("\nTop 5 vignettes with highest disagreement:")
            top_disagreement = gini_df.nlargest(5, 'gini_impurity')[
                ['vignette_idx', 'opioid_status', 'mental_health', 'gini_impurity',
                 'n_unique_dosages', 'modal_dosage']
            ]
            print(top_disagreement.to_string(index=False))
        
            print("\nTop 5 vignettes with lowest disagreement (highest agreement):")
            top_agreement = gini_df.nsmallest(5, 'gini_impurity')[
                ['vignette_idx', 'opioid_status', 'mental_health', 'gini_impurity',
                 'n_unique_dosages', 'modal_dosage']
            ]
            print(top_agreement.to_string(index=False))
        
            # Save detailed results
            save_table_to_csv(gini_df, os.path.join(output_dir, f"gini_impurity_details{suffix}.csv"))
        
            # 5. Visualizations
            print("\n" + "-"*80)
            print("GENERATING VISUALIZATIONS")
            print("-"*80)
        
            create_gini_barplot(gini_df, tier=tier, save_path=os.path.join(output_dir, f"gini_impurity_barplot{suffix}.png"))
            create_gini_heatmap_by_risk(gini_df, save_path=os.path.join(output_dir, f"gini_impurity_heatmap_by_risk{suffix}.png"))
            create_dosage_distribution_heatmap(df, gini_df=gini_df, tier=tier, save_path=os.path.join(output_dir, f"gini_dosage_distribution_heatmap{suffix}.png"))
            create_gini_violin_plot(gini_df, save_path=os.path.join(output_dir, f"gini_impurity_violin{suffix}.png"))
        
            # 6. Analysis by Opioid Status
            print("\n" + "-"*80)
            print("GINI IMPURITY BY OPIOID STATUS")
            print("-"*80)
        
            for status in df['opioid_status'].unique():
                gini_subset = gini_df[gini_df['opioid_status'] == status]
                print(f"\n{status}:")
                print(f"  Mean Gini Impurity: {gini_subset['gini_impurity'].mean():.4f}")
                print(f"  Median Gini Impurity: {gini_subset['gini_impurity'].median():.4f}")
                print(f"  Perfect Agreement: {(gini_subset['gini_impurity'] == 0).sum()} / {len(gini_subset)}")
                print(f"  High Disagreement (>0.5): {(gini_subset['gini_impurity'] > 0.5).sum()} / {len(gini_subset)}")
        
            print(f"\n{tier_name} analysis complete for model {model_id}!")
            print(f"Saved outputs to: {output_dir}")
    
    print("\n" + "="*80)
    print("ALL TIER ANALYSES COMPLETE!")
    print("="*80)
    print()