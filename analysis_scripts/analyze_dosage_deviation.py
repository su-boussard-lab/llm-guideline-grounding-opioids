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

def find_three_dosage_risk_csv(model_dir):
    """
    This deviation analysis expects the older 'risk_factors_3_dosages' style CSV.
    Try to locate an appropriate file inside the model folder.
    """
    patterns = [
        os.path.join(model_dir, "*risk_factors*3*dosag*.csv"),
        os.path.join(model_dir, "*risk_factors_3_dosages*.csv"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None

def standardize_dosage_to_numeric(dosage_str):
    """
    Convert dosage string to numerical scale for quantitative analysis.
    
    Scale:
    - None of Above = 0
    - Low = 1
    - Medium = 2
    - High = 3
    
    Args:
        dosage_str: Dosage string from experiment
        
    Returns:
        int: Numeric dosage level (0-3)
    """
    dosage_lower = str(dosage_str).lower()
    if 'none' in dosage_lower:
        return 0
    elif 'low' in dosage_lower:
        return 1
    elif 'medium' in dosage_lower:
        return 2
    elif 'high' in dosage_lower:
        return 3
    else:
        return np.nan

def load_three_dosage_data(filepath):
    """
    Load the 3-dosage experiment results.
    
    Args:
        filepath: Path to the 3-dosage CSV file
        
    Returns:
        DataFrame with cleaned data and numeric dosage levels
    """
    df = pd.read_csv(filepath)
    
    # Clean up dosage labels
    df['gpt4o_dosage'] = df['gpt4o_dosage'].str.replace('.', '', regex=False).str.strip()
    
    # Convert to numeric scale
    df['dosage_numeric'] = df['gpt4o_dosage'].apply(standardize_dosage_to_numeric)
    
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
        'dosage_str': df['gpt4o_dosage'],
        'dosage_numeric': df['dosage_numeric']
    })
    
    # Remove any rows with invalid dosages
    df_analysis = df_analysis.dropna(subset=['dosage_numeric'])
    
    return df_analysis

def calculate_within_vignette_deviations(df):
    """
    Calculate dosage deviations from vignette mean for each demographic subgroup.
    
    For each vignette (identical clinical scenario):
    1. Calculate mean dosage across all 8 demographic subgroups
    2. For each subgroup, calculate deviation from mean
    3. Negative deviation = under-prescribing relative to average
    4. Positive deviation = over-prescribing relative to average
    
    Args:
        df: DataFrame with dosage data
        
    Returns:
        DataFrame with deviation statistics per vignette
    """
    results = []
    
    # Group by vignette and risk factors (each unique clinical scenario)
    vignette_groups = df.groupby(['vignette_idx', 'opioid_status', 'mental_health', 'preop_pain'])
    
    for (vignette_idx, opioid_status, mental_health, preop_pain), vignette_data in vignette_groups:
        # Calculate vignette mean (the "fair" baseline for this scenario)
        vignette_mean = vignette_data['dosage_numeric'].mean()
        
        # Calculate deviation for each demographic subgroup
        for demo_subgroup in vignette_data['demo_subgroup'].unique():
            subgroup_data = vignette_data[vignette_data['demo_subgroup'] == demo_subgroup]
            
            if len(subgroup_data) > 0:
                subgroup_dosage = subgroup_data['dosage_numeric'].iloc[0]
                deviation = subgroup_dosage - vignette_mean
                
                # Extract race and gender
                race = subgroup_data['race'].iloc[0]
                gender = subgroup_data['gender'].iloc[0]
                
                results.append({
                    'vignette_idx': vignette_idx,
                    'opioid_status': opioid_status,
                    'mental_health': mental_health,
                    'preop_pain': preop_pain,
                    'demo_subgroup': demo_subgroup,
                    'race': race,
                    'gender': gender,
                    'vignette_mean_dosage': vignette_mean,
                    'subgroup_dosage': subgroup_dosage,
                    'deviation': deviation
                })
    
    return pd.DataFrame(results)

def aggregate_deviations_by_subgroup(deviation_df):
    """
    Aggregate deviations across all vignettes for each demographic subgroup.
    
    This reveals systematic bias patterns:
    - Consistently negative = under-prescribing across all scenarios
    - Consistently positive = over-prescribing across all scenarios
    - Near zero = fair treatment
    
    Args:
        deviation_df: DataFrame with per-vignette deviations
        
    Returns:
        DataFrame with aggregated statistics per subgroup
    """
    agg_stats = deviation_df.groupby(['race', 'gender', 'demo_subgroup']).agg({
        'deviation': ['mean', 'std', 'min', 'max', 'count'],
        'subgroup_dosage': 'mean',
        'vignette_mean_dosage': 'mean'
    }).reset_index()
    
    # Flatten column names
    agg_stats.columns = ['race', 'gender', 'demo_subgroup', 
                         'mean_deviation', 'std_deviation', 'min_deviation', 
                         'max_deviation', 'n_vignettes',
                         'mean_subgroup_dosage', 'mean_vignette_dosage']
    
    # Calculate consistency metrics
    # If always under-prescribed: all deviations should be negative
    # If always over-prescribed: all deviations should be positive
    agg_stats['always_under'] = (agg_stats['max_deviation'] <= 0).astype(int)
    agg_stats['always_over'] = (agg_stats['min_deviation'] >= 0).astype(int)
    agg_stats['consistent_direction'] = agg_stats['always_under'] | agg_stats['always_over']
    
    return agg_stats

def calculate_deviation_by_risk_factor(deviation_df):
    """
    Calculate average deviation for each demographic subgroup × risk factor combination.
    
    This shows whether bias patterns differ by clinical context.
    
    Args:
        deviation_df: DataFrame with per-vignette deviations
        
    Returns:
        DataFrame with deviations by subgroup and risk factors
    """
    # By opioid status
    by_opioid = deviation_df.groupby(['race', 'gender', 'opioid_status']).agg({
        'deviation': ['mean', 'std', 'count']
    }).reset_index()
    by_opioid.columns = ['race', 'gender', 'opioid_status', 
                         'mean_deviation', 'std_deviation', 'n_vignettes']
    
    # By mental health
    by_mh = deviation_df.groupby(['race', 'gender', 'mental_health']).agg({
        'deviation': ['mean', 'std', 'count']
    }).reset_index()
    by_mh.columns = ['race', 'gender', 'mental_health',
                     'mean_deviation', 'std_deviation', 'n_vignettes']
    
    return by_opioid, by_mh

def create_deviation_heatmap_overall(agg_stats, save_path=None):
    """
    Create heatmap showing average deviation for each demographic subgroup.
    
    This is the main visualization for detecting systematic bias.
    Red = under-prescribing, Blue = over-prescribing, White = fair
    """
    # Pivot for heatmap: race × gender
    pivot = agg_stats.pivot(index='race', columns='gender', values='mean_deviation')
    
    # Determine color scale range (symmetric around 0)
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmin, vmax = -max_abs, max_abs
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use diverging colormap: red (negative/under) to white (0) to blue (positive/over)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu',
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Mean Deviation from Vignette Average'},
                linewidths=1, linecolor='gray', ax=ax)
    
    ax.set_title('Systematic Dosage Bias Across All Vignettes\n' +
                 'Red = Under-prescribing | Blue = Over-prescribing | White = Fair',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Gender', fontweight='bold', fontsize=11)
    ax.set_ylabel('Race', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overall deviation heatmap to {save_path}")
    
    plt.close()

def create_deviation_heatmap_by_opioid(deviation_df, save_path=None):
    """
    Create heatmap showing deviations separately for opioid-naive vs. tolerant.
    
    This reveals whether bias patterns differ by opioid status.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, status in enumerate(sorted(deviation_df['opioid_status'].unique())):
        subset = deviation_df[deviation_df['opioid_status'] == status]
        
        # Aggregate by subgroup
        agg = subset.groupby(['race', 'gender'])['deviation'].mean().reset_index()
        pivot = agg.pivot(index='race', columns='gender', values='deviation')
        
        # Symmetric color scale
        max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
        if max_abs == 0:
            max_abs = 0.1  # Avoid division by zero
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu',
                    center=0, vmin=-max_abs, vmax=max_abs,
                    cbar_kws={'label': 'Mean Deviation'},
                    linewidths=1, linecolor='gray', ax=axes[idx])
        
        axes[idx].set_title(f'{status}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Gender', fontweight='bold')
        axes[idx].set_ylabel('Race' if idx == 0 else '', fontweight='bold')
    
    fig.suptitle('Dosage Deviation by Opioid Status\n' +
                 'Red = Under-prescribing | Blue = Over-prescribing',
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved deviation by opioid status heatmap to {save_path}")
    
    plt.close()

def create_deviation_boxplot(deviation_df, save_path=None):
    """
    Create box plots showing distribution of deviations for each demographic group.
    
    This shows not just the average bias, but also the consistency/variability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # By race
    sns.boxplot(data=deviation_df, x='race', y='deviation', palette='Set2', ax=ax1)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Race', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Deviation from Vignette Mean', fontweight='bold', fontsize=11)
    ax1.set_title('Dosage Deviation Distribution by Race', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # By gender
    sns.boxplot(data=deviation_df, x='gender', y='deviation', palette='Set2', ax=ax2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Gender', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Deviation from Vignette Mean', fontweight='bold', fontsize=11)
    ax2.set_title('Dosage Deviation Distribution by Gender', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved deviation boxplot to {save_path}")
    
    plt.close()

def create_vignette_specific_deviation_heatmap(deviation_df, save_path=None):
    """
    Create heatmap showing deviations for each vignette × demographic combination.
    
    This is the detailed view showing which specific scenarios have the most bias.
    """
    # Create vignette label
    deviation_df['vignette_label'] = (deviation_df['vignette_idx'].astype(str) + '-' + 
                                      deviation_df['opioid_status'].str[:3])
    
    # Pivot: vignette × demographic subgroup
    pivot = deviation_df.pivot_table(
        index='vignette_label',
        columns='demo_subgroup',
        values='deviation',
        aggfunc='mean'
    )
    
    # Reorder columns by race then gender
    race_order = ['Asian', 'Black', 'Hispanic', 'White']
    gender_order = ['man', 'woman']
    column_order = [f"{race}_{gender}" for race in race_order for gender in gender_order]
    column_order = [col for col in column_order if col in pivot.columns]
    pivot = pivot[column_order]
    
    # Symmetric color scale
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
    if max_abs == 0:
        max_abs = 0.1
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu',
                center=0, vmin=-max_abs, vmax=max_abs,
                cbar_kws={'label': 'Deviation from Vignette Mean'},
                linewidths=0.5, linecolor='white', ax=ax)
    
    ax.set_title('Vignette-Specific Dosage Deviations by Demographics\n' +
                 'Red = Under-prescribing | Blue = Over-prescribing',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Demographic Subgroup', fontweight='bold', fontsize=11)
    ax.set_ylabel('Vignette (Index-OpioidStatus)', fontweight='bold', fontsize=11)
    
    # Improve x-axis labels
    ax.set_xticklabels([col.replace('_', '\n') for col in pivot.columns], 
                       rotation=0, ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved vignette-specific deviation heatmap to {save_path}")
    
    plt.close()

def perform_statistical_tests(deviation_df, agg_stats):
    """
    Perform statistical tests to determine if deviations are significantly different from zero
    and if they differ between demographic groups.
    
    Returns:
        DataFrame with test results
    """
    results = []
    
    # Test each demographic subgroup against zero (one-sample t-test)
    for _, row in agg_stats.iterrows():
        subgroup = row['demo_subgroup']
        subgroup_deviations = deviation_df[deviation_df['demo_subgroup'] == subgroup]['deviation']
        
        if len(subgroup_deviations) > 1:
            # One-sample t-test: is mean deviation significantly different from 0?
            t_stat, p_value = stats.ttest_1samp(subgroup_deviations, 0)
            
            results.append({
                'demo_subgroup': subgroup,
                'race': row['race'],
                'gender': row['gender'],
                'mean_deviation': row['mean_deviation'],
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No',
                'n_vignettes': row['n_vignettes']
            })
    
    return pd.DataFrame(results)

def create_summary_table(agg_stats, test_results):
    """
    Create a comprehensive summary table of deviation analysis.
    """
    # Merge aggregated stats with test results
    summary = agg_stats.merge(test_results[['demo_subgroup', 't_statistic', 'p_value', 'significant']], 
                              on='demo_subgroup', how='left')
    
    # Select and order columns
    summary = summary[[
        'race', 'gender', 'demo_subgroup',
        'mean_deviation', 'std_deviation', 
        'min_deviation', 'max_deviation',
        'always_under', 'always_over',
        't_statistic', 'p_value', 'significant',
        'n_vignettes'
    ]]
    
    return summary

def save_table_to_csv(table, filepath):
    """Save table to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    table.to_csv(filepath, index=False)
    print(f"Saved table to {filepath}")

# Main Execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DOSAGE DEVIATION ANALYSIS - SYSTEMATIC BIAS DETECTION")
    print("="*80)
    print("\nThis analysis measures whether specific demographic groups are")
    print("systematically under- or over-prescribed relative to the vignette average.")
    print("\nDeviation Score:")
    print("  Negative (Red) = Under-prescribing relative to scenario average")
    print("  Zero (White) = Fair treatment (receives average dosage)")
    print("  Positive (Blue) = Over-prescribing relative to scenario average")
    print("="*80)
    
    root = _project_root()
    experiment_results_dir = os.path.join(root, "experiment_results")
    analysis_results_base = os.path.join(root, "analysis_results", "deviation_analysis")

    model_ids = discover_models(experiment_results_dir)
    if not model_ids:
        print(f"\nWARNING: No model folders found under: {experiment_results_dir}")
        sys.exit(0)

    for model_id in model_ids:
        model_dir = os.path.join(experiment_results_dir, model_id)
        output_dir = os.path.join(analysis_results_base, model_id)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load Data
        print("\n" + "="*80)
        print(f"MODEL: {model_id}")
        print("="*80)
        print("\nLoading 3-dosage experiment data...")

        csv_path = find_three_dosage_risk_csv(model_dir)
        print(f"Input CSV:   {csv_path}")
        print(f"Output dir:  {output_dir}")

        if not csv_path or not os.path.exists(csv_path):
            print("WARNING: No risk_factors_3_dosages-style CSV found for this model. Skipping.")
            continue

        df = load_three_dosage_data(csv_path)
        print(f"Loaded {len(df)} records")
        print(f"\nDosage distribution:")
        print(df['dosage_str'].value_counts())
        print(f"\nNumeric dosage scale: None=0, Low=1, Medium=2, High=3")
        print(f"Mean numeric dosage: {df['dosage_numeric'].mean():.2f}")
    
    # 2. Calculate Within-Vignette Deviations
    print("\n" + "="*80)
    print("CALCULATING DEVIATIONS FROM VIGNETTE MEAN")
    print("="*80)
    
    deviation_df = calculate_within_vignette_deviations(df)
    print(f"\nCalculated deviations for {len(deviation_df)} vignette × subgroup combinations")
    print(f"Mean deviation across all: {deviation_df['deviation'].mean():.4f}")
    print(f"Std dev of deviations: {deviation_df['deviation'].std():.4f}")
    
    # 3. Aggregate by Demographic Subgroup
    print("\n" + "="*80)
    print("AGGREGATING DEVIATIONS BY DEMOGRAPHIC SUBGROUP")
    print("="*80)
    
    agg_stats = aggregate_deviations_by_subgroup(deviation_df)
    print("\nAverage deviation by subgroup:")
    print(agg_stats[['race', 'gender', 'mean_deviation', 'std_deviation']].to_string(index=False))
    
    # Identify consistently biased groups
    always_under = agg_stats[agg_stats['always_under'] == 1]
    always_over = agg_stats[agg_stats['always_over'] == 1]
    
    if len(always_under) > 0:
        print(f"\n⚠️  Groups ALWAYS under-prescribed (in all vignettes):")
        print(always_under[['demo_subgroup', 'mean_deviation']].to_string(index=False))
    
    if len(always_over) > 0:
        print(f"\n⚠️  Groups ALWAYS over-prescribed (in all vignettes):")
        print(always_over[['demo_subgroup', 'mean_deviation']].to_string(index=False))
    
    # 4. Statistical Significance Tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    print("Testing if deviations are significantly different from zero...")
    
    test_results = perform_statistical_tests(deviation_df, agg_stats)
    significant = test_results[test_results['significant'] == 'Yes']
    
    print(f"\nGroups with statistically significant bias (p < 0.05):")
    if len(significant) > 0:
        print(significant[['demo_subgroup', 'mean_deviation', 'p_value']].to_string(index=False))
    else:
        print("None found.")
    
    # 5. Create Summary Tables
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLES")
    print("="*80)
    
    summary_table = create_summary_table(agg_stats, test_results)
    save_table_to_csv(summary_table, os.path.join(output_dir, "deviation_summary.csv"))
    save_table_to_csv(deviation_df, os.path.join(output_dir, "deviation_details.csv"))

    # 6. By Risk Factor
    by_opioid, by_mh = calculate_deviation_by_risk_factor(deviation_df)
    save_table_to_csv(by_opioid, os.path.join(output_dir, "deviation_by_opioid_status.csv"))
    save_table_to_csv(by_mh, os.path.join(output_dir, "deviation_by_mental_health.csv"))
    
    print("\nDeviation by Opioid Status:")
    print(by_opioid.to_string(index=False))
    
    # 7. Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_deviation_heatmap_overall(agg_stats, save_path=os.path.join(output_dir, "deviation_heatmap_overall.png"))
    create_deviation_heatmap_by_opioid(deviation_df, save_path=os.path.join(output_dir, "deviation_heatmap_by_opioid.png"))
    create_deviation_boxplot(deviation_df, save_path=os.path.join(output_dir, "deviation_boxplot.png"))
    create_vignette_specific_deviation_heatmap(deviation_df, save_path=os.path.join(output_dir, "deviation_heatmap_vignette_specific.png"))
    
    # 8. Key Findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Most under-prescribed
    most_under = agg_stats.nsmallest(3, 'mean_deviation')
    print("\nTop 3 most under-prescribed groups:")
    print(most_under[['demo_subgroup', 'mean_deviation']].to_string(index=False))
    
    # Most over-prescribed
    most_over = agg_stats.nlargest(3, 'mean_deviation')
    print("\nTop 3 most over-prescribed groups:")
    print(most_over[['demo_subgroup', 'mean_deviation']].to_string(index=False))
    
    # Range of bias
    bias_range = agg_stats['mean_deviation'].max() - agg_stats['mean_deviation'].min()
    print(f"\nBias range across all groups: {bias_range:.4f} dosage levels")
    print(f"(This represents the gap between most privileged and most disadvantaged groups)")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE FOR MODEL: {model_id}")
    print("="*80)
    print(f"Saved outputs to: {output_dir}")
    print()

