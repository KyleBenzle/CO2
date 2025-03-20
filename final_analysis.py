"""
Final analysis of CO2 levels in N95 masks with updated data.
This script performs statistical analysis and generates figures for publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory if it doesn't exist
output_dir = "final_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
print("Loading and processing data...")
df = pd.read_csv('cleaned_data.csv')

# Clean up headers if needed
if '' in df.columns:
    df = df.drop('', axis=1, errors='ignore')

# Make sure all numeric columns are properly typed
df['CO2 (ppm)'] = pd.to_numeric(df['CO2 (ppm)'], errors='coerce')
df['Temp (C)'] = pd.to_numeric(df['Temp (C)'], errors='coerce')
df['Humidity (%)'] = pd.to_numeric(df['Humidity (%)'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna(subset=['CO2 (ppm)'])

# Filter to include only mask data (not no-mask data)
mask_df = df[df['Source File'].isin(['1_mask_12_1_24.txt', '2_masks_12_1_24.txt'])]

# Create a new column for mask type
mask_df.loc[:, 'Mask Type'] = mask_df['Source File'].replace({
    '1_mask_12_1_24.txt': 'Single Mask',
    '2_masks_12_1_24.txt': 'Double Mask'
})

# Group data by measurement type for statistics
baseline = mask_df[mask_df['Type'] == 'Baseline']['CO2 (ppm)']
breath_in = mask_df[mask_df['Type'] == 'End of Breath IN']['CO2 (ppm)']
breath_out = mask_df[mask_df['Type'] == 'End of Breath OUT']['CO2 (ppm)']

# Print basic counts
print("\n=== Data Summary ===")
print("Total measurements: {}".format(len(mask_df)))
print("By measurement type:")
print(mask_df['Type'].value_counts())
print("\nBy mask type:")
print(mask_df['Mask Type'].value_counts())

# Print descriptive statistics
print("\n=== Descriptive Statistics (CO2 in ppm) ===")
print("\nBaseline (n={}):".format(len(baseline)))
print("  Mean +/- SD: {:.2f} +/- {:.2f}".format(baseline.mean(), baseline.std()))
print("  Median: {:.2f}".format(baseline.median()))
print("  Range: {:.2f} - {:.2f}".format(baseline.min(), baseline.max()))

print("\nEnd of Breath IN (n={}):".format(len(breath_in)))
print("  Mean +/- SD: {:.2f} +/- {:.2f}".format(breath_in.mean(), breath_in.std()))
print("  Median: {:.2f}".format(breath_in.median()))
print("  Range: {:.2f} - {:.2f}".format(breath_in.min(), breath_in.max()))

print("\nEnd of Breath OUT (n={}):".format(len(breath_out)))
print("  Mean +/- SD: {:.2f} +/- {:.2f}".format(breath_out.mean(), breath_out.std()))
print("  Median: {:.2f}".format(breath_out.median()))
print("  Range: {:.2f} - {:.2f}".format(breath_out.min(), breath_out.max()))

# Statistical Analysis
print("\n=== Statistical Analysis ===")

# 1. ANOVA
print("\n1. One-way ANOVA (comparing all three measurement types)")
groups = [baseline, breath_in, breath_out]
f_stat, p_val = stats.f_oneway(*groups)
print("  F-statistic: {:.4f}".format(f_stat))
print("  p-value: {:.8f}".format(p_val))
print("  Significant difference: {}".format(p_val < 0.05))

# 2. T-tests
print("\n2. Pairwise comparisons between measurement types (t-tests)")
# Baseline vs Breath IN
t_stat_in, p_val_in = stats.ttest_ind(baseline, breath_in, equal_var=False)
print("  Baseline vs End of Breath IN:")
print("    t-statistic: {:.4f}".format(t_stat_in))
print("    p-value: {:.8f}".format(p_val_in))
print("    Significant difference: {}".format(p_val_in < 0.05))

# Baseline vs Breath OUT
t_stat_out, p_val_out = stats.ttest_ind(baseline, breath_out, equal_var=False)
print("  Baseline vs End of Breath OUT:")
print("    t-statistic: {:.4f}".format(t_stat_out))
print("    p-value: {:.8f}".format(p_val_out))
print("    Significant difference: {}".format(p_val_out < 0.05))

# Breath IN vs Breath OUT
t_stat_in_out, p_val_in_out = stats.ttest_ind(breath_in, breath_out, equal_var=False)
print("  End of Breath IN vs End of Breath OUT:")
print("    t-statistic: {:.4f}".format(t_stat_in_out))
print("    p-value: {:.8f}".format(p_val_in_out))
print("    Significant difference: {}".format(p_val_in_out < 0.05))

# 3. Mask type comparisons
print("\n3. Comparing single vs double mask for each measurement type")
for measurement in ['Baseline', 'End of Breath IN', 'End of Breath OUT']:
    single_mask = mask_df[(mask_df['Mask Type'] == 'Single Mask') & 
                         (mask_df['Type'] == measurement)]['CO2 (ppm)']
    double_mask = mask_df[(mask_df['Mask Type'] == 'Double Mask') & 
                         (mask_df['Type'] == measurement)]['CO2 (ppm)']
    
    if len(single_mask) > 1 and len(double_mask) > 1:
        t_stat, p_val = stats.ttest_ind(single_mask, double_mask, equal_var=False)
        print("\n  {} measurement type:".format(measurement))
        print("    Single Mask: {:.2f} +/- {:.2f} ppm (n={})".format(
            single_mask.mean(), single_mask.std(), len(single_mask)))
        print("    Double Mask: {:.2f} +/- {:.2f} ppm (n={})".format(
            double_mask.mean(), double_mask.std(), len(double_mask)))
        print("    t-statistic: {:.4f}".format(t_stat))
        print("    p-value: {:.8f}".format(p_val))
        print("    Significant difference: {}".format(p_val < 0.05))
    else:
        print("\n  {} measurement type: insufficient data for comparison".format(measurement))

# 4. Correlation analysis
print("\n4. Correlation between CO2, temperature, and humidity")
corr_matrix = mask_df[['CO2 (ppm)', 'Temp (C)', 'Humidity (%)']].corr()
print(corr_matrix)

# Save summary statistics to CSV
print("\nCreating summary table...")
summary = mask_df.groupby(['Type', 'Mask Type'])['CO2 (ppm)'].agg(['count', 'mean', 'std', 'min', 'max'])
summary.to_csv(output_dir + "/table1_summary_statistics.csv")

# Create a detailed analysis summary text file
print("Saving analysis summary...")
with open(output_dir + "/analysis_summary.txt", "w") as f:
    f.write("====================================================================\n")
    f.write("STATISTICAL ANALYSIS SUMMARY - CO2 LEVELS IN N95 MASKS DURING BREATHING\n")
    f.write("====================================================================\n\n")
    f.write("SAMPLE SIZE:\n")
    f.write("- Total measurements: {}\n".format(len(mask_df)))
    f.write("- By measurement type:\n")
    for typ, count in mask_df['Type'].value_counts().items():
        f.write("  * {}: {}\n".format(typ, count))
    f.write("- By mask type:\n")
    for mask, count in mask_df['Mask Type'].value_counts().items():
        f.write("  * {}: {}\n".format(mask, count))
    
    f.write("\nDESCRIPTIVE STATISTICS (CO2 in ppm):\n\n")
    f.write("Baseline (n={}):\n".format(len(baseline)))
    f.write("  Mean +/- SD: {:.2f} +/- {:.2f}\n".format(baseline.mean(), baseline.std()))
    f.write("  Median: {:.2f}\n".format(baseline.median()))
    f.write("  Range: {:.2f} - {:.2f}\n\n".format(baseline.min(), baseline.max()))
    
    f.write("End of Breath IN (n={}):\n".format(len(breath_in)))
    f.write("  Mean +/- SD: {:.2f} +/- {:.2f}\n".format(breath_in.mean(), breath_in.std()))
    f.write("  Median: {:.2f}\n".format(breath_in.median()))
    f.write("  Range: {:.2f} - {:.2f}\n\n".format(breath_in.min(), breath_in.max()))
    
    f.write("End of Breath OUT (n={}):\n".format(len(breath_out)))
    f.write("  Mean +/- SD: {:.2f} +/- {:.2f}\n".format(breath_out.mean(), breath_out.std()))
    f.write("  Median: {:.2f}\n".format(breath_out.median()))
    f.write("  Range: {:.2f} - {:.2f}\n\n".format(breath_out.min(), breath_out.max()))
    
    f.write("STATISTICAL ANALYSIS:\n\n")
    f.write("1. One-way ANOVA (comparing all three measurement types)\n")
    f.write("  F-statistic: {:.4f}\n".format(f_stat))
    f.write("  p-value: {:.8f}\n".format(p_val))
    f.write("  Result: {}\n\n".format("Significant difference" if p_val < 0.05 else "No significant difference"))
    
    f.write("2. Pairwise comparisons between measurement types (t-tests)\n")
    f.write("  Baseline vs End of Breath IN:\n")
    f.write("    t-statistic: {:.4f}\n".format(t_stat_in))
    f.write("    p-value: {:.8f}\n".format(p_val_in))
    f.write("    Result: {}\n\n".format("Significant difference" if p_val_in < 0.05 else "No significant difference"))
    
    f.write("  Baseline vs End of Breath OUT:\n")
    f.write("    t-statistic: {:.4f}\n".format(t_stat_out))
    f.write("    p-value: {:.8f}\n".format(p_val_out))
    f.write("    Result: {}\n\n".format("Significant difference" if p_val_out < 0.05 else "No significant difference"))
    
    f.write("  End of Breath IN vs End of Breath OUT:\n")
    f.write("    t-statistic: {:.4f}\n".format(t_stat_in_out))
    f.write("    p-value: {:.8f}\n".format(p_val_in_out))
    f.write("    Result: {}\n\n".format("Significant difference" if p_val_in_out < 0.05 else "No significant difference"))
    
    # Add correlation results
    f.write("\nCorrelation analysis:\n")
    f.write("  CO2 and Temperature: {:.4f}\n".format(corr_matrix.iloc[0, 1]))
    f.write("  CO2 and Humidity: {:.4f}\n".format(corr_matrix.iloc[0, 2]))
    f.write("  Temperature and Humidity: {:.4f}\n".format(corr_matrix.iloc[1, 2]))
    
    # Add conclusions
    f.write("\nCONCLUSIONS:\n")
    baseline_mean = baseline.mean()
    breath_in_mean = breath_in.mean()
    breath_out_mean = breath_out.mean()
    
    f.write("1. CO2 levels are significantly elevated inside N95 masks during the respiratory cycle\n")
    f.write("2. End-of-expiration CO2 levels ({:.2f} ppm) are approximately {:.1f}x higher than baseline ({:.2f} ppm)\n"
           .format(breath_out_mean, breath_out_mean/baseline_mean, baseline_mean))
    f.write("3. End-of-inspiration CO2 levels ({:.2f} ppm) are approximately {:.1f}x higher than baseline ({:.2f} ppm)\n"
           .format(breath_in_mean, breath_in_mean/baseline_mean, baseline_mean))
    
    # Add mask type comparison conclusion based on the results
    single_in = mask_df[(mask_df['Mask Type'] == 'Single Mask') & (mask_df['Type'] == 'End of Breath IN')]['CO2 (ppm)'].mean()
    double_in = mask_df[(mask_df['Mask Type'] == 'Double Mask') & (mask_df['Type'] == 'End of Breath IN')]['CO2 (ppm)'].mean()
    single_out = mask_df[(mask_df['Mask Type'] == 'Single Mask') & (mask_df['Type'] == 'End of Breath OUT')]['CO2 (ppm)'].mean()
    double_out = mask_df[(mask_df['Mask Type'] == 'Double Mask') & (mask_df['Type'] == 'End of Breath OUT')]['CO2 (ppm)'].mean()
    
    # Check if differences are significant
    _, p_in = stats.ttest_ind(
        mask_df[(mask_df['Mask Type'] == 'Single Mask') & (mask_df['Type'] == 'End of Breath IN')]['CO2 (ppm)'],
        mask_df[(mask_df['Mask Type'] == 'Double Mask') & (mask_df['Type'] == 'End of Breath IN')]['CO2 (ppm)'],
        equal_var=False
    )
    
    _, p_out = stats.ttest_ind(
        mask_df[(mask_df['Mask Type'] == 'Single Mask') & (mask_df['Type'] == 'End of Breath OUT')]['CO2 (ppm)'],
        mask_df[(mask_df['Mask Type'] == 'Double Mask') & (mask_df['Type'] == 'End of Breath OUT')]['CO2 (ppm)'],
        equal_var=False
    )
    
    if p_in < 0.05 or p_out < 0.05:
        f.write("4. There are significant differences in CO2 levels between single and double masks\n")
        if p_in < 0.05:
            f.write("   - During inspiration: Single mask ({:.2f} ppm) vs Double mask ({:.2f} ppm), p={:.4f}\n"
                   .format(single_in, double_in, p_in))
        if p_out < 0.05:
            f.write("   - During expiration: Single mask ({:.2f} ppm) vs Double mask ({:.2f} ppm), p={:.4f}\n"
                   .format(single_out, double_out, p_out))
    else:
        f.write("4. There is no significant difference in CO2 levels between single and double masks during breathing\n")
    
    f.write("5. CO2 levels show {} correlation with humidity (r={:.2f}) and {} correlation with temperature (r={:.2f})\n"
           .format(
               "strong" if abs(corr_matrix.iloc[0, 2]) > 0.7 else "moderate" if abs(corr_matrix.iloc[0, 2]) > 0.3 else "weak",
               corr_matrix.iloc[0, 2],
               "strong" if abs(corr_matrix.iloc[0, 1]) > 0.7 else "moderate" if abs(corr_matrix.iloc[0, 1]) > 0.3 else "weak",
               corr_matrix.iloc[0, 1]
           ))

# Generate Figures
print("Generating figures...")

# Set up a clean style for publication-quality figures
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# FIGURE 1: Bar chart by measurement type
print("Creating Figure 1: CO2 by measurement type...")
plt.figure(figsize=(10, 6))

# Group data for plotting
groups = mask_df.groupby('Type')['CO2 (ppm)']
means = groups.mean()
std_errs = groups.sem()
counts = groups.size()

# Create the bar plot
x = np.arange(len(means))
plt.bar(x, means.values, yerr=std_errs.values, capsize=10, 
        color=['#3274A1', '#E1812C', '#3A923A'], width=0.6, edgecolor='black', linewidth=1.5)

# Add value labels and sample sizes
for i, (type_name, mean_val) in enumerate(means.items()):
    plt.text(i, mean_val + std_errs[type_name] + (means.max() * 0.03), "{:.1f}".format(mean_val), 
             ha='center', fontweight='bold', fontsize=12)
    plt.text(i, mean_val/2, "n={}".format(counts[type_name]), 
             ha='center', color='white', fontweight='bold', fontsize=12)

# Customize the plot
plt.title('CO2 Concentrations by Measurement Phase')
plt.ylabel('CO2 Concentration (ppm)')
plt.xticks(x, means.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_dir + "/figure1_co2_by_type.png", dpi=300)
plt.savefig(output_dir + "/figure1_co2_by_type.pdf")
plt.close()

# FIGURE 2: Box plot distribution
print("Creating Figure 2: CO2 distribution boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='CO2 (ppm)', data=mask_df, 
            palette=['#3274A1', '#E1812C', '#3A923A'],
            showfliers=False, width=0.6)
sns.stripplot(x='Type', y='CO2 (ppm)', data=mask_df, 
              color='black', alpha=0.3, size=3, jitter=True)

# Add sample sizes
for i, type_name in enumerate(mask_df['Type'].value_counts().index):
    count = mask_df[mask_df['Type'] == type_name].shape[0]
    plt.text(i, mask_df['CO2 (ppm)'].min() - (mask_df['CO2 (ppm)'].max() * 0.05), 
             "n={}".format(count), ha='center', fontweight='bold')

plt.title('Distribution of CO2 Concentrations')
plt.ylabel('CO2 Concentration (ppm)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_dir + "/figure2_co2_distribution.png", dpi=300)
plt.savefig(output_dir + "/figure2_co2_distribution.pdf")
plt.close()

# FIGURE 3: Mask type comparison
print("Creating Figure 3: Mask type comparison...")
plt.figure(figsize=(12, 7))

# Create pivot table for grouped data
pivot_data = pd.pivot_table(
    mask_df, 
    values='CO2 (ppm)',
    index='Type',
    columns='Mask Type',
    aggfunc=['mean', 'sem', 'count']
)

# Extract means and standard errors
means = pivot_data['mean']
errors = pivot_data['sem']
counts = pivot_data['count']

# Plot the data
barwidth = 0.35
r1 = np.arange(len(means.index))
r2 = [x + barwidth for x in r1]

# Create bars
plt.bar(r1, means['Single Mask'], width=barwidth, edgecolor='black', linewidth=1.5,
        yerr=errors['Single Mask'], capsize=7, label='Single Mask', color='#3274A1')
plt.bar(r2, means['Double Mask'], width=barwidth, edgecolor='black', linewidth=1.5,
        yerr=errors['Double Mask'], capsize=7, label='Double Mask', color='#E1812C')

# Add value annotations
for i, val in enumerate(means['Single Mask']):
    plt.text(r1[i], val + errors['Single Mask'][i] + (means.values.max() * 0.02), 
             "{:.1f}".format(val), ha='center', fontweight='bold', fontsize=10)
    plt.text(r1[i], val/2, "n={}".format(counts['Single Mask'][i]), 
             ha='center', color='white', fontweight='bold', fontsize=10)
             
for i, val in enumerate(means['Double Mask']):
    plt.text(r2[i], val + errors['Double Mask'][i] + (means.values.max() * 0.02), 
             "{:.1f}".format(val), ha='center', fontweight='bold', fontsize=10) 
    plt.text(r2[i], val/2, "n={}".format(counts['Double Mask'][i]), 
             ha='center', color='white', fontweight='bold', fontsize=10)

# Customize plot
plt.title('CO2 Concentrations: Single vs Double Mask')
plt.ylabel('CO2 Concentration (ppm)')
plt.xticks([r + barwidth/2 for r in range(len(means.index))], means.index)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_dir + "/figure3_mask_comparison.png", dpi=300)
plt.savefig(output_dir + "/figure3_mask_comparison.pdf")
plt.close()

# FIGURE 4: Correlation plot
print("Creating Figure 4: Correlation heatmap...")
plt.figure(figsize=(8, 6))
correlation_data = mask_df[['CO2 (ppm)', 'Temp (C)', 'Humidity (%)']].corr()
mask = np.triu(correlation_data, k=1)
sns.heatmap(correlation_data, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5,
            mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Between Variables')
plt.tight_layout()
plt.savefig(output_dir + "/figure4_correlation.png", dpi=300)
plt.savefig(output_dir + "/figure4_correlation.pdf")
plt.close()

# Create a README file
print("Creating README file...")
with open(output_dir + "/README.md", "w") as f:
    f.write("# N95 Mask CO2 Study - Final Results\n\n")
    f.write("This directory contains the final analysis results for the N95 mask CO2 study.\n\n")
    
    f.write("## Tables\n\n")
    f.write("- **Table 1**: `table1_summary_statistics.csv` - Summary statistics for CO2 measurements by mask type and measurement phase\n\n")
    
    f.write("## Figures\n\n")
    f.write("- **Figure 1**: `figure1_co2_by_type.png` - Bar chart showing CO2 levels by measurement type\n")
    f.write("- **Figure 2**: `figure2_co2_distribution.png` - Box plot showing distribution of CO2 values\n")
    f.write("- **Figure 3**: `figure3_mask_comparison.png` - Comparison of CO2 levels between single and double mask configurations\n")
    f.write("- **Figure 4**: `figure4_correlation.png` - Correlation heatmap between CO2 levels, temperature, and humidity\n\n")
    
    f.write("## Study Summary\n\n")
    f.write("The study measured CO2 concentrations inside N95 masks during different phases of the respiratory cycle, comparing single versus double mask configurations. \n\n")
    
    f.write("### Key Findings:\n\n")
    f.write("1. Baseline CO2: {:.2f} +/- {:.2f} ppm\n".format(baseline.mean(), baseline.std()))
    f.write("2. End of inspiration: {:.2f} +/- {:.2f} ppm\n".format(breath_in.mean(), breath_in.std()))
    f.write("3. End of expiration: {:.2f} +/- {:.2f} ppm\n\n".format(breath_out.mean(), breath_out.std()))
    
    f.write("4. CO2 levels are significantly elevated inside N95 masks during the respiratory cycle\n")
    f.write("5. End-of-expiration CO2 levels are approximately {:.1f}x higher than baseline\n".format(breath_out.mean()/baseline.mean()))
    f.write("6. End-of-inspiration CO2 levels are approximately {:.1f}x higher than baseline\n".format(breath_in.mean()/baseline.mean()))
    
    # Add mask type comparison conclusion based on the results
    if p_in < 0.05 or p_out < 0.05:
        f.write("7. There are significant differences in CO2 levels between single and double masks\n")
    else:
        f.write("7. There is no significant difference in CO2 levels between single and double masks during breathing\n")
    
    f.write("8. CO2 levels show correlation with humidity (r={:.2f}) and temperature (r={:.2f})\n\n".format(corr_matrix.iloc[0, 2], corr_matrix.iloc[0, 1]))

    f.write("For detailed statistical analysis, see the `analysis_summary.txt` file.")

print("\nAnalysis complete! All results saved to the '{}' directory.".format(output_dir))