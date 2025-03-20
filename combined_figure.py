"""
Create a combined figure showing CO2, temperature, and humidity by measurement type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output directory
output_dir = "final_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# Convert columns to numeric
df['CO2 (ppm)'] = pd.to_numeric(df['CO2 (ppm)'], errors='coerce')
df['Temp (C)'] = pd.to_numeric(df['Temp (C)'], errors='coerce')
df['Humidity (%)'] = pd.to_numeric(df['Humidity (%)'], errors='coerce')

# Filter to include only mask data
mask_df = df[df['Source File'].isin(['1_mask_12_1_24.txt', '2_masks_12_1_24.txt'])]

# Create a new column for mask type
mask_df.loc[:, 'Mask Type'] = mask_df['Source File'].replace({
    '1_mask_12_1_24.txt': 'Single Mask',
    '2_masks_12_1_24.txt': 'Double Mask'
})

# Calculate means and standard errors for each measurement type
grouped = mask_df.groupby('Type')
means = grouped.mean()
std_errs = grouped.sem()
counts = grouped.size()

print("Creating combined figure...")

# Set up a clean style for publication-quality figure
plt.style.use('classic')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# Define colors for each measurement type
colors = {'Baseline': '#396AB1', 'End of Breath IN': '#DA7C30', 'End of Breath OUT': '#3E9651'}

# 1. CO2 SUBPLOT
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(means.index))
bars = ax1.bar(x, means['CO2 (ppm)'], yerr=std_errs['CO2 (ppm)'], capsize=10, 
        color=[colors[t] for t in means.index], width=0.6, edgecolor='black', linewidth=1.5)

# Add value labels and sample sizes
for i, (type_name, mean_val) in enumerate(means['CO2 (ppm)'].items()):
    ax1.text(i, mean_val + std_errs['CO2 (ppm)'][type_name] + (max(means['CO2 (ppm)']) * 0.03), 
             "{:.1f}".format(mean_val), ha='center', fontweight='bold', fontsize=12)
    ax1.text(i, mean_val/2, "n={}".format(counts[type_name]), 
             ha='center', color='white', fontweight='bold', fontsize=12)

ax1.set_ylabel('CO2 Concentration (ppm)', fontweight='bold')
ax1.set_title('CO2 by Measurement Phase', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(means.index)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 2. TEMPERATURE SUBPLOT
ax2 = plt.subplot(2, 2, 2)
bars = ax2.bar(x, means['Temp (C)'], yerr=std_errs['Temp (C)'], capsize=10, 
        color=[colors[t] for t in means.index], width=0.6, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (type_name, mean_val) in enumerate(means['Temp (C)'].items()):
    ax2.text(i, mean_val + std_errs['Temp (C)'][type_name] + (max(means['Temp (C)']) * 0.01), 
             "{:.1f}".format(mean_val), ha='center', fontweight='bold', fontsize=12)

ax2.set_ylabel('Temperature (C)', fontweight='bold')
ax2.set_title('Temperature by Measurement Phase', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(means.index)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 3. HUMIDITY SUBPLOT
ax3 = plt.subplot(2, 2, 3)
bars = ax3.bar(x, means['Humidity (%)'], yerr=std_errs['Humidity (%)'], capsize=10, 
        color=[colors[t] for t in means.index], width=0.6, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (type_name, mean_val) in enumerate(means['Humidity (%)'].items()):
    ax3.text(i, mean_val + std_errs['Humidity (%)'][type_name] + (max(means['Humidity (%)']) * 0.01), 
             "{:.1f}".format(mean_val), ha='center', fontweight='bold', fontsize=12)

ax3.set_ylabel('Humidity (%)', fontweight='bold')
ax3.set_title('Humidity by Measurement Phase', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(means.index)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# 4. COMBINED LINE GRAPH (normalized values)
ax4 = plt.subplot(2, 2, 4)

# Normalize each variable to its maximum value for comparison
normalized_data = means.copy()
for col in ['CO2 (ppm)', 'Temp (C)', 'Humidity (%)']:
    normalized_data[col] = normalized_data[col] / normalized_data[col].max()

# Create line graph
ax4.plot(normalized_data.index, normalized_data['CO2 (ppm)'], 'o-', color='#396AB1', 
         linewidth=2, markersize=8, label='CO2')
ax4.plot(normalized_data.index, normalized_data['Temp (C)'], 's-', color='#DA7C30',
         linewidth=2, markersize=8, label='Temperature')
ax4.plot(normalized_data.index, normalized_data['Humidity (%)'], '^-', color='#3E9651',
         linewidth=2, markersize=8, label='Humidity')

# Add actual values as text near each point
for i, idx in enumerate(normalized_data.index):
    ax4.text(i, normalized_data['CO2 (ppm)'][idx] + 0.05, 
             "{:.1f}".format(means['CO2 (ppm)'][idx]), 
             ha='center', color='#396AB1', fontweight='bold')
    
    ax4.text(i, normalized_data['Temp (C)'][idx] - 0.08, 
             "{:.1f}C".format(means['Temp (C)'][idx]), 
             ha='center', color='#DA7C30', fontweight='bold')
    
    ax4.text(i, normalized_data['Humidity (%)'][idx] + 0.05, 
             "{:.1f}%".format(means['Humidity (%)'][idx]), 
             ha='center', color='#3E9651', fontweight='bold')

ax4.set_ylabel('Normalized Values', fontweight='bold')
ax4.set_title('Comparison of Normalized Values', fontweight='bold')
ax4.set_xticks(range(len(normalized_data.index)))
ax4.set_xticklabels(normalized_data.index)
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4.legend(loc='upper left')

# Overall title
plt.suptitle('CO2, Temperature, and Humidity in N95 Masks\nDuring Different Respiratory Phases', 
             fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
plt.savefig(output_dir + "/figure5_combined_variables.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir + "/figure5_combined_variables.pdf", bbox_inches='tight')
print("Combined figure saved to " + output_dir + "/figure5_combined_variables.png")

# Create a data table showing all values
combined_table = pd.DataFrame({
    'Measurement Type': means.index,
    'CO2 (ppm)': ["{:.2f} +/- {:.2f}".format(means['CO2 (ppm)'][t], std_errs['CO2 (ppm)'][t]) for t in means.index],
    'Temperature (C)': ["{:.2f} +/- {:.2f}".format(means['Temp (C)'][t], std_errs['Temp (C)'][t]) for t in means.index],
    'Humidity (%)': ["{:.2f} +/- {:.2f}".format(means['Humidity (%)'][t], std_errs['Humidity (%)'][t]) for t in means.index],
    'Sample Size': [counts[t] for t in means.index]
})

# Save the table
combined_table.to_csv(output_dir + "/table2_combined_variables.csv", index=False)
print("Combined variables table saved to " + output_dir + "/table2_combined_variables.csv")

# Update the README.md to include the new figure and table
with open(output_dir + "/README.md", "r") as f:
    readme_content = f.read()

# Insert new figure and table information
updated_readme = readme_content.replace(
    "- **Figure 4**: `figure4_correlation.png` - Correlation heatmap between CO2 levels, temperature, and humidity\n\n",
    "- **Figure 4**: `figure4_correlation.png` - Correlation heatmap between CO2 levels, temperature, and humidity\n- **Figure 5**: `figure5_combined_variables.png` - Combined visualization of CO2, temperature, and humidity by measurement phase\n\n"
)

updated_readme = updated_readme.replace(
    "- **Table 1**: `table1_summary_statistics.csv` - Summary statistics for CO2 measurements by mask type and measurement phase\n\n",
    "- **Table 1**: `table1_summary_statistics.csv` - Summary statistics for CO2 measurements by mask type and measurement phase\n- **Table 2**: `table2_combined_variables.csv` - Combined statistics for CO2, temperature, and humidity by measurement phase\n\n"
)

with open(output_dir + "/README.md", "w") as f:
    f.write(updated_readme)

print("Updated README.md with new figure and table information")