"""
Create a plot showing CO2, temperature, and humidity for single mask data only,
with side-by-side bars for each measurement phase.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = "final_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# Convert columns to numeric
df['CO2 (ppm)'] = pd.to_numeric(df['CO2 (ppm)'], errors='coerce')
df['Temp (C)'] = pd.to_numeric(df['Temp (C)'], errors='coerce')
df['Humidity (%)'] = pd.to_numeric(df['Humidity (%)'], errors='coerce')

# Filter to include only single mask data
single_mask_df = df[df['Source File'] == '1_mask_12_1_24.txt']

# Calculate means and standard errors for each measurement type
grouped = single_mask_df.groupby('Type')
means = grouped.mean()
std_errs = grouped.sem()
counts = grouped.size()

print("Creating single mask figure...")

# Set up the figure
plt.figure(figsize=(12, 8))
plt.style.use('classic')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Set up the bar positions
x = np.arange(len(means.index))  # Position on the x-axis for each measurement type
width = 0.25  # Width of each bar
opacity = 0.8

# Create the bars
plt.bar(x - width, means['CO2 (ppm)'] / 100, width, alpha=opacity, color='#3274A1', 
        yerr=std_errs['CO2 (ppm)'] / 100, ecolor='black', capsize=5, 
        label='CO2 (x100 ppm)')

plt.bar(x, means['Temp (C)'], width, alpha=opacity, color='#E1812C', 
        yerr=std_errs['Temp (C)'], ecolor='black', capsize=5, 
        label='Temperature (C)')

plt.bar(x + width, means['Humidity (%)'], width, alpha=opacity, color='#3A923A', 
        yerr=std_errs['Humidity (%)'], ecolor='black', capsize=5, 
        label='Humidity (%)')

# Add labels and title
plt.xlabel('Measurement Phase', fontweight='bold')
plt.ylabel('Value', fontweight='bold')
plt.title('CO2, Temperature, and Humidity Measurements\nSingle Mask Only (n={})'.format(len(single_mask_df)), 
          fontweight='bold')
plt.xticks(x, means.index)
plt.legend(loc='upper left')

# Add value annotations
for i, (type_name, co2_val) in enumerate(means['CO2 (ppm)'].items()):
    co2_scaled = co2_val / 100  # Scale down for visualization
    plt.text(i - width, co2_scaled + (std_errs['CO2 (ppm)'][type_name] / 100) + 1, 
             "{:.1f}".format(co2_val), ha='center', fontweight='bold', fontsize=10)
    plt.text(i - width, co2_scaled / 2, "n={}".format(counts[type_name]), 
             ha='center', color='white', fontweight='bold', fontsize=10)
    
for i, (type_name, temp_val) in enumerate(means['Temp (C)'].items()):
    plt.text(i, temp_val + std_errs['Temp (C)'][type_name] + 1, 
             "{:.1f}".format(temp_val), ha='center', fontweight='bold', fontsize=10)
    
for i, (type_name, humid_val) in enumerate(means['Humidity (%)'].items()):
    plt.text(i + width, humid_val + std_errs['Humidity (%)'][type_name] + 1, 
             "{:.1f}".format(humid_val), ha='center', fontweight='bold', fontsize=10)

# Add explanatory note about CO2 scaling
plt.figtext(0.5, 0.01, "Note: CO2 values are scaled down by a factor of 100 for visualization purposes", 
            ha='center', fontsize=10, fontstyle='italic')

# Enhance grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save the figure
plt.savefig(output_dir + "/single_mask_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir + "/single_mask_comparison.pdf", bbox_inches='tight')
print("Figure saved to " + output_dir + "/single_mask_comparison.png")

# Create a summary table for single mask data
summary = pd.DataFrame({
    'Measurement Phase': means.index,
    'CO2 (ppm)': ["{:.2f} +/- {:.2f}".format(means['CO2 (ppm)'][t], std_errs['CO2 (ppm)'][t]) for t in means.index],
    'Temperature (C)': ["{:.2f} +/- {:.2f}".format(means['Temp (C)'][t], std_errs['Temp (C)'][t]) for t in means.index],
    'Humidity (%)': ["{:.2f} +/- {:.2f}".format(means['Humidity (%)'][t], std_errs['Humidity (%)'][t]) for t in means.index],
    'Sample Size': [counts[t] for t in means.index]
})

# Save the summary table
summary.to_csv(output_dir + "/single_mask_summary.csv", index=False)
print("Summary table saved to " + output_dir + "/single_mask_summary.csv")