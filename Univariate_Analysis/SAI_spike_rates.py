import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
tle_df_training = os.path.join(script_dir, "/Dataset/training_261.csv")
tle_df_training = pd.read_csv(tle_df_training)
save_path_2C = os.path.join(script_dir, "Visualization", "Fig2C.jpg")
save_path_2D = os.path.join(script_dir, "Visualization", "Fig2D.jpg")

# Filter for the specified threshold
tle_df_training = tle_df_training[
    (tle_df_training['threshold'] == 0.43)
]

print("Counts of single_lat categories after filtering:")
print(tle_df_training['single_lat'].value_counts())

# Define a consistent order and color palette for the plots
order = ['left', 'right', 'bilateral']
palette = {
    'left': '#7B9FC8',
    'right': '#D55E00',
    'bilateral': '#F0E442'
}

# --- Plot 1: Spike Asymmetry Index (SAI) ---
print("Generating and saving the SAI plot...")

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=tle_df_training,
    x='single_lat',
    y='SAI',
    order=order,
    palette=palette,
    showfliers=False
)
sns.stripplot(
    data=tle_df_training,
    x='single_lat',
    y='SAI',
    order=order,
    color='black',
    size=6,
    jitter=True,
    alpha=0.7
)
plt.ylim(-1.1, 1.1)
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.xlabel('Single Lateralization')
plt.ylabel('Spike Asymmetry Index (SAI)')
plt.grid(True)
plt.tight_layout()

# Save the figure to the specified path
plt.savefig(save_path_2C)
print("SAI plot saved")
# plt.close() # Close the plot to free up memory

# --- Plot 2: Log spike rate plot ---
print("Generating and saving the Log plot...")

# Create a new column with the log-transformed spike rate
# We add 1 to avoid issues with zero values before taking the log.
tle_df_training['log_spike_rate_min'] = np.log10(tle_df_training['spike_rate_min'] + 1)

plt.figure(figsize=(8, 6))
sns.boxplot(data=tle_df_training, x='single_lat', y='log_spike_rate_min', order=order, palette=palette, showfliers=False)
sns.stripplot(data=tle_df_training, x='single_lat', y='log_spike_rate_min', order=order, color='black', size=6, jitter=True, alpha=0.7)

plt.xlabel('Single Lateralization')
plt.ylabel('Log$_{10}$(Spike Rate + 1) [spikes/min]')
plt.grid(True)
plt.tight_layout()

# Save the figure to the specified path
plt.savefig(save_path_2D)
print("Log spike rate plot saved")
# plt.close()
