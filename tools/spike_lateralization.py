import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# --- Configuration ---
POINTS_PER_SECOND = 16
input_csv = ".../input_test_104.csv"
pickle_base_path = ".../pickle_outputs_2324"
thresholds = np.round(np.arange(0.43, 0.99, 0.01), 2)
output_csv = ".../test_104.csv"

# --- Load full metadata ---
df_meta = pd.read_csv(input_csv)

# --- Spike rate calculation ---
def calculate_spike_rate(SN2, threshold):
    SN2 = np.array(SN2)
    spike_count = 0
    i = 8
    while i + 8 < len(SN2):
        if SN2[i] > threshold:
            spike_count += 1
            i += 16
        else:
            i += 1
    total_minutes = len(SN2) / (16 * 60)
    spike_rate = spike_count / total_minutes if total_minutes > 0 else 0
    return spike_rate, spike_count, total_minutes

# --- Spike clustering logic ---
def cluster_spikes_with_max(SN2, SN2_left, SN2_right, threshold):
    final_result = []
    i = 8
    while i + 8 < len(SN2):
        if SN2[i] > threshold:
            win_start = max(0, i - 8)
            win_end = min(len(SN2), i + 8)
            max_right = np.max(SN2_right[win_start:win_end])
            max_left = np.max(SN2_left[win_start:win_end])
            final_result.append('R' if max_right > max_left else 'L')
            i += 16
        else:
            i += 1
    return final_result

# --- AI Calculation ---
def calculate_ai(clustered):
    filtered = [x for x in clustered if x is not None]
    l = filtered.count('L')
    r = filtered.count('R')
    return (l - r) / (l + r) if (l + r) > 0 else 0

# --- Main processing loop ---
results = []

for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
    admission_id = row['admission_id']
    patient_id = row['patient_id']
    age_Erin = row['age_erin']
    gender = row['gender']
    ieeg_file_name = row['ieeg_file_name']
    lat_label = row['single_lat']
    pkl_path = os.path.join(pickle_base_path, admission_id, f"{admission_id}.pkl")

    if not os.path.exists(pkl_path):
        print(f"Missing file: {pkl_path}")
        continue

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        SN2 = data['SN2']
        SN2_left = data['SN2_zero_right']
        SN2_right = data['SN2_zero_left']
    except Exception as e:
        print(f" Failed to load {admission_id}: {e}")
        continue

    for threshold in thresholds:
        cluster_result = cluster_spikes_with_max(SN2, SN2_left, SN2_right, threshold)
        ai = calculate_ai(cluster_result)
        spike_rate, spike_count, duration_min = calculate_spike_rate(SN2, threshold)

        results.append({
            'patient_id': patient_id,
            'age': age_Erin,
            'gender': gender,
            'admission_id': admission_id,
            'ieeg_file_name': ieeg_file_name,
            'single_lat': lat_label,
            'duration_recording_min': duration_min,
            'threshold': threshold,
            'SAI': ai,
            'spike_rate_min': spike_rate,
            'absolute_spike': spike_count
        })

# --- Final DataFrame and Save ---
df_all = pd.DataFrame(results)
df_all.to_csv(output_csv, index=False)
print(f"Full data saved to: {output_csv}")