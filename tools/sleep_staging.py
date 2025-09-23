import os
import re
import pickle
import numpy as np
import pandas as pd
import mne
import yasa
from collections import Counter
from ieeg.auth import Session

# Constants
channels_to_include = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
samples_per_epoch = int(30 / 0.0625)  # 480
pkl_root_folders = [
    '.../pickle_outputs/'
]

# Collect all .pkl paths recursively from all folders
all_pkl_paths = []
for root_folder in pkl_root_folders:
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pkl'):
                all_pkl_paths.append(os.path.join(root, file))

username = 'jurikim'
password_bin = '/mnt/sauce/littlab/users/jurikim/spikenet/jurikim_ieeglogin.bin'

# iEEG loader
def get_iEEG_data(username, password_bin_file, iEEG_filename, start_time_usec, stop_time_usec, select_electrodes):
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)

    all_channel_labels = ds.get_channel_labels()
    channel_ids = [i for i, e in enumerate(all_channel_labels) if e in select_electrodes]

    try:
        data = ds.get_data(start_time_usec, duration, channel_ids)
    except:
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            chunk = ds.get_data(clip_start, clip_size, channel_ids)
            data = chunk if data is None else np.concatenate((data, chunk), axis=0)
            clip_start += clip_size
        data = np.concatenate((data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)), axis=0)

    df = pd.DataFrame(data, columns=select_electrodes)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate
    return df, fs

# Sleep staging
def process_file_YASA(ieeg_file_name, start_sec, end_sec):
    try:
        df, fs = get_iEEG_data(
            username,
            password_bin,
            ieeg_file_name,
            start_sec * 1e6,
            end_sec * 1e6,
            channels_to_include
        )
    except Exception as e:
        print(f"[DATA ERROR] {ieeg_file_name}: {e}")
        return None, None

    if df.empty:
        return None, None

    info = mne.create_info(ch_names=channels_to_include, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(df.values.T, info)

    raw.resample(100, npad="auto")
    raw.filter(l_freq=0.4, h_freq=30, fir_design='firwin')
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()

    return raw, fs

# Majority voting logic
def determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4):
    consensus_stage = []
    for i in range(len(predicted_c3)):
        stages = [predicted_c3[i], predicted_cz[i], predicted_c4[i]]
        stage_counts = Counter(stages)
        if stage_counts.most_common(1)[0][1] >= 2:
            consensus_stage.append(stage_counts.most_common(1)[0][0])
        else:
            consensus_stage.append(np.nan)
    return consensus_stage

# Loop through .pkl files
for pkl_path in all_pkl_paths:
    filename = os.path.basename(pkl_path)

    match = re.search(r'(EMU\d+_Day\d+_\d+)_(\d+\.\d+)_([\d\.]+)\.pkl$', filename)
    if not match:
        print(f"[SKIP] Invalid filename format: {filename}")
        continue

    ieeg_file_name = match.group(1)
    start_sec = float(match.group(2))
    end_sec = float(match.group(3))

    # Create Time array
    time_array = np.arange(0, end_sec - start_sec + 0.0625, 0.0625)
    time_len = len(time_array)

    # Load data
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        continue

    # Pad SN2-related arrays if needed
    for key in ['SN2', 'SN2_zero_right', 'SN2_zero_left']:
        if key in data:
            signal_len = len(data[key])
            diff = time_len - signal_len
            if diff > 0:
                left_pad = diff // 2
                right_pad = diff - left_pad
                data[key] = np.pad(data[key], (left_pad, right_pad), mode='constant')
                print(f"[{filename}] Padded {key}: {left_pad} start, {right_pad} end")
            else:
                print(f"[{filename}] No padding needed for {key}")
        else:
            print(f"[{filename}] {key} not found")

    # Run YASA sleep staging
    raw, fs = process_file_YASA(ieeg_file_name, start_sec, end_sec)
    if raw is None:
        print(f"[SKIP] Could not process EEG for {filename}")
        continue

    try:
        sls_c3 = yasa.SleepStaging(raw, eeg_name="C3").predict()
        sls_cz = yasa.SleepStaging(raw, eeg_name="Cz").predict()
        sls_c4 = yasa.SleepStaging(raw, eeg_name="C4").predict()
        consensus = determine_consensus_stage(sls_c3, sls_cz, sls_c4)

        # Expand consensus into per-sample array
        expanded = []
        for stage in consensus:
            expanded.extend([stage] * samples_per_epoch)
        yasa_array = np.array(expanded)

        # Update the .pkl with YASA and Time
        data['YASA'] = yasa_array
        data['Time'] = time_array

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"[OK] Updated {filename} with YASA and Time")

    except Exception as e:
        print(f"[ERROR] YASA failed for {filename}: {e}")
