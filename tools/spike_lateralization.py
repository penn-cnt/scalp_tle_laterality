import os
import numpy as np
import pickle
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, resample
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import pytorch_lightning as pl
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Debugging: Print sys.path to verify paths
print("sys.path:", sys.path)
from sleeplib.Resnet_15.model import FineTuning
from sleeplib.config import Config
from sleeplib.transforms import extremes_remover
from sleeplib.montages import con_combine_montage
from ieeg.auth import Session

# Create output directory for pickle files
output_directory = '.../pickle_outputs'
os.makedirs(output_directory, exist_ok=True)

# Load the input CSV file
csv_file_path = '.../ieeg_clips.csv'
dataset_details = pd.read_csv(csv_file_path)

def get_iEEG_data(
    username,
    password_bin_file,
    iEEG_filename,
    start_time_usec,
    stop_time_usec,
    select_electrodes=None,
):

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    with open(password_bin_file, "r") as f:
        s = Session(username, f.read())
    ds = s.open_dataset(iEEG_filename)
    all_channel_labels = ds.get_channel_labels()
    interesting_labels = [ch for ch in all_channel_labels if ch in ['T7', 'T8', 'P7', 'P8']] # check if our datasets followed by MCN system
    if interesting_labels:
        print(f"Interesting channels found in {iEEG_filename}: {interesting_labels}")

    # Map selected electrode names to their corresponding indices
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], str):
            # Find indices of the selected electrode names in all_channel_labels
            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
            # print(channel_ids)
        else:
            print("Electrodes must be given as a list of strings")
            return None, None  # Return empty if the input format is incorrect

    try:
        data = ds.get_data(start_time_usec, duration, channel_ids)
    except:
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = ds.get_data(clip_start, clip_size, channel_ids)
            else:
                data = np.concatenate(
                    ([data, ds.get_data(clip_start, clip_size, channel_ids)]), axis=0
                )
            clip_start = clip_start + clip_size
        data = np.concatenate(
            ([data, ds.get_data(clip_start, stop_time_usec - clip_start, channel_ids)]),
            axis=0,
        )

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate
    return df, fs


def high_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def downsample_with_filter(data, original_fs, target_fs, cutoff=None, order=5):
    num_samples, num_channels = data.shape
    downsample_factor = original_fs // target_fs
    if downsample_factor < 2:
        raise ValueError("Downsampling factor must be at least 2.")

    # Determine cutoff frequency for low-pass filter
    if cutoff is None:
        cutoff = target_fs / 2  # Nyquist frequency for the target sample rate

    # Apply low-pass filter
    sos = butter(order, cutoff / (0.5 * original_fs), btype='low', output='sos')
    filtered_data = sosfilt(sos, data, axis=0)

    # Downsample the filtered data
    downsampled_data = resample(filtered_data, num_samples // downsample_factor, axis=0)
    return downsampled_data

def common_average_montage(ieeg_data):
    avg_signal = ieeg_data.mean(axis=1)
    result = ieeg_data - avg_signal[:, np.newaxis]
    if result.shape != ieeg_data.shape:
        raise ValueError("The shape of the resulting data doesn't match the input data.")
    return result


def notch_filter(data, hz, fs):
    b, a = iirnotch(hz, Q=30, fs=fs)
    y = filtfilt(b, a, data, axis=0)
    
    return y

class ContinousToSnippetDataset(Dataset):
    def __init__(self, signal_data, montage=None, transform=None, Fq=128, window_size=1, step=8):
        # Process the signal data
        signal = signal_data
        signal = 1.0 * np.where(np.isnan(signal), 0, signal)
        
        # Move signal to torch
        signal = torch.FloatTensor(signal.astype(np.float32))
        
        # Generate snippets of shape (n_snippets, n_channels, ts)
        self.snippets = signal.unfold(dimension=1, size=window_size * Fq, step=step).permute(1, 0, 2)
        
        # Set transform and montage
        self.transform = transform
        self.montage = montage

    def __len__(self):
        return self.snippets.shape[0]

    def _preprocess(self, signal):
        '''Preprocess signal and apply montage, transform and normalization'''

        if self.montage is not None:
            signal = self.montage(signal)

        # Apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        # Normalize signal
        # signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        signal = signal / (np.quantile(np.abs(signal), q=0.95, axis=-1, keepdims=True) + 1e-8)


        signal = torch.FloatTensor(signal.copy())

        return signal

    def __getitem__(self, idx):
        # Get the snippet
        signal = self.snippets[idx, :, :]
        # Preprocess signal
        signal = self._preprocess(signal)
        
        # Return signal and dummy label, the latter to prevent lightning dataloader from complaining
        return signal, 0

# Define the 18 channels to include
channels_to_include = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                       'Fz', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
current_channel_order = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                         'Fz', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Pz']
new_channel_order = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz',
                     'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

# Define channel groups for zeroing conditions
mask1_channels = ['Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'Fz', 'Cz', 'Pz']  # keep Left hemisphere
mask2_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz']  # keep Right hemisphere

# Configuration
step_size = 8

# Load configuration
config = Config()

# Set up transformations and montage
transform_test = transforms.Compose([extremes_remover(signal_max=2000, signal_min=20)])
con_combine_montage = con_combine_montage()

# Load pretrained model
data_directory = '/mnt/sauce/littlab/users/jurikim/'
model = FineTuning.load_from_checkpoint(
    os.path.join(data_directory, 'ied_yesno/1s-round11-hardmine-chan_weights-v1.ckpt'),
    lr=config.LR,
    head_dropout=config.HEAD_DROPOUT,
    n_channels=config.N_CHANNELS,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    map_location=torch.device('cuda')
)

trainer=pl.Trainer(fast_dev_run=False,enable_progress_bar=False,devices=1,strategy="auto")

# Iterate over all rows in the dataset
for index, row in dataset_details.iterrows():
    ieeg_file_name = row['ieeg_file_name']
    start_sec = row['start_sec']
    end_sec = row['end_sec']

    # Create the folder named with EMUxxxx
    folder_ieeg = row['ieeg_file_name'].split('_')[0]
    folder_name = os.path.join(output_directory, folder_ieeg)

    # check the existence of pickle file
    pickle_file_name = f"{ieeg_file_name}_{start_sec}_{end_sec}.pkl"
    pickle_file_path = os.path.join(folder_name, pickle_file_name)

    # 💡 Add this part to check for file existence before processing
    if os.path.exists(pickle_file_path):
        print(f"File already exists, skipping: {pickle_file_path}")
        continue # Skip to the next iteration

    print(f"Processing {ieeg_file_name} from {start_sec} to {end_sec} seconds.")
    try:
        # Get the data using the modified get_iEEG_data function
        df, fs = get_iEEG_data('jurikim', 'jurikim_ieeglogin.bin', ieeg_file_name,
                               start_sec * 1e6, end_sec * 1e6, channels_to_include)
        
        # Check if our dataset followd by MCN system
        interesting_channels = [ch for ch in df.columns if ch in ['T7', 'T8', 'P7', 'P8']]
        if not interesting_channels:
            continue  # Skip if none of the target channels are present

        print(f"Processing {ieeg_file_name} from {start_sec} to {end_sec} seconds. Found: {interesting_channels}")

        segment_data = df.values

        if np.isnan(segment_data).any():
            print(f"Skipping {ieeg_file_name} due to NaN values in segment_data.")
            continue

        fs = int(fs)
        notch_data = notch_filter(segment_data, hz=60, fs=fs)
        filtered_data = high_pass_filter(notch_data, cutoff=0.5, fs=fs)
        downsampled_data = downsample_with_filter(filtered_data, fs, target_fs=128)

        # Reorder channels
        indices = [2, 12, 13, 10, 11]
        pz_mean = np.mean(downsampled_data[:, indices], axis=1)
        downsampled_data_with_pz = np.column_stack((downsampled_data, pz_mean))
        reorder_index = [current_channel_order.index(ch) for ch in new_channel_order]
        reordered_data = downsampled_data_with_pz[:, reorder_index] #['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        
        # Compute SN2
        data = ContinousToSnippetDataset(
            signal_data=reordered_data.T,
            montage=con_combine_montage,
            transform=transform_test,
            window_size=1,
            step=8
        )
        con_dataloader = DataLoader(data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
        score = trainer.predict(model, con_dataloader)
        SN2 = np.concatenate(score).astype(float).flatten()
        # score = np.concatenate(score).astype(float).flatten()
        # zeros_front = np.zeros(7)
        # zeros_back = np.zeros(8)
        # SN2 = np.concatenate([zeros_front, score, zeros_back])

        # Compute SN2_zero_right and SN2_zero_left
        zeroing_masks = [mask1_channels, mask2_channels]
        hemispheres = ["zero-right", "zero-left"]
        SN2_zero_right, SN2_zero_left = None, None

        for iteration, mask_channels in enumerate(zeroing_masks):
            temp_data = np.copy(reordered_data)
            for ch_name in mask_channels:
                ch_index = new_channel_order.index(ch_name)
                temp_data[:, ch_index] = 0

            data = ContinousToSnippetDataset(
                signal_data=temp_data.T,
                montage=con_combine_montage,
                transform=transform_test,
                window_size=1,
                step=8
            )
            con_dataloader = DataLoader(data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
            score = trainer.predict(model, con_dataloader)
            score = np.concatenate(score).astype(float).flatten()
            # score = np.concatenate([zeros_front, score, zeros_back])

            if hemispheres[iteration] == "zero-right":
                SN2_zero_right = score
            elif hemispheres[iteration] == "zero-left":
                SN2_zero_left = score

        # Create the folder named with EMUxxxx
        folder_ieeg = row['ieeg_file_name'].split('_')[0]  # Remove _DayXX_X
        folder_name = os.path.join(output_directory, folder_ieeg)
        os.makedirs(folder_name, exist_ok=True)

        # Save the pickle file
        pickle_file_name = f"{ieeg_file_name}_{start_sec}_{end_sec}.pkl"
        pickle_file_path = os.path.join(folder_name, pickle_file_name)

        pickle_data = {
            "SN2": SN2,
            "SN2_zero_right": SN2_zero_right,
            "SN2_zero_left": SN2_zero_left
        }

        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(pickle_data, pickle_file)

        print(f"Saved pickle file to: {pickle_file_path}")

    except Exception as e:
        print(f"Error processing {ieeg_file_name}: {e}")

