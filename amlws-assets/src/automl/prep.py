import argparse
from pathlib import Path
import scipy.io
import pandas as pd
import numpy as np
import mlflow
from scipy.signal import butter, filtfilt, resample

def parse_args():
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--input_data", type=str, help="Path to input .mat file")
    parser.add_argument("--processed_data", type=str, help="Path to processed data output")
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--cutoff_frequency", type=int, default=60)
    args = parser.parse_args()
    return args

def process_eeg_data(eeg_data_name, mat_data, desired_length=2560, sampling_rate=256, cutoff_frequency=60):
    eeg_data = mat_data[eeg_data_name]
    channel_names = ['af7', 'af8', 'tp9', 'tp10']
    processed_data_frames = []
    
    for j in range(eeg_data.shape[0]):
        participant = eeg_data[j, 1][0][:4]
        
        for k in range(eeg_data[j, 0].shape[1]):
            data_dict = {'Participant': participant}
            
            for i, channel_name in enumerate(channel_names):
                channel_data = eeg_data[j, 0][0, k][:, i]
                
                # Resample
                resampled_data = resample(channel_data, desired_length)
                
                # Apply low-pass filter
                nyquist = 0.5 * sampling_rate
                low_cutoff = cutoff_frequency / nyquist
                b, a = butter(4, low_cutoff, btype='low')
                filtered_data = filtfilt(b, a, resampled_data)
                
                data_dict[channel_name] = filtered_data
                
            processed_data_frames.append(pd.DataFrame([data_dict]))
    
    return pd.concat(processed_data_frames, ignore_index=True)

def main():
    mlflow.start_run()
    args = parse_args()
    
    print(f"Loading data from: {args.input_data}")
    mat_data = scipy.io.loadmat(args.input_data)
    
    df_non_remission = process_eeg_data(
        'EEG_windows_Non_remission', 
        mat_data, 
        sampling_rate=args.sampling_rate,
        cutoff_frequency=args.cutoff_frequency
    )
    
    df_remission = process_eeg_data(
        'EEG_windows_Remission', 
        mat_data,
        sampling_rate=args.sampling_rate,
        cutoff_frequency=args.cutoff_frequency
    )
    
    output_path = Path(args.processed_data)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df_non_remission.to_parquet(output_path / "non_remission.parquet")
    df_remission.to_parquet(output_path / "remission.parquet")
    
    mlflow.log_metric("non_remission_samples", len(df_non_remission))
    mlflow.log_metric("remission_samples", len(df_remission))
    
    print(f"Processed data saved to: {args.processed_data}")
    mlflow.end_run()

if __name__ == "__main__":
    main()