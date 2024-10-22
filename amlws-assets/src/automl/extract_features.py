import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import nolds
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("extract_features")
    parser.add_argument("--processed_data", type=str, help="Path to processed data")
    parser.add_argument("--features_output", type=str, help="Path to features output")
    parser.add_argument("--sampling_rate", type=int, default=256)
    args = parser.parse_args()
    return args

def band_power(data, sf, band, window_length):
    f, Pxx = welch(data, sf, nperseg=window_length)
    ind_min = np.argmax(f > band[0]) - 1
    ind_max = np.argmax(f > band[1]) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def hfd(data, Kmax):
    # Initialize an empty list to store log-log values
    x = []
    # Get the length of the input data
    N = len(data)
    # Loop over each scale from 1 to Kmax
    for k in range(1, Kmax + 1):
        # Initialize an empty list to store Lmk values for the current scale
        Lk = []
        # Loop over each segment within the current scale
        for m in range(k):
            # Calculate indices for the current segment
            indices = np.arange(m, N, k)
            # Skip if the segment has fewer than 2 points
            if len(indices) < 2:
                continue
            # Calculate the sum of absolute differences for the segment
            Lmk = np.sum(np.abs(np.diff(data[indices])))
            # Normalize Lmk by the segment length and scale
            Lmk *= (N - 1) / ((len(indices) - 1) * k)
            # Append the normalized Lmk to the list for the current scale
            Lk.append(Lmk)
        # If there are valid Lmk values, calculate log-log values and append to x
        if len(Lk) > 0:
            x.append([np.log(1.0 / k), np.log(np.mean(Lk))])

def compute_features(channel_data, sf):
    features = {}
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 60)
    }
    
    for band_name, band_range in bands.items():
        features[f'bp_{band_name}'] = band_power(channel_data, sf, band_range, len(channel_data))
    
    features['hfd'] = hfd(channel_data, Kmax=10)
    features['corr_dim'] = nolds.corr_dim(channel_data, emb_dim=10)
    features['hurst'] = nolds.hurst_rs(channel_data)
    features['lyapunov'] = nolds.lyap_r(channel_data, emb_dim=10)
    features['dfa'] = nolds.dfa(channel_data)
    
    return features

def extract_features_from_df(df, sf):
    all_features = []
    channels = ['af7', 'af8', 'tp9', 'tp10']
    
    for index, row in df.iterrows():
        features = {'Participant': row['Participant']}
        
        for channel in channels:
            channel_features = compute_features(np.array(row[channel]), sf)
            for feature_name, value in channel_features.items():
                features[f'{channel}_{feature_name}'] = value
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)

def main():
    mlflow.start_run()
    args = parse_args()
    
    input_path = Path(args.processed_data)
    df_non_remission = pd.read_parquet(input_path / "non_remission.parquet")
    df_remission = pd.read_parquet(input_path / "remission.parquet")
    
    print("Extracting features for non-remission group...")
    df_non_remission_features = extract_features_from_df(df_non_remission, args.sampling_rate)
    df_non_remission_features['Remission'] = 0
    
    print("Extracting features for remission group...")
    df_remission_features = extract_features_from_df(df_remission, args.sampling_rate)
    df_remission_features['Remission'] = 1
    
    df_combined = pd.concat([df_non_remission_features, df_remission_features], ignore_index=True)
    
    output_path = Path(args.features_output)
    output_path.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(output_path / "features.parquet")
    
    mlflow.log_metric("total_features", len(df_combined.columns) - 2)
    mlflow.log_metric("total_samples", len(df_combined))
    
    print(f"Features saved to: {args.features_output}")
    mlflow.end_run()

if __name__ == "__main__":
    main()