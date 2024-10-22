# extract_features.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import nolds
import mlflow
import logging
import time
from typing import Dict, Any
from scipy.stats import skew, kurtosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("extract_features")
    parser.add_argument("--processed_data", type=str, help="Path to processed data")
    parser.add_argument("--features_output", type=str, help="Path to features output")
    parser.add_argument("--sampling_rate", type=int, default=256)
    args = parser.parse_args()
    return args

def band_power(data: np.ndarray, sf: int, band: tuple, window_length: int) -> float:
    """Calculate power in a specific frequency band."""
    try:
        f, Pxx = welch(data, sf, nperseg=window_length)
        ind_min = np.argmax(f > band[0]) - 1
        ind_max = np.argmax(f > band[1]) - 1
        return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    except Exception as e:
        logger.warning(f"Error calculating band power: {str(e)}")
        return np.nan

def compute_complexity_measures(data: np.ndarray) -> Dict[str, float]:
    """Compute various complexity measures with error handling."""
    features = {}
    try:
        features['hfd'] = nolds.hfd(data, Kmax=10)
        features['corr_dim'] = nolds.corr_dim(data, emb_dim=10)
        features['hurst'] = nolds.hurst_rs(data)
        features['lyap_r'] = nolds.lyap_r(data, emb_dim=10)
        features['dfa'] = nolds.dfa(data)
    except Exception as e:
        logger.warning(f"Error computing complexity measures: {str(e)}")
        features = {k: np.nan for k in ['hfd', 'corr_dim', 'hurst', 'lyap_r', 'dfa']}
    return features

def compute_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """Compute statistical features of the signal."""
    try:
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'zero_crossings': np.sum(np.diff(np.signbit(data))),
            'peak_to_peak': np.ptp(data)
        }
    except Exception as e:
        logger.warning(f"Error computing statistical features: {str(e)}")
        return {k: np.nan for k in ['mean', 'std', 'skewness', 'kurtosis', 'rms', 
                                   'zero_crossings', 'peak_to_peak']}

def compute_features(channel_data: np.ndarray, sf: int) -> Dict[str, Any]:
    """Compute all features for a channel."""
    features = {}
    
    # Frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 60)
    }
    
    # Calculate band powers
    for band_name, band_range in bands.items():
        features[f'bp_{band_name}'] = band_power(channel_data, sf, band_range, len(channel_data))
    
    # Add complexity measures
    features.update(compute_complexity_measures(channel_data))
    
    # Add statistical features
    features.update(compute_statistical_features(channel_data))
    
    return features

def extract_features_from_df(df: pd.DataFrame, sf: int) -> pd.DataFrame:
    """Extract features from all channels in the DataFrame."""
    all_features = []
    channels = ['af7', 'af8', 'tp9', 'tp10']
    
    feature_computation_times = []
    missing_feature_counts = []
    
    for index, row in df.iterrows():
        if index % 10 == 0:
            logger.info(f"Processing row {index+1}/{len(df)}")
        
        features = {'Participant': row['Participant']}
        
        for channel in channels:
            start_time = time.time()
            channel_data = np.array(row[channel])
            
            # Compute features
            channel_features = compute_features(channel_data, sf)
            
            # Track missing features
            missing_count = sum(1 for v in channel_features.values() if pd.isna(v))
            missing_feature_counts.append(missing_count)
            
            # Add channel prefix to features
            for feature_name, value in channel_features.items():
                features[f'{channel}_{feature_name}'] = value
            
            feature_computation_times.append(time.time() - start_time)
        
        all_features.append(features)
    
    # Log feature extraction metrics
    mlflow.log_metric("mean_feature_computation_time", np.mean(feature_computation_times))
    mlflow.log_metric("max_feature_computation_time", np.max(feature_computation_times))
    mlflow.log_metric("mean_missing_features_per_window", np.mean(missing_feature_counts))
    
    return pd.DataFrame(all_features)

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        
        # Load processed data
        input_path = Path(args.processed_data)
        df_non_remission = pd.read_parquet(input_path / "non_remission.parquet")
        df_remission = pd.read_parquet(input_path / "remission.parquet")
        
        # Extract features
        logger.info("Extracting features for non-remission group...")
        df_non_remission_features = extract_features_from_df(df_non_remission, args.sampling_rate)
        df_non_remission_features['Remission'] = 0
        
        logger.info("Extracting features for remission group...")
        df_remission_features = extract_features_from_df(df_remission, args.sampling_rate)
        df_remission_features['Remission'] = 1
        
        # Combine datasets
        df_combined = pd.concat([df_non_remission_features, df_remission_features], 
                              ignore_index=True)
        
        # Calculate and log feature statistics
        feature_cols = df_combined.drop(['Participant', 'Remission'], axis=1).columns
        for col in feature_cols:
            mlflow.log_metric(f"feature_{col}_mean", df_combined[col].mean())
            mlflow.log_metric(f"feature_{col}_std", df_combined[col].std())
            mlflow.log_metric(f"feature_{col}_missing_pct", 
                            (df_combined[col].isna().sum() / len(df_combined)) * 100)
        
        # Save features
        output_path = Path(args.features_output)
        output_path.mkdir(parents=True, exist_ok=True)
        df_combined.to_parquet(output_path / "features.parquet")
        
        # Log execution metrics
        mlflow.log_metric("total_features", len(feature_cols))
        mlflow.log_metric("total_samples", len(df_combined))
        mlflow.log_metric("feature_extraction_time", time.time() - start_time)
        mlflow.log_metric("feature_extraction_status", 1)
        
        logger.info(f"Features saved to: {args.features_output}")
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        mlflow.log_metric("feature_extraction_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()