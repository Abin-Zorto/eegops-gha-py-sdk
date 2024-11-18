import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import nolds
import mlflow
import logging
import time
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("extract_features")
    parser.add_argument("--processed_data", type=str, help="Path to processed data")
    parser.add_argument("--features_output", type=str, help="Path to features output")
    parser.add_argument("--sampling_rate", type=int, default=256)
    args = parser.parse_args()
    return args

def validate_data(data: np.ndarray) -> Tuple[bool, str]:
    """Validate data before feature extraction."""
    try:
        if not isinstance(data, np.ndarray):
            return False, "Data is not a numpy array"
        if len(data) == 0:
            return False, "Empty data array"
        if np.any(np.isnan(data)):
            return False, "Data contains NaN values"
        if np.any(np.isinf(data)):
            return False, "Data contains infinite values"
        return True, "Data validation passed"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def band_power(data: np.ndarray, sf: int, band: tuple, window_length: int) -> float:
    """Calculate power in a specific frequency band with validation."""
    try:
        valid, message = validate_data(data)
        if not valid:
            logger.warning(f"Band power calculation: {message}")
            return np.nan
            
        f, Pxx = welch(data, sf, nperseg=min(window_length, len(data)))
        ind_min = np.argmax(f > band[0]) - 1
        ind_max = np.argmax(f > band[1]) - 1
        power = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
        
        # Validate power calculation
        if np.isnan(power) or np.isinf(power):
            logger.warning("Invalid power value calculated")
            return np.nan
            
        return power
    except Exception as e:
        logger.warning(f"Error calculating band power: {str(e)}")
        return np.nan

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

    # Convert x to a numpy array for linear fitting
    x = np.array(x)
    # Perform a linear fit to the log-log values to determine the slope
    a, b = np.polyfit(x[:, 0], x[:, 1], 1)
    # Return the slope, which is the estimate of the fractal dimension
    return a

def compute_entropy_features(data: np.ndarray) -> Dict[str, float]:
    """Compute entropy-based features."""
    try:
        valid, message = validate_data(data)
        if not valid:
            logger.warning(f"Entropy calculation: {message}")
            return {k: np.nan for k in ['sample_entropy', 'spectral_entropy']}
            
        # Sample entropy
        sample_entropy = nolds.sampen(data)
        
        # Spectral entropy
        f, Pxx = welch(data, nperseg=min(len(data), 256))
        psd_norm = Pxx / np.sum(Pxx)
        spectral_entropy = entropy(psd_norm)
        
        return {
            'sample_entropy': sample_entropy,
            'spectral_entropy': spectral_entropy
        }
    except Exception as e:
        logger.warning(f"Error computing entropy features: {str(e)}")
        return {k: np.nan for k in ['sample_entropy', 'spectral_entropy']}

def compute_complexity_measures(data: np.ndarray) -> Dict[str, float]:
    """Compute various complexity measures with enhanced validation."""
    try:
        valid, message = validate_data(data)
        if not valid:
            logger.warning(f"Complexity calculation: {message}")
            return {k: np.nan for k in ['hfd', 'corr_dim', 'hurst', 'lyap_r', 'dfa']}
        
        # Ensure data is float64 type and handle NaN/inf values
        data = np.array(data, dtype=np.float64)
        if len(data) < 50:  # Minimum length for reliability
            logger.warning("Data length too short for complexity measures")
            return {k: np.nan for k in ['hfd', 'corr_dim', 'hurst', 'lyap_r', 'dfa']}
        
        features = {}
        
        # Higuchi Fractal Dimension
        try:
            features['hfd'] = hfd(data, Kmax=10)
        except Exception as e:
            logger.warning(f"HFD calculation failed: {str(e)}")
            features['hfd'] = np.nan
        
        # Correlation Dimension
        try:
            features['corr_dim'] = nolds.corr_dim(data, emb_dim=10)
        except Exception as e:
            logger.warning(f"Correlation dimension calculation failed: {str(e)}")
            features['corr_dim'] = np.nan
        
        # Hurst Exponent
        try:
            features['hurst'] = nolds.hurst_rs(data)
        except Exception as e:
            logger.warning(f"Hurst exponent calculation failed: {str(e)}")
            features['hurst'] = np.nan
        
        # Largest Lyapunov Exponent
        try:
            # Normalize and prepare data
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
            data_norm = np.ascontiguousarray(data_norm, dtype=np.float64)
            
            # Calculate embedding parameters
            emb_dim = 10
            lag = max(1, int(len(data_norm) // 20))  # Use integer division
            
            # Ensure minimum data length
            min_length = (emb_dim - 1) * lag + 1
            if len(data_norm) < min_length:
                logger.warning(f"Data length {len(data_norm)} insufficient for lyap_r calculation")
                features['lyap_r'] = np.nan
            else:
                logger.info(f"Computing lyap_r with emb_dim={emb_dim}, lag={lag}, data_length={len(data_norm)}")
                features['lyap_r'] = nolds.lyap_r(data_norm, emb_dim=emb_dim, lag=lag, min_tsep=lag)
                
                if not np.isfinite(features['lyap_r']):
                    logger.warning(f"Computed lyap_r is not finite: {features['lyap_r']}")
                    features['lyap_r'] = np.nan
                    
        except Exception as e:
            logger.warning(f"Lyapunov exponent calculation failed: {str(e)}")
            features['lyap_r'] = np.nan
        
        # Detrended Fluctuation Analysis
        try:
            features['dfa'] = nolds.dfa(data)
        except Exception as e:
            logger.warning(f"DFA calculation failed: {str(e)}")
            features['dfa'] = np.nan
        
        # Final validation
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = np.nan
                logger.warning(f"Replacing {key} with nan due to non-finite value")
        
        return features
            
    except Exception as e:
        logger.warning(f"Error computing complexity measures: {str(e)}")
        return {k: np.nan for k in ['hfd', 'corr_dim', 'hurst', 'lyap_r', 'dfa']}


def compute_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """Compute statistical features with validation."""
    try:
        valid, message = validate_data(data)
        if not valid:
            logger.warning(f"Statistical calculation: {message}")
            return {k: np.nan for k in ['mean', 'std', 'skewness', 'kurtosis', 'rms',
                                      'zero_crossings', 'peak_to_peak', 'variance',
                                      'mean_abs_deviation']}
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'variance': np.var(data),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'zero_crossings': np.sum(np.diff(np.signbit(data))),
            'peak_to_peak': np.ptp(data),
            'mean_abs_deviation': np.mean(np.abs(data - np.mean(data)))
        }
    except Exception as e:
        logger.warning(f"Error computing statistical features: {str(e)}")
        return {k: np.nan for k in ['mean', 'std', 'skewness', 'kurtosis', 'rms',
                                   'zero_crossings', 'peak_to_peak', 'variance',
                                   'mean_abs_deviation']}

def compute_features(channel_data: np.ndarray, sf: int) -> Dict[str, Any]:
    """Compute all features for a channel with comprehensive logging."""
    # Ensure data is properly formatted
    try:
        if not isinstance(channel_data, np.ndarray):
            logger.warning("Data is not a NumPy array. Converting...")
            channel_data = np.array(channel_data, dtype=np.float64)
        elif channel_data.dtype != np.float64:
            logger.warning(f"Data type is {channel_data.dtype}. Converting to float64...")
            channel_data = channel_data.astype(np.float64)
        if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
            logger.warning("Channel data contains NaN or Inf values")
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error converting channel data: {str(e)}")
        return {}, {}

    features = {}
    feature_computation_times = {}
    
    # Frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 60)
    }
    
    # Calculate band powers
    start_time = time.time()
    for band_name, band_range in bands.items():
        features[f'bp_{band_name}'] = band_power(channel_data, sf, band_range, len(channel_data))
    feature_computation_times['band_powers'] = time.time() - start_time
    
    # Add complexity measures
    start_time = time.time()
    features.update(compute_complexity_measures(channel_data))
    feature_computation_times['complexity'] = time.time() - start_time
    
    # Add statistical features
    #start_time = time.time()
    #features.update(compute_statistical_features(channel_data))
    #feature_computation_times['statistical'] = time.time() - start_time
    
    # Add entropy features
    start_time = time.time()
    features.update(compute_entropy_features(channel_data))
    feature_computation_times['entropy'] = time.time() - start_time
    
    
    return features, feature_computation_times

def extract_features_from_df(df: pd.DataFrame, sf: int) -> pd.DataFrame:
    """Extract features from all channels with comprehensive logging."""
    all_features = []
    channels = ['af7', 'af8', 'tp9', 'tp10']
    
    feature_timings = {channel: {} for channel in channels}
    missing_features = {channel: 0 for channel in channels}
    
    total_windows = len(df)
    for index, row in df.iterrows():
        if index % 10 == 0:
            logger.info(f"Processing window {index+1}/{total_windows}")
        
        features = {'Participant': row['Participant']}
        
        for channel in channels:
            channel_data = np.array(row[channel])
            channel_features, computation_times = compute_features(channel_data, sf)
            
            # Track timing statistics
            for feature_type, time_taken in computation_times.items():
                if feature_type not in feature_timings[channel]:
                    feature_timings[channel][feature_type] = []
                feature_timings[channel][feature_type].append(time_taken)
            
            # Track missing features
            missing_count = sum(1 for v in channel_features.values() if pd.isna(v))
            missing_features[channel] += missing_count
            
            # Add channel prefix to features
            for feature_name, value in channel_features.items():
                features[f'{channel}_{feature_name}'] = value
        
        all_features.append(features)
    
    # Log detailed metrics
    for channel in channels:
        for feature_type, timings in feature_timings[channel].items():
            mlflow.log_metric(f"{channel}_{feature_type}_mean_time", np.mean(timings))
            mlflow.log_metric(f"{channel}_{feature_type}_max_time", np.max(timings))
        mlflow.log_metric(f"{channel}_missing_features", missing_features[channel])
    
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
        
        # Create MLTable file
        mltable_content = """
        paths:
          - file: ./features.parquet
        transformations:
          - read_parquet:
              include_path_column: false
        """

        # Write MLTable file
        with open(output_path / "MLTable", "w") as f:
            f.write(mltable_content)
        
        # Log execution metrics
        total_time = time.time() - start_time
        mlflow.log_metric("total_features", len(feature_cols))
        mlflow.log_metric("total_samples", len(df_combined))
        mlflow.log_metric("feature_extraction_time", total_time)
        mlflow.log_metric("features_per_second", len(df_combined) / total_time)
        mlflow.log_metric("feature_extraction_status", 1)
        
        logger.info(f"Features saved to: {args.features_output}")
        logger.info(f"Extracted {len(feature_cols)} features for {len(df_combined)} samples")
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        mlflow.log_metric("feature_extraction_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
