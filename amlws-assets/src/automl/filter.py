import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("filter")
    parser.add_argument("--input_data", type=str, help="Path to input data directory")
    parser.add_argument("--output_data", type=str, help="Path to output data directory")
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--cutoff_frequency", type=int, default=60)
    parser.add_argument("--filter_order", type=int, default=4)
    args = parser.parse_args()
    return args

def design_filter(sampling_rate, cutoff_frequency, filter_order):
    """Design Butterworth low-pass filter."""
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist
    return butter(filter_order, normalized_cutoff, btype='low')

def apply_filter(data, b, a):
    """Apply filter to a single array of data."""
    try:
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            # Apply filter with zero-phase forward and reverse digital filtering
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        else:
            logger.warning(f"Invalid data type for filtering: {type(data)}")
            return np.zeros_like(data)
    except Exception as e:
        logger.warning(f"Error filtering data: {str(e)}")
        return np.zeros_like(data)

def process_dataframe(df: pd.DataFrame, b: np.ndarray, a: np.ndarray) -> pd.DataFrame:
    """Process all channels in a DataFrame."""
    results = []
    channels = ['af7', 'af8', 'tp9', 'tp10']
    
    total_windows = len(df)
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing window {idx}/{total_windows}")
            
        processed_row = {'Participant': row['Participant']}
        
        for channel in channels:
            channel_data = row[channel]
            filtered_data = apply_filter(channel_data, b, a)
            processed_row[channel] = filtered_data
            
            # Calculate signal power reduction for first window
            if idx == 0:
                original_power = np.mean(np.square(channel_data))
                filtered_power = np.mean(np.square(filtered_data))
                mlflow.log_metric(f"{channel}_power_reduction_db", 
                                10 * np.log10(filtered_power / original_power))
            
        results.append(processed_row)
    
    return pd.DataFrame(results)

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        input_path = Path(args.input_data)
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Design filter
        b, a = design_filter(args.sampling_rate * 2,  # Account for upsampled rate
                           args.cutoff_frequency, 
                           args.filter_order)
        
        mlflow.log_params({
            "sampling_rate": args.sampling_rate * 2,
            "cutoff_frequency": args.cutoff_frequency,
            "filter_order": args.filter_order
        })
        
        # Process each group
        for group in ['non_remission', 'remission']:
            logger.info(f"Processing {group} data...")
            group_start_time = time.time()
            
            # Load data
            input_file = input_path / f"{group}.parquet"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {len(df)} windows for {group}")
            
            # Log pre-filtering statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                pre_filter_data = np.vstack(df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_pre_filter_mean", 
                                np.mean(pre_filter_data))
                mlflow.log_metric(f"{group}_{channel}_pre_filter_std", 
                                np.std(pre_filter_data))
            
            # Apply filtering
            filtered_df = process_dataframe(df, b, a)
            
            # Save processed data
            output_file = output_path / f"{group}.parquet"
            filtered_df.to_parquet(output_file)
            
            # Log metrics
            group_time = time.time() - group_start_time
            mlflow.log_metric(f"{group}_windows_processed", len(filtered_df))
            mlflow.log_metric(f"{group}_processing_time", group_time)
            
            # Log post-filtering statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                post_filter_data = np.vstack(filtered_df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_post_filter_mean", 
                                np.mean(post_filter_data))
                mlflow.log_metric(f"{group}_{channel}_post_filter_std", 
                                np.std(post_filter_data))
        
        # Log execution metrics
        total_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("filtering_status", 1)
        
        logger.info(f"Filtering completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in filtering: {str(e)}")
        mlflow.log_metric("filtering_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
