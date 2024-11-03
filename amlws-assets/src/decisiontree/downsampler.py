# downsampler.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import decimate
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("downsampler")
    parser.add_argument("--input_data", type=str, help="Path to input data directory")
    parser.add_argument("--output_data", type=str, help="Path to output data directory")
    args = parser.parse_args()
    return args

def downsample_data(data):
    """Downsample data by factor of 2 using decimation."""
    try:
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            return decimate(data, 2)  # Includes anti-aliasing filter
        else:
            logger.warning(f"Invalid data type for downsampling: {type(data)}")
            return np.zeros(len(data) // 2)
    except Exception as e:
        logger.warning(f"Error downsampling data: {str(e)}")
        return np.zeros(len(data) // 2)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
            downsampled_data = downsample_data(channel_data)
            processed_row[channel] = downsampled_data
            
            # Log downsample ratio for first window
            if idx == 0:
                mlflow.log_metric(f"{channel}_downsample_ratio", 
                                len(downsampled_data) / len(channel_data))
            
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
            
            # Log pre-downsample statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                pre_downsample_data = np.vstack(df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_pre_downsample_mean", 
                                np.mean(pre_downsample_data))
                mlflow.log_metric(f"{group}_{channel}_pre_downsample_std", 
                                np.std(pre_downsample_data))
            
            # Process data
            downsampled_df = process_dataframe(df)
            
            # Save processed data
            output_file = output_path / f"{group}.parquet"
            downsampled_df.to_parquet(output_file)
            
            # Log metrics
            group_time = time.time() - group_start_time
            mlflow.log_metric(f"{group}_windows_processed", len(downsampled_df))
            mlflow.log_metric(f"{group}_processing_time", group_time)
            
            # Log post-downsample statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                post_downsample_data = np.vstack(downsampled_df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_post_downsample_mean", 
                                np.mean(post_downsample_data))
                mlflow.log_metric(f"{group}_{channel}_post_downsample_std", 
                                np.std(post_downsample_data))
        
        # Log execution metrics
        total_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("downsampling_status", 1)
        
        logger.info(f"Downsampling completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in downsampling: {str(e)}")
        mlflow.log_metric("downsampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()