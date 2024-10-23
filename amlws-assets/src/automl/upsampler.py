import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("upsampler")
    parser.add_argument("--input_data", type=str, help="Path to input data directory")
    parser.add_argument("--output_data", type=str, help="Path to output data directory")
    parser.add_argument("--upsampling_factor", type=int, default=2, 
                       help="Factor by which to increase the sampling rate")
    args = parser.parse_args()
    return args

def upsample_data(data, upsampling_factor):
    """Upsample a single array of data using linear interpolation."""
    try:
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            # Create original time points
            x = np.arange(len(data))
            # Create interpolation function
            f = interp1d(x, data, kind='linear')
            # Create new time points
            x_new = np.linspace(0, len(data) - 1, len(data) * upsampling_factor)
            # Apply interpolation
            return f(x_new)
        else:
            logger.warning(f"Invalid data type for upsampling: {type(data)}")
            return np.zeros(len(data) * upsampling_factor)
    except Exception as e:
        logger.warning(f"Error upsampling data: {str(e)}")
        return np.zeros(len(data) * upsampling_factor)

def process_dataframe(df: pd.DataFrame, upsampling_factor: int) -> pd.DataFrame:
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
            original_length = len(channel_data) if isinstance(channel_data, (list, np.ndarray)) else 1
            upsampled_data = upsample_data(channel_data, upsampling_factor)
            
            # Log upsampling ratio for verification
            if idx == 0:  # Log only for first window
                mlflow.log_metric(f"{channel}_upsampling_ratio", 
                                len(upsampled_data) / original_length)
            
            processed_row[channel] = upsampled_data
            
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
            
            # Log original data statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                original_data = np.vstack(df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_original_mean", np.mean(original_data))
                mlflow.log_metric(f"{group}_{channel}_original_std", np.std(original_data))
            
            # Process data
            upsampled_df = process_dataframe(df, args.upsampling_factor)
            
            # Save processed data
            output_file = output_path / f"{group}.parquet"
            upsampled_df.to_parquet(output_file)
            
            # Log metrics
            group_time = time.time() - group_start_time
            mlflow.log_metric(f"{group}_windows_processed", len(upsampled_df))
            mlflow.log_metric(f"{group}_processing_time", group_time)
            
            # Log upsampled data statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                upsampled_data = np.vstack(upsampled_df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_upsampled_mean", np.mean(upsampled_data))
                mlflow.log_metric(f"{group}_{channel}_upsampled_std", np.std(upsampled_data))
        
        # Log execution metrics
        total_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("upsampling_status", 1)
        mlflow.log_param("upsampling_factor", args.upsampling_factor)
        
        logger.info(f"Upsampling completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in upsampling: {str(e)}")
        mlflow.log_metric("upsampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()