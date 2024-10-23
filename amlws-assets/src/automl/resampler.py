import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("resampler")
    parser.add_argument("--input_data", type=str, help="Path to input data directory")
    parser.add_argument("--output_data", type=str, help="Path to output data directory")
    parser.add_argument("--desired_length", type=int, default=2560)
    args = parser.parse_args()
    return args

def resample_data(data, desired_length):
    """Resample a single array of data."""
    try:
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            return resample(data, desired_length)
        else:
            logger.warning(f"Invalid data type for resampling: {type(data)}")
            return np.zeros(desired_length)
    except Exception as e:
        logger.warning(f"Error resampling data: {str(e)}")
        return np.zeros(desired_length)

def process_dataframe(df: pd.DataFrame, desired_length: int) -> pd.DataFrame:
    """Process all channels in a DataFrame."""
    results = []
    channels = ['af7', 'af8', 'tp9', 'tp10']
    
    for _, row in df.iterrows():
        if len(results) % 100 == 0:
            logger.info(f"Processing window {len(results)}")
            
        processed_row = {'Participant': row['Participant']}
        
        for channel in channels:
            channel_data = row[channel]
            resampled_data = resample_data(channel_data, desired_length)
            processed_row[channel] = resampled_data
            
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
            
            # Load data
            input_file = input_path / f"{group}.parquet"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {len(df)} windows for {group}")
            
            # Process data
            resampled_df = process_dataframe(df, args.desired_length)
            
            # Save processed data
            output_file = output_path / f"{group}.parquet"
            resampled_df.to_parquet(output_file)
            
            # Log metrics
            mlflow.log_metric(f"{group}_windows_processed", len(resampled_df))
            
            # Log sample statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                channel_data = np.vstack(resampled_df[channel].values)
                mlflow.log_metric(f"{group}_{channel}_mean", np.mean(channel_data))
                mlflow.log_metric(f"{group}_{channel}_std", np.std(channel_data))
        
        # Log execution metrics
        process_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", process_time)
        mlflow.log_metric("resampling_status", 1)
        
        logger.info(f"Resampling completed in {process_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        mlflow.log_metric("resampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()