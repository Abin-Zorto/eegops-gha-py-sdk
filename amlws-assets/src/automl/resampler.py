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
    """Resample a single channel of data."""
    try:
        return resample(data, desired_length)
    except Exception as e:
        logger.warning(f"Error resampling data: {str(e)}")
        return np.zeros(desired_length)

def resample_participant_data(data_df: pd.DataFrame, desired_length: int) -> pd.DataFrame:
    """Resample all channels for a participant's data."""
    channels = ['af7', 'af8', 'tp9', 'tp10']
    resampled_data = {'Participant': data_df['Participant'].iloc[0]}
    
    for channel in channels:
        channel_data = np.array(data_df[channel])
        resampled_data[channel] = resample_data(channel_data, desired_length)
    
    return pd.DataFrame([resampled_data])

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        input_path = Path(args.input_data)
        
        # Load data for both groups
        logger.info("Loading non-remission data...")
        df_non_remission = pd.read_parquet(input_path / "non_remission.parquet")
        logger.info("Loading remission data...")
        df_remission = pd.read_parquet(input_path / "remission.parquet")
        
        mlflow.log_metric("data_load_time", time.time() - start_time)
        
        # Process each group
        resampled_dfs = {}
        for group_name, df in [("non_remission", df_non_remission), 
                             ("remission", df_remission)]:
            logger.info(f"Processing {group_name} group...")
            
            resampled_windows = []
            for _, row in df.iterrows():
                # Convert row to DataFrame to maintain structure
                row_df = pd.DataFrame([row])
                resampled_window = resample_participant_data(row_df, args.desired_length)
                resampled_windows.append(resampled_window)
            
            resampled_df = pd.concat(resampled_windows, ignore_index=True)
            resampled_dfs[group_name] = resampled_df
            
            # Log metrics for this group
            mlflow.log_metric(f"{group_name}_total_windows", len(resampled_df))
            
            # Log data statistics
            for channel in ['af7', 'af8', 'tp9', 'tp10']:
                channel_data = np.vstack(resampled_df[channel].values)
                mlflow.log_metric(f"{group_name}_{channel}_mean", np.mean(channel_data))
                mlflow.log_metric(f"{group_name}_{channel}_std", np.std(channel_data))
        
        # Save resampled data
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving resampled data...")
        resampled_dfs["non_remission"].to_parquet(output_path / "non_remission.parquet")
        resampled_dfs["remission"].to_parquet(output_path / "remission.parquet")
        
        # Log execution metrics
        process_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", process_time)
        mlflow.log_metric("resampling_status", 1)
        
        logger.info(f"Resampling completed successfully in {process_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        mlflow.log_metric("resampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()