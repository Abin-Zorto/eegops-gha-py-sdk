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
    parser.add_argument("--upsampling_factor", type=int, help="Upsampling factor")
    args = parser.parse_args()
    return args

def upsample_data(data: np.ndarray, target_length: int) -> np.ndarray:
    """Upsample data to target length using linear interpolation."""
    x = np.arange(len(data))
    interpolator = interp1d(x, data, kind='linear')
    x_new = np.linspace(0, len(data) - 1, target_length)
    return interpolator(x_new)

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        input_path = Path(args.input_data)
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Fixed parameters
        channel_names = ['af7', 'af8', 'tp9', 'tp10']
        target_length = 2560 * args.upsampling_factor  # Upsampled length
        
        # Process each group
        for group in ['non_remission', 'remission']:
            logger.info(f"Processing {group} data...")
            
            # Load data
            input_file = input_path / f"{group}.parquet"
            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {len(df)} windows for {group}")
            
            # Process each window
            processed_data = []
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing window {idx+1}/{len(df)}")
                
                # Initialize data dictionary
                data_dict = {'Participant': row['Participant']}
                
                # Process each channel
                for channel in channel_names:
                    channel_data = np.array(row[channel])
                    upsampled_data = upsample_data(channel_data, target_length)
                    data_dict[channel] = upsampled_data
                
                processed_data.append(data_dict)
            
            # Create DataFrame and save
            output_df = pd.DataFrame(processed_data)
            output_file = output_path / f"{group}.parquet"
            output_df.to_parquet(output_file)
            
            # Log metrics
            mlflow.log_metric(f"{group}_windows_processed", len(output_df))
            
            # Log sample lengths for verification
            for channel in channel_names:
                channel_length = len(output_df[channel].iloc[0])
                mlflow.log_metric(f"{group}_{channel}_length", channel_length)
        
        # Log execution time
        process_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", process_time)
        mlflow.log_metric("upsampling_status", 1)
        
        logger.info(f"Upsampling completed in {process_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in upsampling: {str(e)}")
        mlflow.log_metric("upsampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()