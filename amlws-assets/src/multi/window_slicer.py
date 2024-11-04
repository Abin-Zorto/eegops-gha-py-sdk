# window_slicer.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("window_slicer")
    parser.add_argument("--input_data", type=str, help="Path to input data directory")
    parser.add_argument("--output_data", type=str, help="Path to output data directory")
    parser.add_argument("--window_seconds", type=int, default=2, help="Window length in seconds")
    parser.add_argument("--sampling_rate", type=int, default=256, help="Sampling rate in Hz")
    args = parser.parse_args()
    return args

def slice_window(data: np.ndarray, window_length: int) -> np.ndarray:
    """Slice array to specified window length."""
    if len(data) >= window_length:
        return data[:window_length]
    else:
        logger.warning(f"Data length {len(data)} shorter than window length {window_length}")
        return np.pad(data, (0, window_length - len(data)), 'constant')

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        
        # Calculate desired window length
        window_length = args.window_seconds * args.sampling_rate
        logger.info(f"Window length: {window_length} points "
                   f"({args.window_seconds} seconds at {args.sampling_rate} Hz)")
        
        input_path = Path(args.input_data)
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        channels = ['af7', 'af8', 'tp9', 'tp10']
        
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
                
                window_dict = {'Participant': row['Participant']}
                
                # Slice each channel
                for channel in channels:
                    channel_data = np.array(row[channel])
                    sliced_data = slice_window(channel_data, window_length)
                    window_dict[channel] = sliced_data
                    
                    # Log length check for first window
                    if idx == 0:
                        mlflow.log_metric(f"{group}_{channel}_window_length", 
                                        len(sliced_data))
                
                processed_data.append(window_dict)
            
            # Create DataFrame and save
            output_df = pd.DataFrame(processed_data)
            output_file = output_path / f"{group}.parquet"
            output_df.to_parquet(output_file)
            
            # Log metrics
            mlflow.log_metric(f"{group}_windows_processed", len(output_df))
        
        # Log execution metrics
        process_time = time.time() - start_time
        mlflow.log_metric("total_processing_time", process_time)
        mlflow.log_metric("window_slicing_status", 1)
        
        logger.info(f"Window slicing completed in {process_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in window slicing: {str(e)}")
        mlflow.log_metric("window_slicing_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()