# data_loader.py
import argparse
from pathlib import Path
import scipy.io
import mlflow
import logging
import time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("data_loader")
    parser.add_argument("--input_data", type=str, help="Path to input .mat file")
    parser.add_argument("--output_data", type=str, help="Path to output data")
    args = parser.parse_args()
    return args

def process_eeg_data(eeg_data, group_name):
    """Process EEG data into DataFrame format."""
    processed_frames = []
    channel_names = ['af7', 'af8', 'tp9', 'tp10']
    
    for j in range(eeg_data.shape[0]):
        participant = eeg_data[j, 1][0][:4]
        
        for k in range(eeg_data[j, 0].shape[1]):
            data_dict = {'Participant': participant}
            
            for i, channel in enumerate(channel_names):
                data_dict[channel] = eeg_data[j, 0][0, k][:, i]
            
            processed_frames.append(pd.DataFrame([data_dict]))
    
    return pd.concat(processed_frames, ignore_index=True)

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        logger.info(f"Loading MATLAB file: {args.input_data}")
        start_time = time.time()
        mat_data = scipy.io.loadmat(args.input_data)
        load_time = time.time() - start_time
        
        # Process each group
        logger.info("Processing non-remission group...")
        df_non_remission = process_eeg_data(
            mat_data['EEG_windows_Non_remission'],
            'non_remission'
        )
        
        logger.info("Processing remission group...")
        df_remission = process_eeg_data(
            mat_data['EEG_windows_Remission'],
            'remission'
        )
        
        # Log data statistics
        mlflow.log_metric("data_load_time_seconds", load_time)
        mlflow.log_metric("non_remission_participants", len(df_non_remission['Participant'].unique()))
        mlflow.log_metric("remission_participants", len(df_remission['Participant'].unique()))
        mlflow.log_metric("non_remission_windows", len(df_non_remission))
        mlflow.log_metric("remission_windows", len(df_remission))
        
        # Save processed data
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df_non_remission.to_parquet(output_path / "non_remission.parquet")
        df_remission.to_parquet(output_path / "remission.parquet")
        
        logger.info("Data loading and processing completed successfully")
        mlflow.log_metric("data_loading_status", 1)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        mlflow.log_metric("data_loading_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
