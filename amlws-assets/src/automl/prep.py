import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--filtered_data", type=str, help="Path to filtered data")
    parser.add_argument("--processed_data", type=str, help="Path to processed data output")
    args = parser.parse_args()
    return args

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        start_time = time.time()
        
        # Load filtered data
        filtered_data = np.load(args.filtered_data)
        channel_names = ['af7', 'af8', 'tp9', 'tp10']
        
        processed_data = {}
        for group in ['EEG_windows_Non_remission', 'EEG_windows_Remission']:
            processed_frames = []
            
            for j in range(filtered_data[group].shape[0]):
                data_dict = {'Participant': f'P{j+1:03d}'}
                
                for i, channel in enumerate(channel_names):
                    data_dict[channel] = filtered_data[group][j, :, i]
                
                processed_frames.append(pd.DataFrame([data_dict]))
            
            df_group = pd.concat(processed_frames, ignore_index=True)
            processed_data[group] = df_group
            
            # Log metrics for each group
            mlflow.log_metric(f"{group}_samples", len(df_group))
            for channel in channel_names:
                channel_data = np.vstack(df_group[channel].values)
                mlflow.log_metric(f"{group}_{channel}_mean", np.mean(channel_data))
                mlflow.log_metric(f"{group}_{channel}_std", np.std(channel_data))
                mlflow.log_metric(f"{group}_{channel}_max", np.max(channel_data))
                mlflow.log_metric(f"{group}_{channel}_min", np.min(channel_data))
        
        # Save processed data
        output_path = Path(args.processed_data)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_data['EEG_windows_Non_remission'].to_parquet(output_path / "non_remission.parquet")
        processed_data['EEG_windows_Remission'].to_parquet(output_path / "remission.parquet")
        
        process_time = time.time() - start_time
        mlflow.log_metric("final_processing_time_seconds", process_time)
        mlflow.log_metric("processing_status", 1)
        
    except Exception as e:
        logger.error(f"Error in final processing: {str(e)}")
        mlflow.log_metric("processing_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()