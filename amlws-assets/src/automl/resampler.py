import argparse
from pathlib import Path
import numpy as np
from scipy.signal import resample
import mlflow
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("resampler")
    parser.add_argument("--input_data", type=str, help="Path to input data")
    parser.add_argument("--output_data", type=str, help="Path to output data")
    parser.add_argument("--desired_length", type=int, default=2560)
    args = parser.parse_args()
    return args

def resample_data(data, desired_length):
    return resample(data, desired_length)

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        # Load data
        input_data = np.load(args.input_data)
        start_time = time.time()
        
        resampled_data = {}
        for group in ['EEG_windows_Non_remission', 'EEG_windows_Remission']:
            resampled_windows = []
            total_windows = 0
            
            for j in range(input_data[group].shape[0]):
                for k in range(input_data[group][j, 0].shape[1]):
                    window_data = input_data[group][j, 0][0, k]
                    resampled_window = resample_data(window_data, args.desired_length)
                    resampled_windows.append(resampled_window)
                    total_windows += 1
            
            resampled_data[group] = np.array(resampled_windows)
            mlflow.log_metric(f"{group}_total_windows", total_windows)
        
        process_time = time.time() - start_time
        mlflow.log_metric("resampling_time_seconds", process_time)
        
        # Save resampled data
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "resampled_data.npy", resampled_data)
        
        mlflow.log_metric("resampling_status", 1)
        
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        mlflow.log_metric("resampling_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
