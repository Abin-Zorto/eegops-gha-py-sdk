import argparse
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
import mlflow
import logging
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("filter")
    parser.add_argument("--input_data", type=str, help="Path to input data")
    parser.add_argument("--output_data", type=str, help="Path to output data")
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--cutoff_frequency", type=int, default=60)
    parser.add_argument("--filter_order", type=int, default=4)
    args = parser.parse_args()
    return args

def design_filter(sampling_rate, cutoff_frequency, filter_order):
    nyquist = 0.5 * sampling_rate
    low_cutoff = cutoff_frequency / nyquist
    return butter(filter_order, low_cutoff, btype='low')

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        # Load resampled data
        input_data = np.load(args.input_data)
        start_time = time.time()
        
        # Design filter
        b, a = design_filter(args.sampling_rate, args.cutoff_frequency, args.filter_order)
        mlflow.log_param("filter_order", args.filter_order)
        mlflow.log_param("cutoff_frequency", args.cutoff_frequency)
        
        filtered_data = {}
        for group in ['EEG_windows_Non_remission', 'EEG_windows_Remission']:
            filtered_windows = []
            signal_powers = []
            noise_powers = []
            
            for window in input_data[group]:
                # Apply filter
                filtered_window = filtfilt(b, a, window)
                filtered_windows.append(filtered_window)
                
                # Calculate signal and noise power
                signal_power = np.mean(np.square(filtered_window))
                noise_power = np.mean(np.square(window - filtered_window))
                signal_powers.append(signal_power)
                noise_powers.append(noise_power)
            
            filtered_data[group] = np.array(filtered_windows)
            
            # Log metrics for each group
            mlflow.log_metric(f"{group}_mean_signal_power", np.mean(signal_powers))
            mlflow.log_metric(f"{group}_mean_noise_power", np.mean(noise_powers))
            mlflow.log_metric(f"{group}_mean_snr", 10 * np.log10(np.mean(signal_powers) / np.mean(noise_powers)))
        
        process_time = time.time() - start_time
        mlflow.log_metric("filtering_time_seconds", process_time)
        
        # Save filtered data
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "filtered_data.npy", filtered_data)
        
        mlflow.log_metric("filtering_status", 1)
        
    except Exception as e:
        logger.error(f"Error in filtering: {str(e)}")
        mlflow.log_metric("filtering_status", 0)
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()