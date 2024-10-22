# data_loader.py
import argparse
from pathlib import Path
import scipy.io
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("data_loader")
    parser.add_argument("--input_data", type=str, help="Path to input .mat file")
    parser.add_argument("--output_data", type=str, help="Path to output data")
    args = parser.parse_args()
    return args

def main():
    mlflow.start_run()
    args = parse_args()
    
    try:
        logger.info(f"Loading MATLAB file: {args.input_data}")
        start_time = time.time()
        mat_data = scipy.io.loadmat(args.input_data)
        load_time = time.time() - start_time
        
        # Log data statistics
        mlflow.log_metric("data_load_time_seconds", load_time)
        mlflow.log_metric("non_remission_participants", mat_data['EEG_windows_Non_remission'].shape[0])
        mlflow.log_metric("remission_participants", mat_data['EEG_windows_Remission'].shape[0])
        
        # Save intermediate data
        output_path = Path(args.output_data)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "eeg_data.npy", mat_data)
        
        logger.info("Data loading completed successfully")
        mlflow.log_metric("data_loading_status", 1)  # 1 for success
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        mlflow.log_metric("data_loading_status", 0)  # 0 for failure
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()